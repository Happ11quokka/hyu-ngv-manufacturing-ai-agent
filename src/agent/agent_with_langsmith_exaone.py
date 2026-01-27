"""
AI Agent with LangSmith Tracing - K-EXAONE Version
K-EXAONE-236B-A23B 모델을 사용하는 반도체 결함 검사 AI Agent
"""

import os
import re
import json
import time
import random
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from langsmith import traceable

# .env 파일에서 환경변수 로드
load_dotenv()

# =========================
# 설정
# =========================
MODEL = "LGAI-EXAONE/K-EXAONE-236B-A23B"

# OpenAI 클라이언트 (Friendli API)
client = OpenAI(
    api_key=os.getenv("FRIENDLI_TOKEN"),
    base_url="https://api.friendli.ai/serverless/v1",
)

# 입력/출력 경로
TEST_CSV_PATH = "./dev.csv"
OUT_PATH = "./submission.csv"

# 대회 라벨 정의
LABEL_NORMAL = 0
LABEL_ABNORMAL = 1

# =========================
# 관찰 항목
# =========================
OBS_ITEMS = [
    ("package_damage", "크랙/파손/깨짐 등 패키지 손상"),
    ("lead_missing_or_broken", "리드 결손/단선"),
    ("solder_bridge_or_blob", "솔더 브리지 또는 납땜 뭉침"),
    ("misalignment_severe", "소자 위치가 과도하게 틀어짐"),
]

KEYS = [k for k, _ in OBS_ITEMS]

# =========================
# 시스템 프롬프트
# =========================
SYSTEM = (
    "너는 반도체 소자 검사 이미지 분석기다.\n"
    "반드시 요청한 JSON만 출력한다. 다른 텍스트는 절대 출력하지 않는다.\n"
)


def build_prompt(strict: bool = False) -> str:
    header = (
        "아래 항목을 이미지에서 관찰해 true/false로 채워 JSON만 출력해.\n"
        "형식은 반드시 아래와 동일해야 한다.\n\n"
        "[이미지 설명]\n"
        "- 갈색 페그보드(구멍 뚫린 판) 위에 TO-220 스타일 반도체 부품(트랜지스터)이 있다.\n"
        "- 부품: 검은 플라스틱 본체 + 아래로 뻗은 3개의 금속 리드(다리)\n"
        "- 정상 상태: 부품이 수직으로 서있고, 3개 리드가 모두 존재하며 비슷한 길이\n\n"
        "[리드(Lead) 정의]\n"
        "- 리드(lead)란 부품 하단에서 아래로 뻗어 있는 은색 또는 회색의 금속 다리들을 의미한다.\n"
        "- 각 리드는 가늘고 금속 재질이며, 전기적으로 기판과 연결되는 단자이다.\n"
        "- 본 이미지에서 보이는 은색(회색) 금속 다리 3개 모두를 리드로 간주한다.\n\n"
        "[중요: 정상으로 판단해야 하는 경우]\n"
        "- 리드가 약간 휘어있어도 3개 모두 존재하고 길이가 유지되면 → 정상\n"
        "- 리드가 살짝 비대칭이어도 서로 닿지 않으면 → 정상\n"
        "- 부품이 살짝 기울어져도 크게 넘어지지 않았으면 → 정상\n\n"
    )
    json_template = "{\n" + \
        ",\n".join([f'  "{k}": false' for k, _ in OBS_ITEMS]) + "\n}"

    if strict:
        rule = "\n\n[판단 기준]\n- 극도로 보수적으로 판단. 확실한 불량만 true.\n- 조금이라도 애매하면 무조건 false.\n\n"
    else:
        rule = "\n\n[판단 기준]\n- 명백한 불량만 true. 애매하면 false.\n- 약간의 휨, 기울어짐은 정상 범위.\n\n"

    criteria = "[검사 항목]\n" + \
        "\n".join([f"- {k}: {desc}" for k, desc in OBS_ITEMS])
    return header + json_template + rule + criteria


PROMPT_NORMAL = build_prompt(strict=False)
PROMPT_STRICT = build_prompt(strict=True)


# =========================
# LangSmith Traceable 함수들
# =========================
@traceable(name="LLM API Call")
def _post_chat(messages, timeout=90):
    """LLM API 호출 - LangSmith에서 추적됨 (K-EXAONE)"""
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            extra_body={
                "parse_reasoning": True,
                "chat_template_kwargs": {
                    "enable_thinking": True
                }
            },
            timeout=timeout,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"API call failed: {e}")


class RefusalError(Exception):
    """LLM이 분석을 거부한 경우"""
    pass


def _safe_json_extract(s: str) -> dict:
    """JSON 파싱 (LLM 출력에서 JSON 추출)"""
    # LLM 거부 응답 감지
    refusal_phrases = ["sorry", "can't assist", "cannot assist", "unable to", "I'm not able"]
    if any(phrase.lower() in s.lower() for phrase in refusal_phrases):
        raise RefusalError(f"LLM refused: {s[:100]}")

    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            return json.loads(m.group(0))
    raise ValueError(f"JSON parse failed: {s[:200]}")


@traceable(name="Observe Image")
def observe(img_url: str, strict: bool = False, verbose: bool = False, max_retries: int = 3) -> dict:
    """Agent Step 1: 이미지 관찰 - LangSmith에서 추적됨 (K-EXAONE)"""
    prompt = PROMPT_STRICT if strict else PROMPT_NORMAL

    for attempt in range(max_retries):
        try:
            content = _post_chat([
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ]},
            ])

            if verbose:
                print(f"    [LLM 원본 응답] {content}")

            obs = _safe_json_extract(content)
            result = {k: bool(obs.get(k, False)) for k in KEYS}

            if verbose:
                print(f"    [파싱 결과] {result}")

            return result

        except RefusalError as e:
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            print(f"  [RefusalError] 정상으로 fallback: {e}")
            return {k: False for k in KEYS}

    return {k: False for k in KEYS}


@traceable(name="Decide Label")
def decide(obs: dict):
    """Agent Step 2: 판단 - LangSmith에서 추적됨"""
    defect_count = sum(1 for v in obs.values() if v)
    label = LABEL_ABNORMAL if defect_count >= 1 else LABEL_NORMAL
    uncertain = (defect_count == 0) or (defect_count == 1)
    return label, uncertain


@traceable(name="Classify Agent", run_type="chain")
def classify_agent(img_url: str, img_id: str = None, max_retries=3, verbose=False) -> int:
    """Agent 전체 파이프라인 - LangSmith에서 추적됨 (K-EXAONE)"""
    for attempt in range(max_retries):
        try:
            # 1) 1차 관찰
            if verbose:
                print(f"  [1차 관찰] strict=False")
            obs1 = observe(img_url, strict=False, verbose=verbose)
            label1, uncertain = decide(obs1)

            # 발견된 결함 출력
            defects1 = [k for k, v in obs1.items() if v]
            if verbose:
                if defects1:
                    print(f"    → 발견된 결함: {defects1}")
                else:
                    print(f"    → 결함 없음")
                print(
                    f"    → 1차 판정: {'불량(1)' if label1 == 1 else '정상(0)'}, 불확실: {uncertain}")

            # 2) 애매하지 않으면 바로 종료
            if not uncertain:
                if verbose:
                    print(f"  [최종 결정] 확실함 → {label1}")
                return label1

            # 3) 애매하면 1회 재검토
            if verbose:
                print(f"  [2차 관찰] strict=True (재검토)")
            obs2 = observe(img_url, strict=True, verbose=verbose)
            label2, _ = decide(obs2)

            defects2 = [k for k, v in obs2.items() if v]
            if verbose:
                if defects2:
                    print(f"    → 발견된 결함: {defects2}")
                else:
                    print(f"    → 결함 없음")
                print(f"    → 2차 판정: {'불량(1)' if label2 == 1 else '정상(0)'}")

            # 4) 재검토에서도 결함이 잡히면 비정상 확정
            if label2 == LABEL_ABNORMAL:
                if verbose:
                    print(f"  [최종 결정] 재검토에서도 결함 발견 → 1 (불량)")
                return LABEL_ABNORMAL

            # 5) 재검토에서 결함 없으면 정상으로 판정
            if verbose:
                print(f"  [최종 결정] 재검토에서 결함 없음 → 0 (정상)")
            return LABEL_NORMAL

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep((0.5 * (2 ** attempt)) + random.uniform(0, 0.2))


@traceable(name="Main Pipeline", run_type="chain")
def main():
    """메인 파이프라인 - 전체 실행이 LangSmith에서 추적됨"""
    print(f"Model: {MODEL}")
    print(f"LangSmith Project: {os.getenv('LANGCHAIN_PROJECT')}")
    print(f"Tracing enabled: {os.getenv('LANGCHAIN_TRACING_V2')}")
    print()

    test_df = pd.read_csv(TEST_CSV_PATH)

    if "id" not in test_df.columns or "img_url" not in test_df.columns:
        raise ValueError(f"columns: {test_df.columns.tolist()}")

    preds = []
    n = len(test_df)

    # verbose 모드 설정 (분석 과정 출력)
    VERBOSE = True

    for i, row in test_df.iterrows():
        _id = row["id"]
        img_url = row["img_url"]

        print(f"\n{'='*50}")
        print(f"[{i+1}/{n}] id={_id}")
        print(f"{'='*50}")

        try:
            label = classify_agent(img_url, img_id=_id, verbose=VERBOSE)
            print(
                f"\n  ★ 최종 결과: {_id} -> {'불량(1)' if label == 1 else '정상(0)'}")
        except Exception as e:
            print(f"\n  ✗ 오류 발생: {e}")
            print(f"  ★ 최종 결과: {_id} -> 정상(0) [fallback]")
            label = LABEL_NORMAL

        preds.append({"id": _id, "label": label})
        time.sleep(0.2)

    out_df = pd.DataFrame(preds, columns=["id", "label"])
    out_df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved: {OUT_PATH}")
    print(out_df.head())
    print(f"\nLangSmith 대시보드: https://smith.langchain.com")
    return out_df


if __name__ == "__main__":
    main()
