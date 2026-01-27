"""
AI Agent with LangSmith Tracing
LangSmith 추적 기능이 추가된 반도체 결함 검사 AI Agent
"""

import os
import re
import json
import time
import random
import requests
import pandas as pd
from dotenv import load_dotenv
from langsmith import traceable

# .env 파일에서 환경변수 로드
load_dotenv()

# =========================
# 설정
# =========================
API_KEY = os.getenv("LUXIA_API_KEY")
BRIDGE_URL = "https://bridge.luxiacloud.com/llm/openai/chat/completions/gpt-4o/create"
MODEL = "gpt-4o"

# 입력/출력 경로
TEST_CSV_PATH = "./dev.csv"
OUT_PATH = "./submission.csv"

# API 호출 헤더
HEADERS = {"apikey": API_KEY, "Content-Type": "application/json"}

# 대회 라벨 정의
LABEL_NORMAL = 0
LABEL_ABNORMAL = 1

# =========================
# 관찰 항목
# =========================
OBS_ITEMS = [
    ("package_damage", "크랙/파손/깨짐 등 패키지 손상"),
    ("lead_missing_or_broken", "리드 결손/단선"),
    ("lead_severe_bend_or_contact", "심한 휨 또는 리드끼리 접촉"),
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
        "형식은 반드시 아래와 동일해야 한다.\n"
    )
    json_template = "{\n" + ",\n".join([f'  "{k}": false' for k, _ in OBS_ITEMS]) + "\n}"

    if strict:
        rule = "\n판단 기준:\n- 매우 보수적으로 판단한다. 애매하면 무조건 false.\n"
    else:
        rule = "\n판단 기준:\n- 아주 명확할 때만 true. 애매하면 false.\n"

    criteria = "\n".join([f"- {k}: {desc}" for k, desc in OBS_ITEMS])
    return header + json_template + rule + criteria


PROMPT_NORMAL = build_prompt(strict=False)
PROMPT_STRICT = build_prompt(strict=True)


# =========================
# LangSmith Traceable 함수들
# =========================
@traceable(name="LLM API Call")
def _post_chat(messages, timeout=90):
    """LLM API 호출 - LangSmith에서 추적됨"""
    payload = {"model": MODEL, "messages": messages, "stream": False}
    r = requests.post(BRIDGE_URL, headers=HEADERS, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"status={r.status_code}, body={r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"].strip()


def _safe_json_extract(s: str) -> dict:
    """JSON 파싱 (LLM 출력에서 JSON 추출)"""
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            return json.loads(m.group(0))
    raise ValueError(f"JSON parse failed: {s[:200]}")


@traceable(name="Observe Image")
def observe(img_url: str, strict: bool = False, verbose: bool = False) -> dict:
    """Agent Step 1: 이미지 관찰 - LangSmith에서 추적됨"""
    prompt = PROMPT_STRICT if strict else PROMPT_NORMAL

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


@traceable(name="Decide Label")
def decide(obs: dict):
    """Agent Step 2: 판단 - LangSmith에서 추적됨"""
    defect_count = sum(1 for v in obs.values() if v)
    label = LABEL_ABNORMAL if defect_count >= 1 else LABEL_NORMAL
    uncertain = (defect_count == 0) or (defect_count == 1)
    return label, uncertain


@traceable(name="Classify Agent", run_type="chain")
def classify_agent(img_url: str, img_id: str = None, max_retries=3, verbose=False) -> int:
    """Agent 전체 파이프라인 - LangSmith에서 추적됨"""
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
                print(f"    → 1차 판정: {'불량(1)' if label1 == 1 else '정상(0)'}, 불확실: {uncertain}")

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
            print(f"\n  ★ 최종 결과: {_id} -> {'불량(1)' if label == 1 else '정상(0)'}")
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
