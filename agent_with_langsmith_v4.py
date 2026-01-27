"""
AI Agent with LangSmith Tracing V3
Two-Stage Prompt 기법을 활용한 반도체 결함 검사 AI Agent
- Stage 1: 관찰 전용 (이미지에서 보이는 것만 JSON으로 출력)
- Stage 2: 판단 + 검증 + 종료조건 (Stage 1 결과 기반)
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
# TEST_CSV_PATH = "./dev_test.csv"  # 테스트용 (DEV_008, 009, 014, 017)
TEST_CSV_PATH = "./dev.csv"

OUT_PATH = "./submission.csv"

# API 호출 헤더
HEADERS = {"apikey": API_KEY, "Content-Type": "application/json"}

# 대회 라벨 정의
LABEL_NORMAL = 0
LABEL_ABNORMAL = 1

# =========================
# Stage 1: 관찰 전용 프롬프트
# =========================
STAGE1_SYSTEM = """You are a visual inspection observer for manufacturing images.
STRICT RULES:
- Do NOT classify as normal/abnormal.
- Report ONLY what is directly visible in the image.
- Output MUST be valid JSON only (no markdown, no extra text).
- Be ACCURATE - report defects when you see them, report normal when normal.

========== IMAGE LAYOUT ==========
1. BLACK BODY (TOP): Black rectangular component
   - Check if it is TILTED or STRAIGHT

2. THREE METAL LEADS (MIDDLE): Silver wires going DOWN from body
   - Left lead, Center lead, Right lead
   - Each should go straight down to its hole

3. THREE TARGET HOLES (BOTTOM): Dark circles at BOTTOM row
   - Each lead must END in its corresponding hole
   - Left lead → left hole, Center → center hole, Right → right hole

4. VERTICAL TRACES: Brown stripes on board
   - Leads should NOT touch these traces
   - If a lead bends and touches a trace = DEFECT

========== DEFECT DETECTION CHECKLIST ==========

CHECK 1 - BODY TILT:
- Is the black body tilted/rotated? → tilt = "mild" or "severe"
- Is it straight/upright? → tilt = "none"

CHECK 2 - TRACE EACH LEAD PATH:
For each lead, follow it from body to bottom and ask:
- Does it go STRAIGHT DOWN to its hole? → "in_hole", "connected"
- Does it BEND AWAY and miss the hole? → "missed_hole", "floating"
- Does it touch a BROWN TRACE instead? → "attached_to_line", touches_vertical_line="yes"

CHECK 3 - LEAD ARRANGEMENT:
- Leads stay SEPARATE, each going to own hole → "parallel"
- Leads CROSS over each other (X pattern) → "crossed"
- Leads TOUCH or MERGE together → "tangled"

CHECK 4 - LEAD TOUCHING TRACE:
- Any lead touching vertical brown trace? → touches_vertical_line = "yes"

========== COMMON DEFECT PATTERNS ==========
Pattern A: Body tilted + lead missing hole (see DEV_008 type)
Pattern B: Leads crossing/tangling together (see DEV_009 type)
Pattern C: Lead bending to touch vertical trace (see DEV_017 type)

========== SURFACE_MARK ==========
- "none" for: reflections, dimples, standard features
- "present" for: obvious debris, damage, solder splatter"""

STAGE1_USER = """Observe the provided image carefully and fill the JSON below.
Do NOT conclude normal/abnormal. Just describe what you see ACCURATELY.

OBSERVATION STEPS:
1. Look at the BLACK BODY - is it tilted or straight?
2. Trace the LEFT LEAD - does it go to the left hole or miss it?
3. Trace the CENTER LEAD - does it go to the center hole or miss it?
4. Trace the RIGHT LEAD - does it go to the right hole or miss it?
5. Check if any leads TOUCH each other (tangled) or CROSS over (crossed)
6. Check if any lead touches a BROWN VERTICAL TRACE

KEY DEFINITIONS:
- "in_hole" + "connected": Lead tip is in its target hole
- "missed_hole" + "floating": Lead bends away and misses its hole
- "attached_to_line": Lead touches/attaches to brown vertical trace
- "tangled": Leads touch or merge together
- "crossed": Leads cross over each other (X pattern)

Return JSON that matches this exact schema and allowed values:

{
  "body": {
    "tilt": {"value": "none|mild|severe|unknown", "evidence": ""},
    "rotation_relative_to_vertical_lines": {"value": "none|mild|severe|unknown", "evidence": ""},
    "center_offset": {"value": "centered|left_shift|right_shift|unknown", "evidence": ""},
    "top_edge_irregularity_or_damage": {"value": "none|present|unknown", "evidence": ""},
    "surface_mark_unusual_blob": {"value": "none|present|unknown", "evidence": ""}
  },
  "leads": {
    "count_visible": {"value": 0, "evidence": ""},
    "left_lead": {
      "visibility": "clear|partial|unclear",
      "shape": "straightish|curved|severely_deformed|unknown",
      "shape_detail": "points_down|bends_left|bends_right|twisted|missing|unknown",
      "length_impression": "normal_like|short_like|unknown",
      "end_position": "in_hole|missed_hole|attached_to_line|unknown",
      "contact_with_board": "connected|floating|unknown",
      "touches_vertical_line": "yes|no|unknown",
      "evidence": ""
    },
    "center_lead": {
      "visibility": "clear|partial|unclear",
      "shape": "straightish|curved|blob_like|unknown",
      "shape_detail": "points_down|bends_left|bends_right|twisted|missing|unknown",
      "length_impression": "normal_like|short_like|unknown",
      "end_position": "in_hole|missed_hole|attached_to_line|unknown",
      "contact_with_board": "connected|floating|unknown",
      "touches_vertical_line": "yes|no|unknown",
      "evidence": ""
    },
    "right_lead": {
      "visibility": "clear|partial|unclear",
      "shape": "straightish|curved|severely_deformed|unknown",
      "shape_detail": "points_down|bends_left|bends_right|twisted|missing|unknown",
      "length_impression": "normal_like|short_like|unknown",
      "end_position": "in_hole|missed_hole|attached_to_line|unknown",
      "contact_with_board": "connected|floating|unknown",
      "touches_vertical_line": "yes|no|unknown",
      "evidence": ""
    },
    "symmetry_left_vs_right": {"value": "symmetric|slightly_asymmetric|asymmetric|unknown", "evidence": ""},
    "lead_overlap_or_clumping": {"value": "none|present|unknown", "evidence": ""},
    "lead_arrangement": {"value": "parallel|crossed|tangled|splayed|unknown", "evidence": ""},
    "all_leads_reach_holes": {"value": "yes|no|unknown", "evidence": ""}
  },
  "background_alignment": {
    "holes_pattern_visible": {"value": "yes|no|unknown", "evidence": ""},
    "bottom_three_holes_near_lead_ends": {"value": "clearly_visible|partially_visible|unclear", "evidence": ""}
  },
  "image_quality": {
    "blur": "low|medium|high",
    "lighting_glare_on_metal": "low|medium|high",
    "occlusion": "none|some|high"
  },
  "notes": ""
}"""

# =========================
# Stage 2: 판단 + 검증 프롬프트
# =========================
STAGE2_SYSTEM = """You are a manufacturing visual inspection decision agent.
You will receive:
(1) a structured observation JSON from Stage 1.

Your job:
- Decide "normal" or "abnormal" using the Stage 1 observations ONLY.
- Provide a confidence score (0.00 to 1.00).
- Provide key reasons grounded in the Stage 1 evidence fields.
- Decide whether to trigger a recheck action, based on uncertainty and image quality.
- Output MUST be valid JSON only (no markdown, no extra text).

Rules:
- If multiple Stage 1 fields are "unknown/unclear", lower confidence.
- If image_quality is poor (high blur or high occlusion), prefer recheck.
- If you trigger a recheck, specify exactly ONE action from:
  ["recheck_leads_focus", "recheck_body_alignment", "patch_recheck_leads", "dual_model_check", "none"].
- Do not invent new evidence not present in the Stage 1 JSON."""

STAGE2_USER_TEMPLATE = """Using ONLY the Stage 1 observation JSON below, make a final decision.

DEFECT CATEGORIES:

CRITICAL DEFECTS (Any ONE = ABNORMAL):
1) ANY lead end_position == "missed_hole" (lead not in its hole)
2) ANY lead end_position == "attached_to_line" (lead on trace)
3) ANY lead touches_vertical_line == "yes" (lead touching trace)
4) ANY lead contact_with_board == "floating" (lead not connected)
5) lead_arrangement == "crossed" OR "tangled" (leads crossing/tangling)
6) all_leads_reach_holes == "no"

HIGH-SIGNAL DEFECTS:
7) body.tilt.value == "severe" OR "mild" (tilted body often indicates problem)
8) center_lead.shape == "blob_like"
9) ANY lead shape == "severely_deformed"
10) ANY lead shape_detail == "twisted" OR "missing"
11) lead_overlap_or_clumping == "present"

IGNORE (False Positives):
- surface_mark_unusual_blob (usually reflections)
- bends_left/bends_right alone (normal variation)
- splayed (unless extreme)
- asymmetric (minor differences normal)

Scoring:

CRITICAL (+5 each):
- ANY lead end_position == "missed_hole": +5
- ANY lead end_position == "attached_to_line": +5
- ANY lead touches_vertical_line == "yes": +5
- ANY lead contact_with_board == "floating": +5
- lead_arrangement == "crossed" or "tangled": +5
- all_leads_reach_holes == "no": +5

HIGH-SIGNAL (+3 each):
- body.tilt.value == "severe": +3
- body.tilt.value == "mild": +2
- center_lead.shape == "blob_like": +3
- ANY lead shape == "severely_deformed": +3
- ANY lead shape_detail == "twisted" or "missing": +3
- lead_overlap_or_clumping == "present": +3

LOW-SIGNAL (+0):
- surface_mark_unusual_blob: +0
- bends_left/bends_right: +0
- splayed: +0

Classification:
- Total score >= 5: label = "abnormal"
- Total score >= 3: label = "abnormal"
- Total score < 3: label = "normal"

TRUST THE STAGE 1 OBSERVATIONS.
If Stage 1 says a lead missed its hole or touches a trace, classify as ABNORMAL.

Confidence calculation:
- Base confidence = 0.95
- Apply unknown field penalties (critical: -0.15, non-critical: -0.05)
- Score-based adjustment:
  * score 0: confidence stays high (0.90+)
  * score 1-2: confidence = 0.75~0.85
  * score 3-4: confidence = 0.80~0.90
  * score 5+: confidence = 0.90+ (high certainty abnormal)
- Final confidence = base - unknown_penalties - quality_penalties

Unknown field handling rules:
- Critical fields: leads.*.end_position, leads.*.touches_vertical_line, leads.*.contact_with_board, leads.lead_arrangement, leads.all_leads_reach_holes, leads.center_lead.shape
- If a critical field is "unknown" or "unclear": confidence -0.15 per field
- If a non-critical field is "unknown" or "unclear": confidence -0.05 per field
- If 3 or more fields are "unknown/unclear": MUST trigger a recheck action
- If 5 or more fields are "unknown/unclear": cap maximum confidence at 0.70

Termination rule:
- If confidence >= 0.80, set triggered_checks = "none".
- If confidence < 0.80 and there are "unknown/unclear" critical fields or poor image quality, trigger exactly one recheck action.
- If 3+ unknown fields exist, MUST trigger recheck regardless of confidence.

Return JSON in this exact schema:

{
  "label": "normal|abnormal",
  "confidence": 0.00,
  "key_reasons": ["...","...","..."],
  "triggered_checks": "recheck_leads_focus|recheck_body_alignment|patch_recheck_leads|dual_model_check|none",
  "termination_reason": "high_confidence|recheck_requested|insufficient_quality"
}

Stage 1 JSON:
"""


# =========================
# LangSmith Traceable 함수들
# =========================
# Temperature 설정 (낮을수록 일관된 결과, 0.0~1.0)
TEMPERATURE = 0.1


@traceable(name="LLM API Call")
def _post_chat(messages, timeout=90):
    """LLM API 호출 - LangSmith에서 추적됨"""
    payload = {"model": MODEL, "messages": messages,
               "stream": False, "temperature": TEMPERATURE}
    r = requests.post(BRIDGE_URL, headers=HEADERS,
                      json=payload, timeout=timeout)
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


@traceable(name="Stage1 Observe")
def stage1_observe(img_url: str, verbose: bool = False) -> dict:
    """Stage 1: 이미지 관찰 - 보이는 것만 JSON으로 출력"""
    content = _post_chat([
        {"role": "system", "content": STAGE1_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": STAGE1_USER},
            {"type": "image_url", "image_url": {"url": img_url}},
        ]},
    ])

    if verbose:
        print(f"    [Stage1 원본 응답]\n{content[:500]}...")

    obs = _safe_json_extract(content)

    if verbose:
        print(f"    [Stage1 파싱 완료]")

    return obs


@traceable(name="Stage2 Decide")
def stage2_decide(stage1_json: dict, verbose: bool = False) -> dict:
    """Stage 2: Stage1 결과를 기반으로 판단"""
    stage1_str = json.dumps(stage1_json, indent=2, ensure_ascii=False)
    user_prompt = STAGE2_USER_TEMPLATE + stage1_str

    content = _post_chat([
        {"role": "system", "content": STAGE2_SYSTEM},
        {"role": "user", "content": user_prompt},
    ])

    if verbose:
        print(f"    [Stage2 원본 응답]\n{content}")

    decision = _safe_json_extract(content)

    if verbose:
        print(f"    [Stage2 파싱 완료]")

    return decision


@traceable(name="Stage1 Recheck")
def stage1_recheck(img_url: str, focus_area: str, verbose: bool = False) -> dict:
    """Stage 1 재검사: 특정 영역에 집중하여 재관찰"""
    focus_prompts = {
        "recheck_leads_focus": "Focus specifically on the THREE METAL LEADS. Check their shapes, lengths, alignment to holes, and any overlap or clumping. ",
        "recheck_body_alignment": "Focus specifically on the BLACK BODY component. Check tilt, rotation, center offset, and any surface damage or marks. ",
        "patch_recheck_leads": "Focus on the LEAD TIPS and their position relative to the bottom holes. Check if tips are properly aligned. ",
        "dual_model_check": "Perform a detailed comprehensive check of ALL components: body position, all three leads, and their alignment to the holes. ",
    }

    extra_focus = focus_prompts.get(focus_area, "")
    modified_prompt = extra_focus + STAGE1_USER

    content = _post_chat([
        {"role": "system", "content": STAGE1_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": modified_prompt},
            {"type": "image_url", "image_url": {"url": img_url}},
        ]},
    ])

    if verbose:
        print(f"    [Stage1 Recheck ({focus_area}) 응답]\n{content[:500]}...")

    obs = _safe_json_extract(content)
    return obs


@traceable(name="Two-Stage Classify Agent", run_type="chain")
def classify_agent(img_url: str, img_id: str = None, max_retries=3, verbose=False) -> int:
    """Two-Stage Agent 전체 파이프라인"""
    for attempt in range(max_retries):
        try:
            # ========== Stage 1: 관찰 ==========
            if verbose:
                print(f"  [Stage 1] 이미지 관찰 시작...")
            stage1_result = stage1_observe(img_url, verbose=verbose)

            # ========== Stage 2: 판단 ==========
            if verbose:
                print(f"  [Stage 2] 판단 시작...")
            stage2_result = stage2_decide(stage1_result, verbose=verbose)

            label_str = stage2_result.get("label", "normal")
            confidence = stage2_result.get("confidence", 0.5)
            triggered_checks = stage2_result.get("triggered_checks", "none")
            termination_reason = stage2_result.get(
                "termination_reason", "high_confidence")
            key_reasons = stage2_result.get("key_reasons", [])

            if verbose:
                print(
                    f"    → 1차 판정: {label_str}, confidence: {confidence:.2f}")
                print(f"    → 근거: {key_reasons}")
                print(f"    → triggered_checks: {triggered_checks}")
                print(f"    → termination_reason: {termination_reason}")

            # ========== Recheck 필요 시 ==========
            if triggered_checks != "none" and confidence < 0.80:
                if verbose:
                    print(f"  [Recheck] {triggered_checks} 수행...")

                # Stage 1 재검사
                stage1_recheck_result = stage1_recheck(
                    img_url, triggered_checks, verbose=verbose)

                # Stage 2 재판단
                if verbose:
                    print(f"  [Stage 2 재판단] Recheck 결과 기반...")
                stage2_recheck_result = stage2_decide(
                    stage1_recheck_result, verbose=verbose)

                label_str = stage2_recheck_result.get("label", "normal")
                confidence = stage2_recheck_result.get("confidence", 0.5)
                key_reasons = stage2_recheck_result.get("key_reasons", [])

                if verbose:
                    print(
                        f"    → 재검토 판정: {label_str}, confidence: {confidence:.2f}")
                    print(f"    → 근거: {key_reasons}")

            # ========== 최종 라벨 결정 ==========
            final_label = LABEL_ABNORMAL if label_str == "abnormal" else LABEL_NORMAL

            if verbose:
                print(
                    f"  [최종 결정] {'불량(1)' if final_label == 1 else '정상(0)'} (confidence: {confidence:.2f})")

            return final_label

        except Exception as e:
            if verbose:
                print(f"  [오류] attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep((0.5 * (2 ** attempt)) + random.uniform(0, 0.2))


@traceable(name="Main Pipeline", run_type="chain")
def main():
    """메인 파이프라인 - 전체 실행이 LangSmith에서 추적됨"""
    print("=" * 60)
    print("Two-Stage Prompt Agent V3")
    print("=" * 60)
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

    print(f"\n{'='*60}")
    print(f"Saved: {OUT_PATH}")
    print(out_df.head())
    print(f"\nLangSmith 대시보드: https://smith.langchain.com")
    print(f"{'='*60}")
    return out_df


if __name__ == "__main__":
    main()
