"""
AI Agent with LangSmith Tracing V5
Two-Stage Prompt + 이미지 전처리 도구 연동

새로운 규칙:
1. Recheck 시 해당 focus에 맞는 전처리 도구 자동 호출
2. 다중 focus 명령 시 순차적으로 하나씩 recheck
3. 4번 이상 recheck 시 결과 평균값 출력 + 검토 필요 표시
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
from typing import List, Dict, Tuple, Optional

# 이미지 전처리 도구 import
from image_preprocessing_tools import (
    preprocess_focus_leads,
    preprocess_focus_body,
    preprocess_focus_lead_tips,
    preprocess_full_enhanced,
    load_image,
    image_to_base64,
    base64_to_data_url,
)

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

# Temperature 설정
TEMPERATURE = 0.1

# 최대 recheck 횟수
MAX_RECHECK_COUNT = 6

# =========================
# Focus 액션 → 전처리 도구 매핑
# =========================
FOCUS_TO_TOOL = {
    "recheck_leads_focus": preprocess_focus_leads,
    "recheck_body_alignment": preprocess_focus_body,
    "patch_recheck_leads": preprocess_focus_lead_tips,
    "dual_model_check": preprocess_full_enhanced,
}

# 다중 focus 시퀀스 정의 (우선순위 순)
FOCUS_SEQUENCE = [
    "recheck_leads_focus",
    "patch_recheck_leads",
    "recheck_body_alignment",
    "dual_model_check",
]


# =========================
# Stage 1: 관찰 전용 프롬프트
# =========================
STAGE1_SYSTEM = """You are a visual inspection observer for manufacturing images.
STRICT RULES:
- Do NOT classify as normal/abnormal.
- Report ONLY what is directly visible in the image.
- If uncertain, use "unknown" or "unclear" rather than guessing.
- Output MUST be valid JSON only (no markdown, no extra text).

========== IMAGE COMPONENT IDENTIFICATION (CRITICAL) ==========
The image contains these distinct elements - IDENTIFY EACH CORRECTLY:

1. BLACK BODY: The black rectangular component at the top center
   - This is the main component body

2. THREE METAL LEADS: Silver/metallic wires extending DOWN from the body
   - LEFT lead, CENTER lead, RIGHT lead
   - They should go straight down into the holes

3. THREE TARGET HOLES: Dark circular holes at the BOTTOM of the image
   - Located directly below where each lead should end
   - Each lead must enter its corresponding hole

4. VERTICAL LINES (TRACES): Brown/copper colored vertical stripes on the board
   - These run parallel to leads but are PART OF THE BOARD, not leads
   - Leads should NOT touch or attach to these lines
   - If a lead touches/attaches to a vertical line = DEFECT (broken/misrouted)

========== CRITICAL INSPECTION: LEAD-TO-HOLE CONNECTION ==========
For EACH lead (left, center, right), check:
1. Does the lead END inside its target hole? (connected)
2. Does the lead MISS the hole and end elsewhere? (floating)
3. Does the lead touch/attach to a vertical LINE instead of hole? (floating + defect)

COMMON DEFECTS TO DETECT:
- Lead bending away and missing its hole
- Lead attaching to vertical line instead of going to hole
- Lead too short to reach the hole
- Lead crossing over to wrong position

========== SURFACE_MARK_UNUSUAL_BLOB (STRICT) ==========
- "present" ONLY for: foreign objects, solder splatter, physical damage
- "none" for: normal reflections, standard markings, circular indent on body
- When in doubt, mark as "none"

========== LEAD ARRANGEMENT ==========
- "parallel": Leads go in same direction toward their holes (NORMAL)
- "crossed": Leads physically cross over each other (X pattern)
- "tangled": Leads touch, overlap, or merge together
- "splayed": Leads spread so wide they miss their holes completely

========== CENTER LEAD INSPECTION ==========
- Check if center lead looks like a BLOB (lumpy, not wire-shaped) = "blob_like"
- Check if center lead is SHORTER than others = "short_like"
- Check if center lead is tangled with adjacent leads"""

STAGE1_USER = """Observe the provided image and fill the JSON below using ONLY visible evidence.
Do NOT conclude normal/abnormal.

IMPORTANT DEFINITIONS:
- HOLE: Dark circular opening at the bottom where lead should INSERT into
- VERTICAL LINE/TRACE: Brown/copper stripe running vertically on the board (NOT a hole)
- "connected": Lead tip is INSIDE the hole
- "floating": Lead tip is NOT inside the hole (missed, bent away, or attached to line instead)
- "touches_line": Lead is touching or attached to a vertical line/trace (this is a DEFECT)

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

Decision strategy:
- Treat these as CRITICAL abnormal indicators (IMMEDIATE FAIL):
  1) ANY lead end_position == "missed_hole" (lead not in its hole)
  2) ANY lead end_position == "attached_to_line" (lead stuck to vertical trace)
  3) ANY lead touches_vertical_line == "yes" (lead touching board trace)
  4) ANY lead contact_with_board == "floating" (lead not connected)
  5) leads.lead_arrangement == "crossed" OR "tangled" (leads crossing/tangled)
  6) leads.all_leads_reach_holes == "no"

- Treat these as high-signal abnormal indicators:
  7) body.tilt.value == "severe"
  8) body.top_edge_irregularity_or_damage.value == "present"
  9) leads.center_lead.shape == "blob_like" OR leads.center_lead.length_impression == "short_like"
  10) leads.lead_overlap_or_clumping.value == "present"
  11) ANY lead shape == "severely_deformed"
  12) ANY lead shape_detail == "twisted" OR "missing"

- Medium-signal abnormal indicators:
  13) leads.symmetry_left_vs_right.value == "asymmetric"

- LOW-signal (NOT abnormal alone):
  14) bends_left or bends_right ALONE = normal variation
  15) body.surface_mark_unusual_blob = often false positive, verify carefully

Scoring guideline (for internal decision) - ADD scores for each matching condition:

CRITICAL-SIGNAL (+5 each) - AUTOMATIC ABNORMAL:
- ANY lead end_position == "missed_hole": +5 (lead missed its target hole!)
- ANY lead end_position == "attached_to_line": +5 (lead stuck to trace!)
- ANY lead touches_vertical_line == "yes": +5 (lead touching board trace!)
- ANY lead contact_with_board == "floating": +5
- leads.lead_arrangement == "crossed" or "tangled": +5
- leads.all_leads_reach_holes == "no": +5

HIGH-SIGNAL (+3 to +4 each):
- body.tilt.value == "severe": +3
- body.top_edge_irregularity_or_damage.value == "present": +3
- leads.center_lead.shape == "blob_like": +4
- leads.center_lead.length_impression == "short_like": +4
- leads.lead_overlap_or_clumping.value == "present": +3
- ANY lead shape == "severely_deformed": +4
- ANY lead shape_detail == "twisted" or "missing": +5

MEDIUM-SIGNAL (+2 each):
- leads.symmetry_left_vs_right.value == "asymmetric": +2

LOW-SIGNAL (careful evaluation):
- body.surface_mark_unusual_blob.value == "present": +1 (often false positive)
- leads.lead_arrangement == "splayed": +1 (only if severe)
- ANY lead shape_detail == "bends_left" or "bends_right": +0 (normal variation)

Calculate TOTAL SCORE by summing ALL matching conditions. Do NOT skip any matching condition.

CRITICAL RULES (IMMEDIATE ABNORMAL - NO EXCEPTIONS):
- If ANY lead has end_position == "missed_hole": ABNORMAL
- If ANY lead has end_position == "attached_to_line": ABNORMAL
- If ANY lead has touches_vertical_line == "yes": ABNORMAL
- If ANY lead has contact_with_board == "floating": ABNORMAL
- If leads.lead_arrangement == "crossed" or "tangled": ABNORMAL

Classification threshold:
- Total score >= 3: label = "abnormal"
- Total score < 3: label = "normal"
- Any CRITICAL RULE match: label = "abnormal" (regardless of score)

IMPORTANT EXAMPLES:
- If right_lead.end_position == "missed_hole" → ABNORMAL (lead not in hole)
- If left_lead.touches_vertical_line == "yes" → ABNORMAL (lead touching trace)
- If lead_arrangement == "tangled" → ABNORMAL (leads tangled)
- Do NOT override with your own judgment. Follow the rules strictly.

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


@traceable(name="Preprocess Image")
def preprocess_image_for_focus(img_url: str, focus_action: str, verbose: bool = False) -> str:
    """
    Focus 액션에 맞는 전처리 도구를 호출하여 이미지를 전처리

    Args:
        img_url: 원본 이미지 URL
        focus_action: recheck 액션 이름
        verbose: 상세 출력 여부

    Returns:
        전처리된 이미지의 data URL (base64)
    """
    if focus_action not in FOCUS_TO_TOOL:
        if verbose:
            print(f"    [전처리] 알 수 없는 focus 액션: {focus_action}, 원본 이미지 사용")
        return img_url

    tool = FOCUS_TO_TOOL[focus_action]

    if verbose:
        print(f"    [전처리] {focus_action} → {tool.name} 호출")

    # 도구 호출 (output_path=None이면 base64 data URL 반환)
    try:
        result = tool.invoke({
            "image_source": img_url,
            "output_path": None,
        })
        if verbose:
            print(f"    [전처리] 완료 (data URL 길이: {len(result)})")
        return result
    except Exception as e:
        if verbose:
            print(f"    [전처리] 실패: {e}, 원본 이미지 사용")
        return img_url


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


@traceable(name="Stage1 Recheck with Preprocessing")
def stage1_recheck_with_preprocessing(
    img_url: str,
    focus_action: str,
    verbose: bool = False
) -> dict:
    """
    Stage 1 재검사: 전처리 도구로 이미지 처리 후 재관찰

    Args:
        img_url: 원본 이미지 URL
        focus_action: recheck 액션 이름
        verbose: 상세 출력 여부

    Returns:
        Stage 1 관찰 결과 JSON
    """
    # 1. 전처리 도구 호출
    preprocessed_url = preprocess_image_for_focus(img_url, focus_action, verbose)

    # 2. Focus에 맞는 추가 지시 프롬프트
    focus_prompts = {
        "recheck_leads_focus": "Focus specifically on the THREE METAL LEADS. Check their shapes, lengths, alignment to holes, and any overlap or clumping. ",
        "recheck_body_alignment": "Focus specifically on the BLACK BODY component. Check tilt, rotation, center offset, and any surface damage or marks. ",
        "patch_recheck_leads": "Focus on the LEAD TIPS and their position relative to the bottom holes. Check if tips are properly aligned. ",
        "dual_model_check": "Perform a detailed comprehensive check of ALL components: body position, all three leads, and their alignment to the holes. ",
    }

    extra_focus = focus_prompts.get(focus_action, "")
    modified_prompt = extra_focus + STAGE1_USER

    # 3. LLM 호출 (전처리된 이미지 사용)
    content = _post_chat([
        {"role": "system", "content": STAGE1_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": modified_prompt},
            {"type": "image_url", "image_url": {"url": preprocessed_url}},
        ]},
    ])

    if verbose:
        print(f"    [Stage1 Recheck ({focus_action}) 응답]\n{content[:500]}...")

    obs = _safe_json_extract(content)
    return obs


def determine_next_focus(
    current_focus: str,
    used_focuses: List[str],
    stage2_result: dict
) -> Optional[str]:
    """
    다음 focus 액션을 결정

    규칙:
    - 아직 사용하지 않은 focus 중에서 선택
    - FOCUS_SEQUENCE 우선순위에 따라 선택

    Returns:
        다음 focus 액션 또는 None
    """
    for focus in FOCUS_SEQUENCE:
        if focus not in used_focuses:
            return focus
    return None


def calculate_average_result(results: List[dict]) -> Tuple[str, float, bool]:
    """
    여러 recheck 결과의 평균을 계산

    Args:
        results: Stage 2 결과 목록

    Returns:
        (평균 label, 평균 confidence, 검토 필요 여부)
    """
    abnormal_count = sum(1 for r in results if r.get("label") == "abnormal")
    total_confidence = sum(r.get("confidence", 0.5) for r in results)

    avg_confidence = total_confidence / len(results)
    # 과반수가 abnormal이면 abnormal
    avg_label = "abnormal" if abnormal_count > len(results) / 2 else "normal"

    # 4번 이상 recheck했으면 검토 필요
    needs_review = len(results) >= 4

    return avg_label, avg_confidence, needs_review


@traceable(name="Two-Stage Classify Agent V5", run_type="chain")
def classify_agent(
    img_url: str,
    img_id: str = None,
    max_retries: int = 3,
    verbose: bool = False
) -> Tuple[int, bool, str]:
    """
    Two-Stage Agent V5 전체 파이프라인

    새로운 규칙 적용:
    1. Recheck 시 해당 focus에 맞는 전처리 도구 자동 호출
    2. 다중 focus 명령 시 순차적으로 하나씩 recheck
    3. 4번 이상 recheck 시 결과 평균값 출력 + 검토 필요 표시

    Returns:
        (label, needs_review, review_message)
        - label: 0 (정상) 또는 1 (불량)
        - needs_review: 검토 필요 여부
        - review_message: 검토 메시지
    """
    recheck_count = 0
    used_focuses = []
    all_results = []  # 모든 Stage 2 결과 저장

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
            all_results.append(stage2_result)

            label_str = stage2_result.get("label", "normal")
            confidence = stage2_result.get("confidence", 0.5)
            triggered_checks = stage2_result.get("triggered_checks", "none")
            termination_reason = stage2_result.get("termination_reason", "high_confidence")
            key_reasons = stage2_result.get("key_reasons", [])

            if verbose:
                print(f"    → 1차 판정: {label_str}, confidence: {confidence:.2f}")
                print(f"    → 근거: {key_reasons}")
                print(f"    → triggered_checks: {triggered_checks}")
                print(f"    → termination_reason: {termination_reason}")

            # ========== 강제 Recheck: abnormal 판정 시 1회 확인 ==========
            if label_str == "abnormal" and recheck_count == 0:
                if verbose:
                    print(f"\n  [강제 Recheck] abnormal 판정 → 확인을 위해 1회 재검사 수행")

                # 강제로 recheck 트리거
                triggered_checks = "recheck_leads_focus"
                # confidence를 임시로 낮춰서 recheck 조건 충족
                confidence = 0.70

            # ========== Recheck 로직 ==========
            while triggered_checks != "none" and confidence < 0.80 and recheck_count < MAX_RECHECK_COUNT:
                recheck_count += 1
                current_focus = triggered_checks

                if verbose:
                    print(f"\n  [Recheck #{recheck_count}] {current_focus}")
                    print(f"    → 사용된 focus 목록: {used_focuses}")

                # 이미 사용한 focus면 다음 focus로 변경
                if current_focus in used_focuses:
                    next_focus = determine_next_focus(current_focus, used_focuses, stage2_result)
                    if next_focus:
                        if verbose:
                            print(f"    → {current_focus}는 이미 사용됨, 다음 focus로 변경: {next_focus}")
                        current_focus = next_focus
                    else:
                        if verbose:
                            print(f"    → 모든 focus를 사용함, recheck 종료")
                        break

                used_focuses.append(current_focus)

                # Stage 1 재검사 (전처리 도구 사용)
                stage1_recheck_result = stage1_recheck_with_preprocessing(
                    img_url, current_focus, verbose=verbose
                )

                # Stage 2 재판단
                if verbose:
                    print(f"  [Stage 2 재판단] Recheck 결과 기반...")
                stage2_recheck_result = stage2_decide(stage1_recheck_result, verbose=verbose)
                all_results.append(stage2_recheck_result)

                label_str = stage2_recheck_result.get("label", "normal")
                confidence = stage2_recheck_result.get("confidence", 0.5)
                triggered_checks = stage2_recheck_result.get("triggered_checks", "none")
                key_reasons = stage2_recheck_result.get("key_reasons", [])

                if verbose:
                    print(f"    → 재검토 #{recheck_count} 판정: {label_str}, confidence: {confidence:.2f}")
                    print(f"    → 근거: {key_reasons}")
                    print(f"    → 다음 triggered_checks: {triggered_checks}")

            # ========== 최종 결과 결정 ==========
            needs_review = False
            review_message = ""

            # 4번 이상 recheck한 경우: 평균값 계산 + 검토 필요
            if recheck_count >= 4:
                avg_label, avg_confidence, needs_review = calculate_average_result(all_results)
                label_str = avg_label
                confidence = avg_confidence

                review_message = (
                    f"[검토 필요] {recheck_count}회 recheck 수행됨. "
                    f"결과 평균: {avg_label} (신뢰도: {avg_confidence:.2f}). "
                    f"abnormal 비율: {sum(1 for r in all_results if r.get('label') == 'abnormal')}/{len(all_results)}"
                )

                if verbose:
                    print(f"\n  {'='*50}")
                    print(f"  [주의] {review_message}")
                    print(f"  {'='*50}")

            # ========== 최종 라벨 결정 ==========
            final_label = LABEL_ABNORMAL if label_str == "abnormal" else LABEL_NORMAL

            if verbose:
                result_text = '불량(1)' if final_label == 1 else '정상(0)'
                print(f"\n  [최종 결정] {result_text} (confidence: {confidence:.2f})")
                if needs_review:
                    print(f"  [!] 이 결과는 수동 검토가 필요합니다.")
                print(f"  [통계] 총 recheck 횟수: {recheck_count}, 사용된 focus: {used_focuses}")

            return final_label, needs_review, review_message

        except Exception as e:
            if verbose:
                print(f"  [오류] attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep((0.5 * (2 ** attempt)) + random.uniform(0, 0.2))

    # Fallback
    return LABEL_NORMAL, True, "최대 재시도 횟수 초과로 기본값 반환"


@traceable(name="Main Pipeline V5", run_type="chain")
def main():
    """메인 파이프라인 V5 - 전처리 도구 연동"""
    print("=" * 60)
    print("Two-Stage Prompt Agent V5 (전처리 도구 연동)")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"LangSmith Project: {os.getenv('LANGCHAIN_PROJECT')}")
    print(f"Tracing enabled: {os.getenv('LANGCHAIN_TRACING_V2')}")
    print()
    print("[새로운 규칙]")
    print("1. Recheck 시 해당 focus에 맞는 전처리 도구 자동 호출")
    print("2. 다중 focus 명령 시 순차적으로 하나씩 recheck")
    print("3. 4번 이상 recheck 시 결과 평균값 출력 + 검토 필요 표시")
    print()

    test_df = pd.read_csv(TEST_CSV_PATH)

    if "id" not in test_df.columns or "img_url" not in test_df.columns:
        raise ValueError(f"columns: {test_df.columns.tolist()}")

    preds = []
    review_items = []  # 검토 필요 항목
    n = len(test_df)

    VERBOSE = True

    for i, row in test_df.iterrows():
        _id = row["id"]
        img_url = row["img_url"]

        print(f"\n{'='*50}")
        print(f"[{i+1}/{n}] id={_id}")
        print(f"{'='*50}")

        try:
            label, needs_review, review_message = classify_agent(
                img_url, img_id=_id, verbose=VERBOSE
            )

            result_text = '불량(1)' if label == 1 else '정상(0)'
            print(f"\n  ★ 최종 결과: {_id} -> {result_text}")

            if needs_review:
                print(f"  ⚠️  {review_message}")
                review_items.append({
                    "id": _id,
                    "label": label,
                    "message": review_message
                })

        except Exception as e:
            print(f"\n  ✗ 오류 발생: {e}")
            print(f"  ★ 최종 결과: {_id} -> 정상(0) [fallback]")
            label = LABEL_NORMAL
            review_items.append({
                "id": _id,
                "label": label,
                "message": f"오류 발생: {str(e)}"
            })

        preds.append({"id": _id, "label": label})
        time.sleep(0.2)

    out_df = pd.DataFrame(preds, columns=["id", "label"])
    out_df.to_csv(OUT_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"Saved: {OUT_PATH}")
    print(out_df.head())

    # 검토 필요 항목 출력
    if review_items:
        print(f"\n{'='*60}")
        print(f"[검토 필요 항목] 총 {len(review_items)}개")
        print(f"{'='*60}")
        for item in review_items:
            print(f"  - {item['id']}: label={item['label']}, {item['message']}")

    print(f"\nLangSmith 대시보드: https://smith.langchain.com")
    print(f"{'='*60}")

    return out_df, review_items


if __name__ == "__main__":
    main()
