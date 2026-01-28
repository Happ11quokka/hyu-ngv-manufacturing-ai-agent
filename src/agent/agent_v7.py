"""
AI Agent with LangSmith Tracing V7
Two-Stage Prompt + 가중치 투표 시스템 + 조건부 툴 호출

V6.1 대비 변경사항 (V7):
1. Stage 1 프롬프트에 "의심하라" 지시 추가
   - COMMON MISTAKES TO AVOID 섹션 추가
   - VERIFICATION STEP 추가 (각 리드별 검증 단계)
   - 리드가 구멍 "옆"을 지나가는 경우를 명확히 경고
"""

import os
import sys
import re
import json
import time
import random
import requests
import pandas as pd
from dotenv import load_dotenv
from langsmith import traceable
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# 프로젝트 루트 경로 계산
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# 이미지 전처리 도구 import (src/preprocessing 경로 추가)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "preprocessing"))

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
TEST_CSV_PATH = os.path.join(PROJECT_ROOT, "test", "test.csv")
OUT_PATH = os.path.join(PROJECT_ROOT, "test", "sample_submission.csv")

# API 호출 헤더
HEADERS = {"apikey": API_KEY, "Content-Type": "application/json"}

# 대회 라벨 정의
LABEL_NORMAL = 0
LABEL_ABNORMAL = 1

# Temperature 설정
TEMPERATURE = 0.1

# 최대 recheck 횟수
MAX_RECHECK_COUNT = 4

# Critical 근거 키워드 (이 근거가 있으면 abnormal을 쉽게 뒤집지 않음)
CRITICAL_REASONS = [
    "missed_hole",
    "floating",
    "attached_to_line",
    "touches_vertical_line",
    "crossed",
    "tangled",
    "all_leads_reach_holes == 'no'",
    "severely_deformed",
]

# Suspicious 근거 키워드 (확인이 필요한 의심스러운 근거)
SUSPICIOUS_REASONS = [
    "blob_like",
    "short_like",
    "asymmetric",
    "clumping",
    "overlap",
    "splayed",
    "surface_mark",
    "unusual_blob",
    "damage",
    "irregularity",
]

# 가중치 설정
WEIGHTS = {
    "original": 1.5,           # 원본 판정 가중치
    "critical_reason": 2.0,    # Critical 근거가 있는 결과 가중치
    "suspicious_reason": 1.2,  # Suspicious 근거가 있는 결과 가중치
    "recheck": 1.0,            # 일반 recheck 결과 가중치
}

# 툴 호출 조건 임계값
THRESHOLDS = {
    "confidence_for_recheck": 0.85,      # 이 confidence 이하면 recheck
    "min_votes_for_decision": 2,         # 최소 투표 수 (이하면 추가 recheck)
    "suspicious_always_recheck": True,   # Suspicious 근거 있으면 항상 recheck
}

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
# Stage 1: 관찰 전용 프롬프트 (V7 - 의심하라 지시 추가)
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

========== V7: CRITICAL WARNING - COMMON MISTAKES TO AVOID ==========
1. Do NOT assume leads are in holes just because they point downward!
2. TRACE each lead tip to its EXACT endpoint - is it truly INSIDE the dark circle?
3. If a lead PASSES BESIDE a hole but doesn't actually enter it = "missed_hole"
4. Bent/curved leads often MISS their target holes - check the TIP position carefully!
5. A lead that bends LEFT or RIGHT may end up NEXT TO the hole, not IN it
6. Look at WHERE THE METAL ENDS, not where it's pointing toward

========== V7: MANDATORY VERIFICATION STEP ==========
Before filling end_position for each lead, perform this check:
1. Locate the TARGET HOLE (dark circle at bottom)
2. Locate the LEAD TIP (where the metal actually ends)
3. Ask: "Is the lead tip OVERLAPPING with the dark hole area?"
   - YES, tip is inside/overlapping the dark circle → "in_hole"
   - NO, tip is beside/above/missing the dark circle → "missed_hole"
   - Cannot determine clearly → "unknown"

DO NOT mark "in_hole" unless you can clearly see the lead tip entering the hole!

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

V7 REMINDER - CHECK EACH LEAD CAREFULLY:
For each lead, ask yourself: "Where does this lead's TIP actually end?"
- If the tip is INSIDE the dark hole → end_position = "in_hole"
- If the tip is BESIDE, ABOVE, or MISSING the hole → end_position = "missed_hole"
- If you cannot clearly see → end_position = "unknown"

DO NOT assume "in_hole" just because the lead points toward the hole direction!

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


def has_critical_reasons(key_reasons: List[str]) -> bool:
    """Critical 근거가 있는지 확인"""
    for reason in key_reasons:
        for critical in CRITICAL_REASONS:
            if critical.lower() in reason.lower():
                return True
    return False


def count_critical_reasons(key_reasons: List[str]) -> int:
    """Critical 근거 개수 세기"""
    count = 0
    for reason in key_reasons:
        for critical in CRITICAL_REASONS:
            if critical.lower() in reason.lower():
                count += 1
                break
    return count


def has_suspicious_reasons(key_reasons: List[str]) -> bool:
    """Suspicious 근거가 있는지 확인"""
    for reason in key_reasons:
        for suspicious in SUSPICIOUS_REASONS:
            if suspicious.lower() in reason.lower():
                return True
    return False


def count_suspicious_reasons(key_reasons: List[str]) -> int:
    """Suspicious 근거 개수 세기"""
    count = 0
    for reason in key_reasons:
        for suspicious in SUSPICIOUS_REASONS:
            if suspicious.lower() in reason.lower():
                count += 1
                break
    return count


def should_trigger_recheck(stage2_result: dict, is_original: bool = False, num_results: int = 1) -> Tuple[bool, str]:
    """
    추가 recheck가 필요한지 판단

    Returns:
        (should_recheck: bool, reason: str)
    """
    label = stage2_result.get("label", "normal")
    confidence = stage2_result.get("confidence", 0.5)
    key_reasons = stage2_result.get("key_reasons", [])
    triggered_checks = stage2_result.get("triggered_checks", "none")

    # 1. 명시적으로 triggered_checks가 있으면 recheck
    if triggered_checks != "none":
        return True, f"triggered_checks={triggered_checks}"

    # 2. Confidence가 낮으면 recheck
    if confidence < THRESHOLDS["confidence_for_recheck"]:
        return True, f"low_confidence={confidence:.2f}"

    # 3. Suspicious 근거가 있으면 recheck (설정에 따라)
    if THRESHOLDS["suspicious_always_recheck"] and has_suspicious_reasons(key_reasons):
        suspicious_count = count_suspicious_reasons(key_reasons)
        return True, f"suspicious_reasons={suspicious_count}"

    # 4. 투표 수가 부족하면 recheck (최소 2개 필요)
    if num_results < THRESHOLDS["min_votes_for_decision"]:
        return True, f"insufficient_votes={num_results}"

    return False, "no_recheck_needed"


@traceable(name="Preprocess Image")
def preprocess_image_for_focus(img_url: str, focus_action: str, verbose: bool = False) -> str:
    """Focus 액션에 맞는 전처리 도구를 호출하여 이미지를 전처리"""
    if focus_action not in FOCUS_TO_TOOL:
        if verbose:
            print(f"    [전처리] 알 수 없는 focus 액션: {focus_action}, 원본 이미지 사용")
        return img_url

    tool = FOCUS_TO_TOOL[focus_action]

    if verbose:
        print(f"    [전처리] {focus_action} → {tool.name} 호출")

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


@traceable(name="Stage1 Observe V7")
def stage1_observe(img_url: str, verbose: bool = False) -> dict:
    """Stage 1: 이미지 관찰 - 보이는 것만 JSON으로 출력 (V7 프롬프트)"""
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


@traceable(name="Stage1 Recheck with Preprocessing V7")
def stage1_recheck_with_preprocessing(
    img_url: str,
    focus_action: str,
    verbose: bool = False
) -> dict:
    """Stage 1 재검사: 전처리 도구로 이미지 처리 후 재관찰 (V7 프롬프트)"""
    preprocessed_url = preprocess_image_for_focus(
        img_url, focus_action, verbose)

    focus_prompts = {
        "recheck_leads_focus": "Focus specifically on the THREE METAL LEADS. Check their shapes, lengths, alignment to holes, and any overlap or clumping. VERIFY: Does each lead tip actually END INSIDE its target hole? ",
        "recheck_body_alignment": "Focus specifically on the BLACK BODY component. Check tilt, rotation, center offset, and any surface damage or marks. ",
        "patch_recheck_leads": "Focus on the LEAD TIPS and their position relative to the bottom holes. CRITICAL: Check if tips are INSIDE the holes or BESIDE them. ",
        "dual_model_check": "Perform a detailed comprehensive check of ALL components: body position, all three leads, and their alignment to the holes. VERIFY each lead's end_position carefully. ",
    }

    extra_focus = focus_prompts.get(focus_action, "")
    modified_prompt = extra_focus + STAGE1_USER

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


@traceable(name="Stage1 Original Recheck V7")
def stage1_original_recheck(img_url: str, verbose: bool = False) -> dict:
    """Stage 1 재검사: 원본 이미지로 다시 관찰 (전처리 없이, V7 프롬프트)"""
    extra_prompt = """IMPORTANT: This is a verification recheck.
Please carefully re-examine the image, especially:
1. Check each lead's END POSITION - does it actually go INTO the hole or miss it?
2. Look at the BOTTOM of the image where the holes are located.
3. Trace each lead from the body down to where it ends.
4. A bent lead may POINT toward a hole but END beside it - check the TIP!

"""

    content = _post_chat([
        {"role": "system", "content": STAGE1_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": extra_prompt + STAGE1_USER},
            {"type": "image_url", "image_url": {"url": img_url}},
        ]},
    ])

    if verbose:
        print(f"    [Stage1 Original Recheck 응답]\n{content[:500]}...")

    obs = _safe_json_extract(content)
    return obs


def determine_next_focus(
    current_focus: str,
    used_focuses: List[str],
    stage2_result: dict
) -> Optional[str]:
    """다음 focus 액션을 결정"""
    for focus in FOCUS_SEQUENCE:
        if focus not in used_focuses:
            return focus
    return None


def vote_decision(
    results: List[dict],
    verbose: bool = False,
    result_metadata: List[dict] = None
) -> Tuple[str, float, List[str], bool]:
    """
    가중치 기반 투표

    규칙:
    1. 각 결과에 가중치 부여:
       - 원본 판정: 1.5배
       - Critical 근거 있는 결과: 2.0배
       - Suspicious 근거 있는 결과: 1.2배
       - 일반 recheck: 1.0배
    2. 가중치 합산으로 최종 결정
    3. 동점 시 is_tie=True 반환 → 추가 recheck 필요

    Args:
        results: Stage2 결과 리스트
        verbose: 상세 출력 여부
        result_metadata: 각 결과의 메타데이터 (is_original 등)
    """
    if not results:
        return "normal", 0.5, [], False

    # 메타데이터가 없으면 기본값 생성 (첫 번째만 original)
    if result_metadata is None:
        result_metadata = [{"is_original": (i == 0)}
                          for i in range(len(results))]

    abnormal_weight = 0.0
    normal_weight = 0.0
    all_reasons = []

    if verbose:
        print(f"    [V7 가중치 투표]")

    for i, result in enumerate(results):
        label = result.get("label", "normal")
        key_reasons = result.get("key_reasons", [])
        confidence = result.get("confidence", 0.5)
        meta = result_metadata[i] if i < len(result_metadata) else {
            "is_original": False}

        # 기본 가중치 계산
        weight = WEIGHTS["recheck"]

        # 원본 판정 가중치
        if meta.get("is_original", False):
            weight = WEIGHTS["original"]

        # Critical 근거가 있으면 추가 가중치
        if has_critical_reasons(key_reasons):
            weight *= WEIGHTS["critical_reason"]
        # Suspicious 근거가 있으면 추가 가중치 (Critical보다 낮음)
        elif has_suspicious_reasons(key_reasons):
            weight *= WEIGHTS["suspicious_reason"]

        # Confidence 반영 (가중치에 confidence를 곱함)
        weight *= confidence

        if verbose:
            origin_str = "원본" if meta.get("is_original") else "recheck"
            crit_str = "[CRIT]" if has_critical_reasons(key_reasons) else ""
            susp_str = "[SUSP]" if has_suspicious_reasons(key_reasons) else ""
            print(
                f"      - {origin_str} {crit_str}{susp_str}: {label} (conf={confidence:.2f}, weight={weight:.2f})")

        if label == "abnormal":
            abnormal_weight += weight
            all_reasons.extend(key_reasons)
        else:
            normal_weight += weight

    if verbose:
        print(
            f"    [가중치 합계] abnormal: {abnormal_weight:.2f}, normal: {normal_weight:.2f}")

    # 동점 체크 (가중치 차이가 0.1 이하면 동점으로 간주)
    weight_diff = abs(abnormal_weight - normal_weight)
    is_tie = weight_diff < 0.1

    # 최종 결정
    if abnormal_weight > normal_weight:
        final_label = "abnormal"
    elif normal_weight > abnormal_weight:
        final_label = "normal"
    else:
        # 동점 - 일단 abnormal로 설정 (보수적 판단)
        final_label = "abnormal"

    # confidence 계산 (가중치 비율 기반)
    total_weight = abnormal_weight + normal_weight
    if final_label == "abnormal":
        final_confidence = abnormal_weight / total_weight if total_weight > 0 else 0.5
    else:
        final_confidence = normal_weight / total_weight if total_weight > 0 else 0.5

    return final_label, final_confidence, all_reasons, is_tie


@traceable(name="Two-Stage Classify Agent V7", run_type="chain")
def classify_agent(
    img_url: str,
    img_id: str = None,
    max_retries: int = 3,
    verbose: bool = False
) -> Tuple[int, bool, str]:
    """
    Two-Stage Agent V7 - 의심하라 프롬프트 + 가중치 투표 + 조건부 툴 호출

    V6.1 대비 변경사항:
    1. Stage 1 프롬프트에 "의심하라" 지시 추가
    2. COMMON MISTAKES TO AVOID 섹션
    3. MANDATORY VERIFICATION STEP 섹션
    """
    recheck_count = 0
    used_focuses = []
    all_results = []  # 모든 Stage 2 결과 저장
    result_metadata = []  # 각 결과의 메타데이터

    for attempt in range(max_retries):
        try:
            # ========== Stage 1: 원본 이미지 관찰 ==========
            if verbose:
                print(f"  [Stage 1] 원본 이미지 관찰 시작...")
            stage1_result = stage1_observe(img_url, verbose=verbose)

            # ========== Stage 2: 1차 판단 ==========
            if verbose:
                print(f"  [Stage 2] 1차 판단 시작...")
            stage2_result = stage2_decide(stage1_result, verbose=verbose)
            all_results.append(stage2_result)
            result_metadata.append({"is_original": True})

            original_label = stage2_result.get("label", "normal")
            original_confidence = stage2_result.get("confidence", 0.5)
            original_reasons = stage2_result.get("key_reasons", [])
            triggered_checks = stage2_result.get("triggered_checks", "none")

            if verbose:
                print(
                    f"    → 1차 판정: {original_label}, confidence: {original_confidence:.2f}")
                print(f"    → 근거: {original_reasons}")
                print(
                    f"    → Critical 근거 수: {count_critical_reasons(original_reasons)}")
                print(
                    f"    → Suspicious 근거 수: {count_suspicious_reasons(original_reasons)}")

            # ========== 조건부 툴 호출 로직 ==========
            should_recheck, recheck_reason = should_trigger_recheck(
                stage2_result, is_original=True, num_results=len(all_results)
            )

            if verbose:
                print(
                    f"    → Recheck 필요: {should_recheck} (이유: {recheck_reason})")

            # ========== Case 1: Critical 근거가 있는 abnormal ==========
            if original_label == "abnormal" and has_critical_reasons(original_reasons):
                if verbose:
                    print(f"\n  [V7] Critical 근거가 있는 abnormal → 원본 이미지로 재검증")

                recheck_count += 1
                stage1_recheck = stage1_original_recheck(
                    img_url, verbose=verbose)
                stage2_recheck = stage2_decide(stage1_recheck, verbose=verbose)
                all_results.append(stage2_recheck)
                result_metadata.append(
                    {"is_original": False, "focus": "original_recheck"})

                recheck_label = stage2_recheck.get("label", "normal")
                recheck_confidence = stage2_recheck.get("confidence", 0.5)
                recheck_reasons = stage2_recheck.get("key_reasons", [])

                if verbose:
                    print(
                        f"    → 재검증 판정: {recheck_label}, confidence: {recheck_confidence:.2f}")
                    print(f"    → 재검증 근거: {recheck_reasons}")

                # 원본과 재검증 결과가 다르면 추가 recheck
                if original_label != recheck_label:
                    if verbose:
                        print(
                            f"\n  [V7] 판정 충돌! 원본({original_label}) vs 재검증({recheck_label})")
                        print(f"  [V7] 전처리 이미지로 추가 검증 수행")

                    for focus in FOCUS_SEQUENCE[:2]:
                        if recheck_count >= MAX_RECHECK_COUNT:
                            break

                        recheck_count += 1
                        used_focuses.append(focus)

                        if verbose:
                            print(f"\n  [Recheck #{recheck_count}] {focus}")

                        stage1_focus = stage1_recheck_with_preprocessing(
                            img_url, focus, verbose=verbose)
                        stage2_focus = stage2_decide(
                            stage1_focus, verbose=verbose)
                        all_results.append(stage2_focus)
                        result_metadata.append(
                            {"is_original": False, "focus": focus})

                        if verbose:
                            focus_label = stage2_focus.get("label", "normal")
                            focus_conf = stage2_focus.get("confidence", 0.5)
                            print(
                                f"    → {focus} 판정: {focus_label}, confidence: {focus_conf:.2f}")

            # ========== Case 2: Suspicious 근거가 있거나 confidence가 낮은 경우 ==========
            elif should_recheck:
                if verbose:
                    print(f"\n  [V7] 조건부 recheck 트리거: {recheck_reason}")

                # 첫 번째 recheck: 명시된 triggered_checks가 있으면 그것 사용, 없으면 leads_focus
                first_focus = triggered_checks if triggered_checks != "none" else "recheck_leads_focus"

                recheck_count += 1
                used_focuses.append(first_focus)

                if verbose:
                    print(f"\n  [Recheck #{recheck_count}] {first_focus}")

                stage1_recheck = stage1_recheck_with_preprocessing(
                    img_url, first_focus, verbose=verbose)
                stage2_recheck = stage2_decide(stage1_recheck, verbose=verbose)
                all_results.append(stage2_recheck)
                result_metadata.append(
                    {"is_original": False, "focus": first_focus})

                recheck_label = stage2_recheck.get("label", "normal")
                recheck_confidence = stage2_recheck.get("confidence", 0.5)

                if verbose:
                    print(
                        f"    → Recheck 판정: {recheck_label}, confidence: {recheck_confidence:.2f}")

                # 원본과 recheck 결과가 다르면 추가 검증
                if original_label != recheck_label and recheck_count < MAX_RECHECK_COUNT:
                    if verbose:
                        print(f"\n  [V7] 판정 충돌! 추가 검증 수행")

                    # 추가 focus로 한 번 더 검증
                    for focus in FOCUS_SEQUENCE:
                        if focus not in used_focuses and recheck_count < MAX_RECHECK_COUNT:
                            recheck_count += 1
                            used_focuses.append(focus)

                            if verbose:
                                print(
                                    f"\n  [Recheck #{recheck_count}] {focus}")

                            stage1_focus = stage1_recheck_with_preprocessing(
                                img_url, focus, verbose=verbose)
                            stage2_focus = stage2_decide(
                                stage1_focus, verbose=verbose)
                            all_results.append(stage2_focus)
                            result_metadata.append(
                                {"is_original": False, "focus": focus})

                            if verbose:
                                focus_label = stage2_focus.get(
                                    "label", "normal")
                                focus_conf = stage2_focus.get(
                                    "confidence", 0.5)
                                print(
                                    f"    → {focus} 판정: {focus_label}, confidence: {focus_conf:.2f}")
                            break

            # ========== 가중치 기반 투표 ==========
            if verbose:
                print(f"\n  [V7 가중치 투표] 총 {len(all_results)}개 결과로 투표...")

            final_label, final_confidence, critical_reasons, is_tie = vote_decision(
                all_results, verbose=verbose, result_metadata=result_metadata
            )

            # ========== 동점 시 추가 recheck ==========
            while is_tie and recheck_count < MAX_RECHECK_COUNT:
                if verbose:
                    print(f"\n  [V7 동점] 추가 recheck로 타이브레이크!")

                next_focus = None
                for focus in FOCUS_SEQUENCE:
                    if focus not in used_focuses:
                        next_focus = focus
                        break

                if next_focus is None:
                    if verbose:
                        print(f"    → 모든 focus 사용됨, 동점 시 abnormal 우선")
                    break

                recheck_count += 1
                used_focuses.append(next_focus)

                if verbose:
                    print(
                        f"    → 타이브레이크 recheck #{recheck_count}: {next_focus}")

                stage1_tiebreak = stage1_recheck_with_preprocessing(
                    img_url, next_focus, verbose=verbose)
                stage2_tiebreak = stage2_decide(
                    stage1_tiebreak, verbose=verbose)
                all_results.append(stage2_tiebreak)
                result_metadata.append(
                    {"is_original": False, "focus": next_focus})

                tiebreak_label = stage2_tiebreak.get("label", "normal")
                if verbose:
                    print(f"    → 타이브레이크 결과: {tiebreak_label}")

                final_label, final_confidence, critical_reasons, is_tie = vote_decision(
                    all_results, verbose=verbose, result_metadata=result_metadata
                )

            # ========== 최종 라벨 결정 ==========
            needs_review = False
            review_message = ""

            labels = [r.get("label") for r in all_results]
            if len(set(labels)) > 1:
                needs_review = True
                abnormal_count = sum(1 for l in labels if l == "abnormal")
                review_message = f"[판정 충돌] {abnormal_count}/{len(labels)} abnormal. 투표 결과: {final_label}"

            final_label_int = LABEL_ABNORMAL if final_label == "abnormal" else LABEL_NORMAL

            if verbose:
                result_text = '불량(1)' if final_label_int == 1 else '정상(0)'
                print(
                    f"\n  [최종 결정] {result_text} (confidence: {final_confidence:.2f})")
                if needs_review:
                    print(f"  [!] {review_message}")
                print(
                    f"  [통계] 총 recheck 횟수: {recheck_count}, 사용된 focus: {used_focuses}")
                print(f"  [통계] 투표 결과: {[r.get('label') for r in all_results]}")

            return final_label_int, needs_review, review_message

        except Exception as e:
            if verbose:
                print(f"  [오류] attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep((0.5 * (2 ** attempt)) + random.uniform(0, 0.2))

    # Fallback
    return LABEL_NORMAL, True, "최대 재시도 횟수 초과로 기본값 반환"


def process_single_image(args: Tuple[int, str, str, int, bool]) -> dict:
    """단일 이미지 처리 함수 (병렬 처리용)"""
    idx, _id, img_url, total, verbose = args

    result = {
        "id": _id,
        "label": LABEL_NORMAL,
        "needs_review": False,
        "review_message": ""
    }

    try:
        label, needs_review, review_message = classify_agent(
            img_url, img_id=_id, verbose=verbose
        )

        result["label"] = label
        result["needs_review"] = needs_review
        result["review_message"] = review_message

        result_text = '불량(1)' if label == 1 else '정상(0)'
        print(f"[{idx+1}/{total}] {_id} -> {result_text}")

    except Exception as e:
        print(f"[{idx+1}/{total}] {_id} -> 오류: {e}")
        result["needs_review"] = True
        result["review_message"] = f"오류 발생: {str(e)}"

    return result


@traceable(name="Main Pipeline V7 Parallel", run_type="chain")
def main():
    """메인 파이프라인 V7 - 병렬 처리 버전"""
    # 병렬 처리 설정
    MAX_WORKERS = 20  # 동시 처리할 이미지 수 (API 제한에 따라 조절)
    VERBOSE = False   # 병렬 처리 시 verbose는 False 권장

    print("=" * 60)
    print("Two-Stage Prompt Agent V7 (의심하라 프롬프트)")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Max Workers: {MAX_WORKERS}")
    print(f"LangSmith Project: {os.getenv('LANGCHAIN_PROJECT')}")
    print()

    test_df = pd.read_csv(TEST_CSV_PATH)

    if "id" not in test_df.columns or "img_url" not in test_df.columns:
        raise ValueError(f"columns: {test_df.columns.tolist()}")

    n = len(test_df)
    print(f"총 {n}개 이미지 처리 시작...\n")

    # 작업 목록 생성
    tasks = [
        (i, row["id"], row["img_url"], n, VERBOSE)
        for i, row in test_df.iterrows()
    ]

    results = []
    review_items = []

    # 병렬 처리 실행
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(process_single_image, task): task for task in tasks}

        for future in as_completed(future_to_task):
            result = future.result()
            results.append(result)

            if result["needs_review"]:
                review_items.append({
                    "id": result["id"],
                    "label": result["label"],
                    "message": result["review_message"]
                })

    elapsed_time = time.time() - start_time

    # ID 순서대로 정렬
    results.sort(key=lambda x: x["id"])

    # 결과 저장
    preds = [{"id": r["id"], "label": r["label"]} for r in results]
    out_df = pd.DataFrame(preds, columns=["id", "label"])
    out_df.to_csv(OUT_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"완료! 소요 시간: {elapsed_time:.1f}초")
    print(f"Saved: {OUT_PATH}")
    print(out_df.head())

    # 결과 요약
    abnormal_count = sum(1 for r in results if r["label"] == 1)
    normal_count = sum(1 for r in results if r["label"] == 0)
    print(f"\n[결과 요약] 정상: {normal_count}개, 불량: {abnormal_count}개")

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
