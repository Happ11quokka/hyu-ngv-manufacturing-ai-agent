"""
AI Agent with LangSmith Tracing V10
YOLO Detection + Two-Stage Prompt + 가중치 투표 시스템

V7 대비 변경사항 (V10):
1. Stage 0 추가: YOLO 기반 객체 검출 (Roboflow API)
   - hole, lead_tip, body 검출
   - 검출 결과를 기반으로 위치 분석
2. Stage 1: YOLO 검출 결과 + 이미지로 정밀 관찰
   - YOLO 검출 좌표를 참고하여 더 정확한 관찰
3. Stage 2: 기존 V7 판단 로직 유지
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

# Roboflow inference SDK
try:
    from inference_sdk import InferenceHTTPClient
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False
    print("[Warning] inference_sdk not installed. Run: pip install inference-sdk")

# 프로젝트 루트 경로 계산
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# 이미지 전처리 도구 import
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

# Roboflow 설정
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "4jFqretpF99k2W3RnV1S")
ROBOFLOW_API_URL = "https://serverless.roboflow.com"
ROBOFLOW_WORKSPACE = "ngv-ra5w3"

# 파인튜닝된 모델 설정 (학습 완료 후 업데이트)
# 옵션 1: Roboflow Workflow (현재 사용)
ROBOFLOW_WORKFLOW_ID = "find-dark-circular-holes-silver-metal-leads-and-black-rectangular-bodies"
# 옵션 2: 파인튜닝된 모델 (학습 완료 후 설정)
ROBOFLOW_MODEL_ID = None  # 예: "ngv-component-detection/1"

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

# Critical/Suspicious 근거 키워드 (V7과 동일)
CRITICAL_REASONS = [
    "missed_hole", "floating", "attached_to_line", "touches_vertical_line",
    "crossed", "tangled", "all_leads_reach_holes == 'no'", "severely_deformed",
]

SUSPICIOUS_REASONS = [
    "blob_like", "short_like", "asymmetric", "clumping", "overlap",
    "splayed", "surface_mark", "unusual_blob", "damage", "irregularity",
]

# 가중치 설정
WEIGHTS = {
    "original": 1.5,
    "critical_reason": 2.0,
    "suspicious_reason": 1.2,
    "recheck": 1.0,
}

# 툴 호출 조건 임계값
THRESHOLDS = {
    "confidence_for_recheck": 0.85,
    "min_votes_for_decision": 2,
    "suspicious_always_recheck": True,
}

# Focus 액션 → 전처리 도구 매핑
FOCUS_TO_TOOL = {
    "recheck_leads_focus": preprocess_focus_leads,
    "recheck_body_alignment": preprocess_focus_body,
    "patch_recheck_leads": preprocess_focus_lead_tips,
    "dual_model_check": preprocess_full_enhanced,
}

FOCUS_SEQUENCE = [
    "recheck_leads_focus",
    "patch_recheck_leads",
    "recheck_body_alignment",
    "dual_model_check",
]

# =========================
# YOLO 검출 클래스 매핑
# =========================
YOLO_CLASS_MAPPING = {
    "hole": 0,
    "dark-circular-hole": 0,
    "dark circular hole": 0,
    "lead_tip": 1,
    "lead": 1,
    "silver-metal-lead": 1,
    "silver metal lead": 1,
    "body": 2,
    "black-rectangular-body": 2,
    "black rectangular body": 2,
}

YOLO_CLASS_NAMES = {0: "hole", 1: "lead_tip", 2: "body"}


# =========================
# Stage 0: YOLO Detection
# =========================
@traceable(name="Stage0 YOLO Detection")
def stage0_yolo_detect(img_url: str, verbose: bool = False) -> dict:
    """
    Stage 0: YOLO 기반 객체 검출

    Returns:
        {
            "holes": [{"x": cx, "y": cy, "w": w, "h": h, "confidence": conf}, ...],
            "leads": [{"x": cx, "y": cy, "w": w, "h": h, "confidence": conf}, ...],
            "body": {"x": cx, "y": cy, "w": w, "h": h, "confidence": conf} or None,
            "raw_predictions": [...],
            "detection_success": bool
        }
    """
    if not ROBOFLOW_AVAILABLE:
        if verbose:
            print("    [Stage0] Roboflow SDK 미설치, 스킵")
        return {"holes": [], "leads": [], "body": None, "detection_success": False}

    try:
        client = InferenceHTTPClient(
            api_url=ROBOFLOW_API_URL,
            api_key=ROBOFLOW_API_KEY
        )

        # 파인튜닝된 모델이 있으면 사용, 없으면 workflow 사용
        if ROBOFLOW_MODEL_ID:
            if verbose:
                print(f"    [Stage0] 파인튜닝 모델 사용: {ROBOFLOW_MODEL_ID}")
            result = client.infer(img_url, model_id=ROBOFLOW_MODEL_ID)
        else:
            if verbose:
                print(f"    [Stage0] Workflow 사용: {ROBOFLOW_WORKFLOW_ID}")
            result = client.run_workflow(
                workspace_name=ROBOFLOW_WORKSPACE,
                workflow_id=ROBOFLOW_WORKFLOW_ID,
                images={"image": img_url},
                use_cache=True
            )
            # workflow 결과가 리스트면 첫 번째 요소
            if isinstance(result, list) and len(result) > 0:
                result = result[0]

        # 검출 결과 파싱
        predictions = []
        if isinstance(result, dict):
            if "predictions" in result:
                preds = result["predictions"]
                if isinstance(preds, list):
                    predictions = preds
                elif isinstance(preds, dict) and "predictions" in preds:
                    predictions = preds["predictions"]

        # 클래스별로 분류
        holes = []
        leads = []
        body = None

        for pred in predictions:
            if not isinstance(pred, dict):
                continue

            class_name = pred.get("class", pred.get("label", "")).lower()
            confidence = pred.get("confidence", 0.5)

            # 바운딩 박스 추출 (center x, y, width, height)
            bbox = {
                "x": pred.get("x", 0),
                "y": pred.get("y", 0),
                "w": pred.get("width", 0),
                "h": pred.get("height", 0),
                "confidence": confidence
            }

            # 클래스 분류
            class_id = None
            for key, cid in YOLO_CLASS_MAPPING.items():
                if key in class_name:
                    class_id = cid
                    break

            if class_id == 0:  # hole
                holes.append(bbox)
            elif class_id == 1:  # lead
                leads.append(bbox)
            elif class_id == 2:  # body
                if body is None or confidence > body.get("confidence", 0):
                    body = bbox

        # x 좌표로 정렬
        holes.sort(key=lambda h: h["x"])
        leads.sort(key=lambda l: l["x"])

        if verbose:
            print(f"    [Stage0] 검출 결과: holes={len(holes)}, leads={len(leads)}, body={'O' if body else 'X'}")

        return {
            "holes": holes,
            "leads": leads,
            "body": body,
            "raw_predictions": predictions,
            "detection_success": True
        }

    except Exception as e:
        if verbose:
            print(f"    [Stage0] 오류: {e}")
        return {"holes": [], "leads": [], "body": None, "detection_success": False, "error": str(e)}


def analyze_lead_hole_alignment(yolo_result: dict, img_width: int = 448, img_height: int = 448) -> dict:
    """
    YOLO 검출 결과를 기반으로 lead-hole 정렬 분석

    Returns:
        {
            "left_lead": {"reaches_hole": bool, "target_hole_idx": int, "distance": float},
            "center_lead": {...},
            "right_lead": {...},
            "alignment_issues": [...]
        }
    """
    holes = yolo_result.get("holes", [])
    leads = yolo_result.get("leads", [])

    # 하단 3개의 홀만 선택 (y 좌표가 큰 것)
    bottom_holes = sorted(holes, key=lambda h: h["y"], reverse=True)[:3]
    bottom_holes.sort(key=lambda h: h["x"])  # x 좌표로 재정렬

    # 리드를 x 좌표로 정렬 (left, center, right)
    sorted_leads = sorted(leads, key=lambda l: l["x"])[:3]

    result = {
        "left_lead": {"reaches_hole": "unknown", "distance": -1},
        "center_lead": {"reaches_hole": "unknown", "distance": -1},
        "right_lead": {"reaches_hole": "unknown", "distance": -1},
        "alignment_issues": []
    }

    lead_names = ["left_lead", "center_lead", "right_lead"]

    for i, lead in enumerate(sorted_leads):
        if i >= 3:
            break

        lead_name = lead_names[i]
        lead_bottom_y = lead["y"] + lead["h"] / 2  # 리드 하단

        # 가장 가까운 홀 찾기
        min_distance = float("inf")
        reaches_hole = False

        if i < len(bottom_holes):
            target_hole = bottom_holes[i]
            hole_center_x = target_hole["x"]
            hole_center_y = target_hole["y"]
            hole_radius = max(target_hole["w"], target_hole["h"]) / 2

            # 리드 끝점과 홀 중심 사이의 거리
            dx = lead["x"] - hole_center_x
            dy = lead_bottom_y - hole_center_y
            distance = (dx**2 + dy**2) ** 0.5

            # 홀 반경의 1.5배 이내면 연결된 것으로 판단
            if distance < hole_radius * 1.5:
                reaches_hole = True
            min_distance = distance

            result[lead_name] = {
                "reaches_hole": "yes" if reaches_hole else "no",
                "distance": min_distance,
                "target_hole_x": hole_center_x,
                "target_hole_y": hole_center_y
            }

            if not reaches_hole:
                result["alignment_issues"].append(f"{lead_name}_missed_hole")
        else:
            result[lead_name] = {"reaches_hole": "unknown", "distance": -1}
            result["alignment_issues"].append(f"{lead_name}_no_target_hole")

    return result


def yolo_to_stage1_hints(yolo_result: dict, alignment: dict) -> str:
    """
    YOLO 검출 결과를 Stage 1 프롬프트 힌트로 변환
    """
    hints = []

    holes = yolo_result.get("holes", [])
    leads = yolo_result.get("leads", [])
    body = yolo_result.get("body")

    hints.append(f"YOLO Detection Results (for reference):")
    hints.append(f"- Detected {len(holes)} holes")
    hints.append(f"- Detected {len(leads)} leads")
    hints.append(f"- Body detected: {'Yes' if body else 'No'}")

    # 정렬 분석 결과
    if alignment.get("alignment_issues"):
        hints.append(f"\nPotential Issues Detected by YOLO:")
        for issue in alignment["alignment_issues"]:
            hints.append(f"  - {issue}")

    # 각 리드의 상태
    hints.append(f"\nLead-Hole Alignment Analysis:")
    for lead_name in ["left_lead", "center_lead", "right_lead"]:
        lead_info = alignment.get(lead_name, {})
        reaches = lead_info.get("reaches_hole", "unknown")
        hints.append(f"  - {lead_name}: reaches_hole={reaches}")

    return "\n".join(hints)


# =========================
# Stage 1: 관찰 프롬프트 (V10 - YOLO 힌트 추가)
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

========== V10: USE YOLO DETECTION HINTS ==========
You will receive YOLO detection hints that indicate:
- Number of detected holes, leads, body
- Potential alignment issues (e.g., lead_missed_hole)
- Lead-hole connection analysis

USE these hints to FOCUS your observation, but VERIFY with the actual image!
The YOLO hints may have errors - always confirm visually.

========== CRITICAL INSPECTION: LEAD-TO-HOLE CONNECTION ==========
For EACH lead (left, center, right), check:
1. Does the lead END inside its target hole? (connected)
2. Does the lead MISS the hole and end elsewhere? (floating)
3. Does the lead touch/attach to a vertical LINE instead of hole? (floating + defect)

========== MANDATORY VERIFICATION STEP ==========
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
- "splayed": Leads spread so wide they miss their holes completely"""

STAGE1_USER_TEMPLATE = """Observe the provided image and fill the JSON below using ONLY visible evidence.
Do NOT conclude normal/abnormal.

{yolo_hints}

IMPORTANT DEFINITIONS:
- HOLE: Dark circular opening at the bottom where lead should INSERT into
- VERTICAL LINE/TRACE: Brown/copper stripe running vertically on the board (NOT a hole)
- "connected": Lead tip is INSIDE the hole
- "floating": Lead tip is NOT inside the hole (missed, bent away, or attached to line instead)

V10 REMINDER - VERIFY YOLO HINTS:
The YOLO detection suggests potential issues. Please VERIFY each one:
{alignment_issues}

Return JSON that matches this exact schema and allowed values:

{{
  "body": {{
    "tilt": {{"value": "none|mild|severe|unknown", "evidence": ""}},
    "rotation_relative_to_vertical_lines": {{"value": "none|mild|severe|unknown", "evidence": ""}},
    "center_offset": {{"value": "centered|left_shift|right_shift|unknown", "evidence": ""}},
    "top_edge_irregularity_or_damage": {{"value": "none|present|unknown", "evidence": ""}},
    "surface_mark_unusual_blob": {{"value": "none|present|unknown", "evidence": ""}}
  }},
  "leads": {{
    "count_visible": {{"value": 0, "evidence": ""}},
    "left_lead": {{
      "visibility": "clear|partial|unclear",
      "shape": "straightish|curved|severely_deformed|unknown",
      "shape_detail": "points_down|bends_left|bends_right|twisted|missing|unknown",
      "length_impression": "normal_like|short_like|unknown",
      "end_position": "in_hole|missed_hole|attached_to_line|unknown",
      "contact_with_board": "connected|floating|unknown",
      "touches_vertical_line": "yes|no|unknown",
      "evidence": ""
    }},
    "center_lead": {{
      "visibility": "clear|partial|unclear",
      "shape": "straightish|curved|blob_like|unknown",
      "shape_detail": "points_down|bends_left|bends_right|twisted|missing|unknown",
      "length_impression": "normal_like|short_like|unknown",
      "end_position": "in_hole|missed_hole|attached_to_line|unknown",
      "contact_with_board": "connected|floating|unknown",
      "touches_vertical_line": "yes|no|unknown",
      "evidence": ""
    }},
    "right_lead": {{
      "visibility": "clear|partial|unclear",
      "shape": "straightish|curved|severely_deformed|unknown",
      "shape_detail": "points_down|bends_left|bends_right|twisted|missing|unknown",
      "length_impression": "normal_like|short_like|unknown",
      "end_position": "in_hole|missed_hole|attached_to_line|unknown",
      "contact_with_board": "connected|floating|unknown",
      "touches_vertical_line": "yes|no|unknown",
      "evidence": ""
    }},
    "symmetry_left_vs_right": {{"value": "symmetric|slightly_asymmetric|asymmetric|unknown", "evidence": ""}},
    "lead_overlap_or_clumping": {{"value": "none|present|unknown", "evidence": ""}},
    "lead_arrangement": {{"value": "parallel|crossed|tangled|splayed|unknown", "evidence": ""}},
    "all_leads_reach_holes": {{"value": "yes|no|unknown", "evidence": ""}}
  }},
  "background_alignment": {{
    "holes_pattern_visible": {{"value": "yes|no|unknown", "evidence": ""}},
    "bottom_three_holes_near_lead_ends": {{"value": "clearly_visible|partially_visible|unclear", "evidence": ""}}
  }},
  "image_quality": {{
    "blur": "low|medium|high",
    "lighting_glare_on_metal": "low|medium|high",
    "occlusion": "none|some|high"
  }},
  "yolo_verification": {{
    "yolo_holes_count_matches": "yes|no|unknown",
    "yolo_leads_count_matches": "yes|no|unknown",
    "yolo_alignment_issues_confirmed": []
  }},
  "notes": ""
}}"""


# =========================
# Stage 2: 판단 프롬프트 (V7과 동일)
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

V10 NOTE: Check yolo_verification field for additional confirmation:
- If yolo_alignment_issues_confirmed contains issues, these are CONFIRMED defects
- Use this to increase confidence in abnormal detection

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

Classification threshold:
- Total score >= 3: label = "abnormal"
- Total score < 3: label = "normal"
- Any CRITICAL RULE match: label = "abnormal" (regardless of score)

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
    """LLM API 호출"""
    payload = {"model": MODEL, "messages": messages,
               "stream": False, "temperature": TEMPERATURE}
    r = requests.post(BRIDGE_URL, headers=HEADERS,
                      json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"status={r.status_code}, body={r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"].strip()


def _safe_json_extract(s: str) -> dict:
    """JSON 파싱"""
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


def should_trigger_recheck(stage2_result: dict, is_original: bool = False, num_results: int = 1) -> Tuple[bool, str]:
    """추가 recheck가 필요한지 판단"""
    confidence = stage2_result.get("confidence", 0.5)
    key_reasons = stage2_result.get("key_reasons", [])
    triggered_checks = stage2_result.get("triggered_checks", "none")

    if triggered_checks != "none":
        return True, f"triggered_checks={triggered_checks}"

    if confidence < THRESHOLDS["confidence_for_recheck"]:
        return True, f"low_confidence={confidence:.2f}"

    if THRESHOLDS["suspicious_always_recheck"] and has_suspicious_reasons(key_reasons):
        return True, "suspicious_reasons"

    if num_results < THRESHOLDS["min_votes_for_decision"]:
        return True, f"insufficient_votes={num_results}"

    return False, "no_recheck_needed"


@traceable(name="Preprocess Image")
def preprocess_image_for_focus(img_url: str, focus_action: str, verbose: bool = False) -> str:
    """Focus 액션에 맞는 전처리"""
    if focus_action not in FOCUS_TO_TOOL:
        return img_url

    tool = FOCUS_TO_TOOL[focus_action]

    try:
        result = tool.invoke({"image_source": img_url, "output_path": None})
        return result
    except Exception as e:
        if verbose:
            print(f"    [전처리] 실패: {e}")
        return img_url


@traceable(name="Stage1 Observe V10 with YOLO")
def stage1_observe(img_url: str, yolo_result: dict = None, alignment: dict = None, verbose: bool = False) -> dict:
    """Stage 1: 이미지 관찰 (YOLO 힌트 포함)"""

    # YOLO 힌트 생성
    if yolo_result and yolo_result.get("detection_success"):
        yolo_hints = yolo_to_stage1_hints(yolo_result, alignment or {})
        alignment_issues = alignment.get("alignment_issues", []) if alignment else []
        alignment_str = "\n".join([f"  - {issue}" for issue in alignment_issues]) if alignment_issues else "  - None detected"
    else:
        yolo_hints = "YOLO Detection: Not available (using visual inspection only)"
        alignment_str = "  - No YOLO data available"

    user_prompt = STAGE1_USER_TEMPLATE.format(
        yolo_hints=yolo_hints,
        alignment_issues=alignment_str
    )

    content = _post_chat([
        {"role": "system", "content": STAGE1_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": img_url}},
        ]},
    ])

    if verbose:
        print(f"    [Stage1 응답]\n{content[:500]}...")

    obs = _safe_json_extract(content)
    return obs


@traceable(name="Stage2 Decide")
def stage2_decide(stage1_json: dict, verbose: bool = False) -> dict:
    """Stage 2: 판단"""
    stage1_str = json.dumps(stage1_json, indent=2, ensure_ascii=False)
    user_prompt = STAGE2_USER_TEMPLATE + stage1_str

    content = _post_chat([
        {"role": "system", "content": STAGE2_SYSTEM},
        {"role": "user", "content": user_prompt},
    ])

    if verbose:
        print(f"    [Stage2 응답]\n{content}")

    decision = _safe_json_extract(content)
    return decision


@traceable(name="Stage1 Recheck with Preprocessing V10")
def stage1_recheck_with_preprocessing(img_url: str, focus_action: str, yolo_result: dict = None, alignment: dict = None, verbose: bool = False) -> dict:
    """Stage 1 재검사: 전처리 + YOLO 힌트"""
    preprocessed_url = preprocess_image_for_focus(img_url, focus_action, verbose)

    # YOLO 힌트
    if yolo_result and yolo_result.get("detection_success"):
        yolo_hints = yolo_to_stage1_hints(yolo_result, alignment or {})
        alignment_issues = alignment.get("alignment_issues", []) if alignment else []
        alignment_str = "\n".join([f"  - {issue}" for issue in alignment_issues]) if alignment_issues else "  - None"
    else:
        yolo_hints = "YOLO Detection: Not available"
        alignment_str = "  - N/A"

    focus_prompts = {
        "recheck_leads_focus": "Focus specifically on the THREE METAL LEADS. VERIFY: Does each lead tip actually END INSIDE its target hole? ",
        "recheck_body_alignment": "Focus specifically on the BLACK BODY component. Check tilt, rotation, center offset, and any surface damage. ",
        "patch_recheck_leads": "Focus on the LEAD TIPS and their position relative to the bottom holes. CRITICAL: Check if tips are INSIDE the holes or BESIDE them. ",
        "dual_model_check": "Perform a detailed comprehensive check of ALL components. VERIFY each lead's end_position carefully. ",
    }

    extra_focus = focus_prompts.get(focus_action, "")
    user_prompt = extra_focus + STAGE1_USER_TEMPLATE.format(
        yolo_hints=yolo_hints,
        alignment_issues=alignment_str
    )

    content = _post_chat([
        {"role": "system", "content": STAGE1_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": preprocessed_url}},
        ]},
    ])

    obs = _safe_json_extract(content)
    return obs


def vote_decision(results: List[dict], verbose: bool = False, result_metadata: List[dict] = None) -> Tuple[str, float, List[str], bool]:
    """가중치 기반 투표"""
    if not results:
        return "normal", 0.5, [], False

    if result_metadata is None:
        result_metadata = [{"is_original": (i == 0)} for i in range(len(results))]

    abnormal_weight = 0.0
    normal_weight = 0.0
    all_reasons = []

    for i, result in enumerate(results):
        label = result.get("label", "normal")
        key_reasons = result.get("key_reasons", [])
        confidence = result.get("confidence", 0.5)
        meta = result_metadata[i] if i < len(result_metadata) else {"is_original": False}

        weight = WEIGHTS["recheck"]

        if meta.get("is_original", False):
            weight = WEIGHTS["original"]

        if has_critical_reasons(key_reasons):
            weight *= WEIGHTS["critical_reason"]
        elif has_suspicious_reasons(key_reasons):
            weight *= WEIGHTS["suspicious_reason"]

        weight *= confidence

        if label == "abnormal":
            abnormal_weight += weight
            all_reasons.extend(key_reasons)
        else:
            normal_weight += weight

    weight_diff = abs(abnormal_weight - normal_weight)
    is_tie = weight_diff < 0.1

    if abnormal_weight > normal_weight:
        final_label = "abnormal"
    elif normal_weight > abnormal_weight:
        final_label = "normal"
    else:
        final_label = "abnormal"

    total_weight = abnormal_weight + normal_weight
    if final_label == "abnormal":
        final_confidence = abnormal_weight / total_weight if total_weight > 0 else 0.5
    else:
        final_confidence = normal_weight / total_weight if total_weight > 0 else 0.5

    return final_label, final_confidence, all_reasons, is_tie


@traceable(name="Two-Stage Classify Agent V10", run_type="chain")
def classify_agent(img_url: str, img_id: str = None, max_retries: int = 3, verbose: bool = False) -> Tuple[int, bool, str]:
    """
    Two-Stage Agent V10 - YOLO Detection + 가중치 투표

    V7 대비 변경:
    1. Stage 0: YOLO 검출 먼저 수행
    2. Stage 1: YOLO 힌트를 참고하여 관찰
    3. Stage 2: 기존 V7 로직
    """
    recheck_count = 0
    used_focuses = []
    all_results = []
    result_metadata = []

    for attempt in range(max_retries):
        try:
            # ========== Stage 0: YOLO Detection ==========
            if verbose:
                print(f"  [Stage 0] YOLO 검출 시작...")

            yolo_result = stage0_yolo_detect(img_url, verbose=verbose)

            # YOLO 결과 분석
            alignment = None
            if yolo_result.get("detection_success"):
                alignment = analyze_lead_hole_alignment(yolo_result)
                if verbose:
                    print(f"    [Stage0] 정렬 분석: {alignment.get('alignment_issues', [])}")

            # ========== Stage 1: 원본 이미지 관찰 (YOLO 힌트 포함) ==========
            if verbose:
                print(f"  [Stage 1] 이미지 관찰 시작 (YOLO 힌트 포함)...")

            stage1_result = stage1_observe(img_url, yolo_result, alignment, verbose=verbose)

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
                print(f"    → 1차 판정: {original_label}, confidence: {original_confidence:.2f}")

            # ========== 조건부 recheck 로직 (V7과 동일) ==========
            should_recheck, recheck_reason = should_trigger_recheck(
                stage2_result, is_original=True, num_results=len(all_results)
            )

            if original_label == "abnormal" and has_critical_reasons(original_reasons):
                if verbose:
                    print(f"\n  [V10] Critical 근거 abnormal → 추가 검증")

                for focus in FOCUS_SEQUENCE[:2]:
                    if recheck_count >= MAX_RECHECK_COUNT:
                        break

                    recheck_count += 1
                    used_focuses.append(focus)

                    stage1_focus = stage1_recheck_with_preprocessing(
                        img_url, focus, yolo_result, alignment, verbose=verbose)
                    stage2_focus = stage2_decide(stage1_focus, verbose=verbose)
                    all_results.append(stage2_focus)
                    result_metadata.append({"is_original": False, "focus": focus})

            elif should_recheck:
                if verbose:
                    print(f"\n  [V10] 조건부 recheck: {recheck_reason}")

                first_focus = triggered_checks if triggered_checks != "none" else "recheck_leads_focus"
                recheck_count += 1
                used_focuses.append(first_focus)

                stage1_recheck = stage1_recheck_with_preprocessing(
                    img_url, first_focus, yolo_result, alignment, verbose=verbose)
                stage2_recheck = stage2_decide(stage1_recheck, verbose=verbose)
                all_results.append(stage2_recheck)
                result_metadata.append({"is_original": False, "focus": first_focus})

                recheck_label = stage2_recheck.get("label", "normal")

                if original_label != recheck_label and recheck_count < MAX_RECHECK_COUNT:
                    for focus in FOCUS_SEQUENCE:
                        if focus not in used_focuses and recheck_count < MAX_RECHECK_COUNT:
                            recheck_count += 1
                            used_focuses.append(focus)

                            stage1_focus = stage1_recheck_with_preprocessing(
                                img_url, focus, yolo_result, alignment, verbose=verbose)
                            stage2_focus = stage2_decide(stage1_focus, verbose=verbose)
                            all_results.append(stage2_focus)
                            result_metadata.append({"is_original": False, "focus": focus})
                            break

            # ========== 가중치 투표 ==========
            final_label, final_confidence, critical_reasons, is_tie = vote_decision(
                all_results, verbose=verbose, result_metadata=result_metadata
            )

            # 동점 처리
            while is_tie and recheck_count < MAX_RECHECK_COUNT:
                next_focus = None
                for focus in FOCUS_SEQUENCE:
                    if focus not in used_focuses:
                        next_focus = focus
                        break

                if next_focus is None:
                    break

                recheck_count += 1
                used_focuses.append(next_focus)

                stage1_tiebreak = stage1_recheck_with_preprocessing(
                    img_url, next_focus, yolo_result, alignment, verbose=verbose)
                stage2_tiebreak = stage2_decide(stage1_tiebreak, verbose=verbose)
                all_results.append(stage2_tiebreak)
                result_metadata.append({"is_original": False, "focus": next_focus})

                final_label, final_confidence, critical_reasons, is_tie = vote_decision(
                    all_results, verbose=verbose, result_metadata=result_metadata
                )

            # ========== 최종 결정 ==========
            needs_review = False
            review_message = ""

            labels = [r.get("label") for r in all_results]
            if len(set(labels)) > 1:
                needs_review = True
                abnormal_count = sum(1 for l in labels if l == "abnormal")
                review_message = f"[판정 충돌] {abnormal_count}/{len(labels)} abnormal"

            final_label_int = LABEL_ABNORMAL if final_label == "abnormal" else LABEL_NORMAL

            if verbose:
                result_text = '불량(1)' if final_label_int == 1 else '정상(0)'
                print(f"\n  [최종] {result_text} (conf: {final_confidence:.2f})")
                print(f"  [통계] YOLO: {'성공' if yolo_result.get('detection_success') else '실패'}, recheck: {recheck_count}")

            return final_label_int, needs_review, review_message

        except Exception as e:
            if verbose:
                print(f"  [오류] attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep((0.5 * (2 ** attempt)) + random.uniform(0, 0.2))

    return LABEL_NORMAL, True, "최대 재시도 횟수 초과"


def process_single_image(args: Tuple[int, str, str, int, bool]) -> dict:
    """단일 이미지 처리"""
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
        result["review_message"] = f"오류: {str(e)}"

    return result


@traceable(name="Main Pipeline V10 YOLO+LLM", run_type="chain")
def main():
    """메인 파이프라인 V10"""
    MAX_WORKERS = 20
    VERBOSE = False

    print("=" * 60)
    print("Two-Stage Agent V10 (YOLO Detection + LLM)")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"YOLO: {'Finetuned' if ROBOFLOW_MODEL_ID else 'Workflow'}")
    print(f"Max Workers: {MAX_WORKERS}")
    print()

    test_df = pd.read_csv(TEST_CSV_PATH)

    if "id" not in test_df.columns or "img_url" not in test_df.columns:
        raise ValueError(f"columns: {test_df.columns.tolist()}")

    n = len(test_df)
    print(f"총 {n}개 이미지 처리 시작...\n")

    tasks = [
        (i, row["id"], row["img_url"], n, VERBOSE)
        for i, row in test_df.iterrows()
    ]

    results = []
    review_items = []

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

    results.sort(key=lambda x: x["id"])

    preds = [{"id": r["id"], "label": r["label"]} for r in results]
    out_df = pd.DataFrame(preds, columns=["id", "label"])
    out_df.to_csv(OUT_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"완료! 소요 시간: {elapsed_time:.1f}초")
    print(f"Saved: {OUT_PATH}")

    abnormal_count = sum(1 for r in results if r["label"] == 1)
    normal_count = sum(1 for r in results if r["label"] == 0)
    print(f"\n[결과] 정상: {normal_count}개, 불량: {abnormal_count}개")

    if review_items:
        print(f"\n[검토 필요] 총 {len(review_items)}개")
        for item in review_items[:5]:
            print(f"  - {item['id']}: {item['message']}")

    return out_df, review_items


if __name__ == "__main__":
    main()
