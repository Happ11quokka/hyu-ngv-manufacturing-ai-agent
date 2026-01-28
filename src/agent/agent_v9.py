"""
AI Agent with LangSmith Tracing V9
Two-Stage Prompt + OpenCV 위치 분석 → LLM 제공 + 안정성 개선

V7 대비 변경사항 (V9):
1. OpenCV로 홀 위치 검출 (HoughCircles)
2. OpenCV로 리드 끝점 검출 (색상 기반 + 컨투어)
3. 리드-홀 간 거리 계산 (픽셀 단위)
4. CV 분석 결과를 Stage1 프롬프트에 추가 제공
   → LLM이 "curved = missed_hole"로 잘못 판단하는 문제 해결

V9 안정성 개선사항:
1. 이미지 로드 재시도 로직 (최대 3회)
   - 응답 내용 유효성 검사
   - 이미지 크기 검사
2. API 응답 검증 강화
   - JSON 응답 구조 검증
   - 타임아웃 및 재시도 로직
3. 오류 메시지 상세화
   - 오류 유형별 구분 (ValueError, RuntimeError 등)
   - 디버깅에 유용한 상세 정보 출력
"""

import os
import sys
import re
import json
import time
import random
import requests
import pandas as pd
import cv2
import numpy as np
from dotenv import load_dotenv
from langsmith import traceable
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# 프로젝트 루트 경로 계산
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

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
MAX_RECHECK_COUNT = 3

# =========================
# CV 분석 설정 (이미지 크기 고정 가정)
# =========================
CV_CONFIG = {
    # 홀 검출 파라미터
    "hole_detection": {
        "dp": 1.2,
        "minDist": 30,
        "param1": 50,
        "param2": 25,
        "minRadius": 8,
        "maxRadius": 25,
    },
    # 홀 검색 영역 (이미지 비율 기준) - 하단 영역
    "hole_region": {
        "top": 0.65,
        "bottom": 0.95,
        "left": 0.15,
        "right": 0.85,
    },
    # 리드 끝점 검색 영역 - 홀 바로 위
    "lead_tip_region": {
        "top": 0.50,
        "bottom": 0.85,
        "left": 0.20,
        "right": 0.80,
    },
    # 리드 색상 범위 (HSV) - 은색/회색 금속
    "lead_color_hsv": {
        "lower": (0, 0, 120),
        "upper": (180, 60, 255),
    },
    # in_hole 판정 거리 임계값 (픽셀)
    "in_hole_threshold": 15,
    # 의심 거리 임계값 (픽셀)
    "suspicious_threshold": 25,
}


# =========================
# OpenCV 기반 분석 함수
# =========================

def load_image_from_url(url: str, max_retries: int = 3) -> np.ndarray:
    """URL에서 이미지를 로드하여 OpenCV 형식(BGR)으로 반환

    V9 개선사항:
    - 최대 3회 재시도 로직
    - 응답 내용 유효성 검사
    - 이미지 크기 검사

    Args:
        url: 이미지 URL
        max_retries: 최대 재시도 횟수

    Returns:
        OpenCV BGR 형식의 이미지 배열

    Raises:
        ValueError: 이미지 로드 또는 디코딩 실패 시
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # 응답 내용이 비어있는지 확인
            if not response.content or len(response.content) < 100:
                raise ValueError(f"Empty or too small response content (size: {len(response.content)})")

            img_array = np.frombuffer(response.content, np.uint8)

            # 배열 크기 확인
            if img_array.size < 100:
                raise ValueError(f"Image array too small (size: {img_array.size})")

            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("cv2.imdecode returned None")

            # 이미지 크기 확인
            if img.shape[0] < 10 or img.shape[1] < 10:
                raise ValueError(f"Decoded image too small: {img.shape}")

            return img

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))  # 점진적 대기
                continue

    raise ValueError(f"Failed to load image from URL after {max_retries} attempts: {url}, last error: {last_error}")


def detect_holes(img: np.ndarray) -> List[Dict]:
    """
    HoughCircles를 사용하여 홀(구멍) 검출

    Returns:
        List of {"x": int, "y": int, "radius": int} sorted by x coordinate (left to right)
    """
    h, w = img.shape[:2]
    config = CV_CONFIG["hole_detection"]
    region = CV_CONFIG["hole_region"]

    # 홀 검색 영역 크롭
    top = int(h * region["top"])
    bottom = int(h * region["bottom"])
    left = int(w * region["left"])
    right = int(w * region["right"])
    roi = img[top:bottom, left:right]

    # 그레이스케일 변환 및 블러
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 원 검출
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=config["dp"],
        minDist=config["minDist"],
        param1=config["param1"],
        param2=config["param2"],
        minRadius=config["minRadius"],
        maxRadius=config["maxRadius"],
    )

    holes = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0]:
            # ROI 좌표를 원본 이미지 좌표로 변환
            holes.append({
                "x": int(c[0]) + left,
                "y": int(c[1]) + top,
                "radius": int(c[2]),
            })

    # x 좌표로 정렬 (왼쪽 → 오른쪽)
    holes.sort(key=lambda h: h["x"])

    # 상위 3개만 반환 (3개 홀 예상)
    return holes[:3]


def detect_lead_tips(img: np.ndarray) -> List[Dict]:
    """
    색상 기반으로 리드(금속) 끝점 검출

    Returns:
        List of {"x": int, "y": int, "area": int} sorted by x coordinate (left to right)
    """
    h, w = img.shape[:2]
    region = CV_CONFIG["lead_tip_region"]
    color_range = CV_CONFIG["lead_color_hsv"]

    # 리드 검색 영역 크롭
    top = int(h * region["top"])
    bottom = int(h * region["bottom"])
    left = int(w * region["left"])
    right = int(w * region["right"])
    roi = img[top:bottom, left:right]

    # HSV 변환
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 금속(은색/회색) 색상 마스크
    mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])

    # 모폴로지 연산으로 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 컨투어 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lead_tips = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # 너무 작은 영역 무시
            continue

        # 바운딩 박스
        x, y, bw, bh = cv2.boundingRect(cnt)

        # 세로로 긴 형태 (리드 특성) 또는 일정 크기 이상
        if bh > bw * 0.5 or area > 200:
            # 컨투어에서 가장 아래쪽 점 (리드 끝점)
            bottom_point = tuple(cnt[cnt[:, :, 1].argmax()][0])

            lead_tips.append({
                "x": int(bottom_point[0]) + left,
                "y": int(bottom_point[1]) + top,
                "area": int(area),
                "bbox": {"x": x + left, "y": y + top, "w": bw, "h": bh},
            })

    # x 좌표로 정렬 (왼쪽 → 오른쪽)
    lead_tips.sort(key=lambda t: t["x"])

    # 3개 리드에 해당하는 영역으로 그룹화
    if len(lead_tips) >= 3:
        # 가장 큰 3개 또는 x 위치로 그룹화
        roi_width = right - left
        third = roi_width // 3

        grouped = {"left": [], "center": [], "right": []}
        for tip in lead_tips:
            rel_x = tip["x"] - left
            if rel_x < third:
                grouped["left"].append(tip)
            elif rel_x < third * 2:
                grouped["center"].append(tip)
            else:
                grouped["right"].append(tip)

        # 각 그룹에서 가장 아래쪽 점 선택
        result = []
        for key in ["left", "center", "right"]:
            if grouped[key]:
                # y가 가장 큰 점 (가장 아래)
                best = max(grouped[key], key=lambda t: t["y"])
                result.append(best)

        return result

    return lead_tips[:3]


def calculate_distances(lead_tips: List[Dict], holes: List[Dict]) -> List[Dict]:
    """
    각 리드 끝점과 가장 가까운 홀 간의 거리 계산

    Returns:
        List of {
            "lead_position": "left"|"center"|"right",
            "lead_tip": {"x": int, "y": int},
            "nearest_hole": {"x": int, "y": int},
            "distance_px": float,
            "status": "in_hole"|"near_hole"|"missed_hole"|"unknown"
        }
    """
    positions = ["left", "center", "right"]
    results = []

    for i, pos in enumerate(positions):
        result = {
            "lead_position": pos,
            "lead_tip": None,
            "nearest_hole": None,
            "distance_px": None,
            "status": "unknown",
        }

        # 리드 끝점이 있는 경우
        if i < len(lead_tips):
            tip = lead_tips[i]
            result["lead_tip"] = {"x": tip["x"], "y": tip["y"]}

            # 가장 가까운 홀 찾기
            if holes:
                min_dist = float("inf")
                nearest = None

                for hole in holes:
                    dist = np.sqrt((tip["x"] - hole["x"])**2 + (tip["y"] - hole["y"])**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = hole

                if nearest:
                    result["nearest_hole"] = {"x": nearest["x"], "y": nearest["y"]}
                    result["distance_px"] = round(min_dist, 1)

                    # 상태 판정
                    if min_dist <= CV_CONFIG["in_hole_threshold"]:
                        result["status"] = "in_hole"
                    elif min_dist <= CV_CONFIG["suspicious_threshold"]:
                        result["status"] = "near_hole"
                    else:
                        result["status"] = "missed_hole"

        results.append(result)

    return results


@traceable(name="CV Analysis")
def analyze_image_cv(img_url: str) -> Dict:
    """
    OpenCV를 사용하여 이미지 분석

    Returns:
        {
            "success": bool,
            "holes_detected": int,
            "leads_detected": int,
            "analysis": [
                {"lead_position": "left", "distance_px": 5.2, "status": "in_hole", ...},
                ...
            ],
            "summary": "all_in_hole" | "some_missed" | "analysis_failed",
            "details": str (LLM에 제공할 텍스트)
        }
    """
    try:
        img = load_image_from_url(img_url)

        # 홀 검출
        holes = detect_holes(img)

        # 리드 끝점 검출
        lead_tips = detect_lead_tips(img)

        # 거리 계산
        distances = calculate_distances(lead_tips, holes)

        # 요약 생성
        statuses = [d["status"] for d in distances]
        if all(s == "in_hole" for s in statuses):
            summary = "all_in_hole"
        elif any(s == "missed_hole" for s in statuses):
            summary = "some_missed"
        elif any(s == "near_hole" for s in statuses):
            summary = "some_near"
        else:
            summary = "uncertain"

        # LLM에 제공할 상세 텍스트 생성
        details_lines = [
            f"[CV Analysis Result]",
            f"- Holes detected: {len(holes)}",
            f"- Lead tips detected: {len(lead_tips)}",
            f"- Distance threshold for 'in_hole': {CV_CONFIG['in_hole_threshold']}px",
            f"",
        ]

        for d in distances:
            pos = d["lead_position"].upper()
            if d["distance_px"] is not None:
                details_lines.append(
                    f"- {pos} lead: distance={d['distance_px']}px → {d['status']}"
                )
            else:
                details_lines.append(f"- {pos} lead: not detected")

        details_lines.append(f"")
        details_lines.append(f"CV Summary: {summary}")

        return {
            "success": True,
            "holes_detected": len(holes),
            "leads_detected": len(lead_tips),
            "analysis": distances,
            "summary": summary,
            "details": "\n".join(details_lines),
        }

    except Exception as e:
        return {
            "success": False,
            "holes_detected": 0,
            "leads_detected": 0,
            "analysis": [],
            "summary": "analysis_failed",
            "details": f"[CV Analysis Failed: {str(e)}]",
        }


# =========================
# Stage 1: 관찰 전용 프롬프트 (V9 - CV 분석 결과 포함)
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

========== V9: USE CV ANALYSIS AS REFERENCE ==========
You will receive CV (Computer Vision) analysis results that measured the ACTUAL DISTANCE
between each lead tip and its nearest hole in pixels.

IMPORTANT: Use this CV data as an OBJECTIVE reference!
- If CV says "distance=5px, status=in_hole" → The lead IS in the hole
- If CV says "distance=30px, status=missed_hole" → The lead MISSED the hole
- CV analysis is more reliable than visual estimation for distances

However, CV can fail or be inaccurate. Cross-check with what you see:
- If CV says "in_hole" but you clearly see the lead is NOT in hole → report what you see
- If CV analysis failed → rely entirely on visual inspection

========== LEAD-TO-HOLE CONNECTION ==========
For EACH lead (left, center, right), check:
1. Does the lead END inside its target hole? (connected)
2. Does the lead MISS the hole and end elsewhere? (floating)
3. Does the lead touch/attach to a vertical LINE instead of hole? (floating + defect)

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

{cv_analysis}

IMPORTANT DEFINITIONS:
- HOLE: Dark circular opening at the bottom where lead should INSERT into
- VERTICAL LINE/TRACE: Brown/copper stripe running vertically on the board (NOT a hole)
- "connected": Lead tip is INSIDE the hole
- "floating": Lead tip is NOT inside the hole (missed, bent away, or attached to line instead)

V9 GUIDANCE - USE CV ANALYSIS:
The CV analysis above measured actual pixel distances. Use this as your primary reference:
- distance < 15px = "in_hole" (lead tip is inside the hole)
- distance 15-25px = "near_hole" (borderline, check visually)
- distance > 25px = "missed_hole" (lead clearly missed)

If CV says a lead is "in_hole", trust it unless you see clear evidence otherwise.
Curved leads CAN be "in_hole" if their TIP is inside - CV measures the TIP position!

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
  "notes": ""
}}"""

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
- Output MUST be valid JSON only (no markdown, no extra text).

Rules:
- If multiple Stage 1 fields are "unknown/unclear", lower confidence.
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

- LOW-signal (NOT abnormal alone):
  13) bends_left or bends_right ALONE = normal variation (curved leads can still be in_hole!)
  14) body.surface_mark_unusual_blob = often false positive, verify carefully

Classification:
- If ANY CRITICAL indicator is present → label = "abnormal"
- Otherwise → label = "normal"

Confidence:
- High confidence (0.90+): Clear evidence, no ambiguity
- Medium confidence (0.70-0.89): Some uncertainty but clear trend
- Low confidence (<0.70): Multiple unknowns or conflicting evidence

Return JSON in this exact schema:

{
  "label": "normal|abnormal",
  "confidence": 0.00,
  "key_reasons": ["...","...","..."],
  "termination_reason": "high_confidence|low_confidence"
}

Stage 1 JSON:
"""


# =========================
# LangSmith Traceable 함수들
# =========================

@traceable(name="LLM API Call V9")
def _post_chat(messages, timeout=90, max_retries=2):
    """LLM API 호출 - LangSmith에서 추적됨

    V9 개선사항:
    - 응답 JSON 구조 검증
    - 타임아웃 및 재시도 로직
    - 상세한 오류 메시지

    Args:
        messages: 대화 메시지 리스트
        timeout: 요청 타임아웃 (초)
        max_retries: 최대 재시도 횟수

    Returns:
        LLM 응답 텍스트

    Raises:
        RuntimeError: API 호출 실패 시
    """
    payload = {"model": MODEL, "messages": messages,
               "stream": False, "temperature": TEMPERATURE}

    last_error = None

    for attempt in range(max_retries):
        try:
            r = requests.post(BRIDGE_URL, headers=HEADERS,
                              json=payload, timeout=timeout)

            if r.status_code != 200:
                raise RuntimeError(f"status={r.status_code}, body={r.text[:300]}")

            # 응답 JSON 파싱
            try:
                response_json = r.json()
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse API response as JSON: {e}, response: {r.text[:300]}")

            # 응답 구조 검증
            if "choices" not in response_json:
                raise RuntimeError(f"API response missing 'choices': {r.text[:300]}")

            if not response_json["choices"]:
                raise RuntimeError(f"API response 'choices' is empty: {r.text[:300]}")

            if "message" not in response_json["choices"][0]:
                raise RuntimeError(f"API response missing 'message': {r.text[:300]}")

            if "content" not in response_json["choices"][0]["message"]:
                raise RuntimeError(f"API response missing 'content': {r.text[:300]}")

            content = response_json["choices"][0]["message"]["content"]

            if content is None:
                raise RuntimeError("API response content is None")

            return content.strip()

        except requests.exceptions.Timeout:
            last_error = f"Request timeout after {timeout}s"
        except requests.exceptions.RequestException as e:
            last_error = f"Request failed: {e}"
        except RuntimeError as e:
            last_error = str(e)
            # API 응답 오류는 재시도 안함
            raise

        if attempt < max_retries - 1:
            time.sleep(1.0 * (attempt + 1))

    raise RuntimeError(f"API call failed after {max_retries} attempts: {last_error}")


def _safe_json_extract(s: str) -> dict:
    """JSON 파싱 (LLM 출력에서 JSON 추출)

    V9 개선사항:
    - 반환값이 딕셔너리인지 검증
    - 더 상세한 오류 메시지
    """
    if not s or not s.strip():
        raise ValueError("Empty response from LLM")

    # 1. 마크다운 코드 블록 제거
    s = re.sub(r"```json\s*", "", s)
    s = re.sub(r"```\s*", "", s)
    s = s.strip()

    result = None

    # 2. 직접 파싱 시도
    try:
        result = json.loads(s)
    except json.JSONDecodeError:
        pass

    # 3. 중첩 JSON 추출 (balanced braces)
    if result is None:
        start = s.find("{")
        if start == -1:
            raise ValueError(f"JSON parse failed - no opening brace in response: {s[:300]}")

        depth = 0
        end = start
        for i, c in enumerate(s[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if depth != 0:
            raise ValueError(f"JSON parse failed - unbalanced braces: {s[:300]}")

        json_str = s[start:end]

        try:
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parse failed: {e}, content: {json_str[:300]}")

    # 4. 결과 검증 - 반드시 딕셔너리여야 함
    if not isinstance(result, dict):
        raise ValueError(f"Parsed JSON is not a dict but {type(result).__name__}: {str(result)[:300]}")

    # 5. Stage 1 응답의 경우 'body' 키가 있어야 함 (기본 검증)
    # Stage 2 응답의 경우 'label' 키가 있어야 함
    # 둘 다 없으면 잘못된 응답
    if "body" not in result and "label" not in result:
        raise ValueError(f"Parsed JSON missing expected keys (body or label): {list(result.keys())[:10]}")

    return result


@traceable(name="Stage1 Observe V9")
def stage1_observe(img_url: str, cv_result: Dict, verbose: bool = False) -> dict:
    """Stage 1: 이미지 관찰 - CV 분석 결과와 함께 (V9)"""

    # CV 분석 결과를 프롬프트에 포함
    cv_text = cv_result.get("details", "[CV Analysis not available]")
    user_prompt = STAGE1_USER_TEMPLATE.format(cv_analysis=cv_text)

    content = _post_chat([
        {"role": "system", "content": STAGE1_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": img_url}},
        ]},
    ])

    if verbose:
        print(f"    [Stage1 원본 응답]\n{content[:500]}...")

    try:
        obs = _safe_json_extract(content)
    except ValueError as e:
        # Stage 1 파싱 실패 시 더 상세한 정보 제공
        raise ValueError(f"Stage1 JSON parse error: {e}, raw response: {content[:500]}")

    # Stage 1 결과 기본 검증
    if "body" not in obs:
        raise ValueError(f"Stage1 response missing 'body' key: {list(obs.keys())}")
    if "leads" not in obs:
        raise ValueError(f"Stage1 response missing 'leads' key: {list(obs.keys())}")

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

    try:
        decision = _safe_json_extract(content)
    except ValueError as e:
        # Stage 2 파싱 실패 시 더 상세한 정보 제공
        raise ValueError(f"Stage2 JSON parse error: {e}, raw response: {content[:500]}")

    # Stage 2 결과 기본 검증
    if "label" not in decision:
        raise ValueError(f"Stage2 response missing 'label' key: {list(decision.keys())}")

    # label 값 검증
    label = decision.get("label", "")
    if label not in ("normal", "abnormal"):
        raise ValueError(f"Stage2 'label' has invalid value: {label}")

    if verbose:
        print(f"    [Stage2 파싱 완료]")

    return decision


@traceable(name="Two-Stage Classify Agent V9", run_type="chain")
def classify_agent(
    img_url: str,
    img_id: str = None,
    max_retries: int = 3,
    verbose: bool = False
) -> Tuple[int, bool, str]:
    """
    Two-Stage Agent V9 - OpenCV 위치 분석 + LLM 판단

    흐름:
    1. CV 분석: 홀 검출, 리드 끝점 검출, 거리 계산
    2. Stage 1: CV 결과와 함께 이미지 관찰
    3. Stage 2: 관찰 결과 기반 판단
    """
    for attempt in range(max_retries):
        try:
            # ========== CV 분석 ==========
            if verbose:
                print(f"  [CV 분석] 시작...")

            cv_result = analyze_image_cv(img_url)

            if verbose:
                print(f"    → 홀 검출: {cv_result['holes_detected']}개")
                print(f"    → 리드 검출: {cv_result['leads_detected']}개")
                print(f"    → 요약: {cv_result['summary']}")
                for a in cv_result.get("analysis", []):
                    print(f"    → {a['lead_position']}: {a['distance_px']}px, {a['status']}")

            # ========== Stage 1: CV 결과와 함께 관찰 ==========
            if verbose:
                print(f"  [Stage 1] 이미지 관찰 시작...")

            stage1_result = stage1_observe(img_url, cv_result, verbose=verbose)

            # ========== Stage 2: 판단 ==========
            if verbose:
                print(f"  [Stage 2] 판단 시작...")

            stage2_result = stage2_decide(stage1_result, verbose=verbose)

            label = stage2_result.get("label", "normal")
            confidence = stage2_result.get("confidence", 0.5)
            key_reasons = stage2_result.get("key_reasons", [])

            if verbose:
                print(f"    → 판정: {label}, confidence: {confidence:.2f}")
                print(f"    → 근거: {key_reasons}")

            # ========== 최종 결정 ==========
            final_label_int = LABEL_ABNORMAL if label == "abnormal" else LABEL_NORMAL

            # CV와 LLM 결과 비교 (검토 필요 여부)
            needs_review = False
            review_message = ""

            cv_summary = cv_result.get("summary", "unknown")
            if cv_summary == "all_in_hole" and label == "abnormal":
                needs_review = True
                review_message = f"[CV-LLM 충돌] CV=all_in_hole, LLM=abnormal"
            elif cv_summary == "some_missed" and label == "normal":
                needs_review = True
                review_message = f"[CV-LLM 충돌] CV=some_missed, LLM=normal"

            if verbose:
                result_text = '불량(1)' if final_label_int == 1 else '정상(0)'
                print(f"\n  [최종 결정] {result_text} (confidence: {confidence:.2f})")
                if needs_review:
                    print(f"  [!] {review_message}")

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
    """단일 이미지 처리 함수 (병렬 처리용)

    V9 개선사항:
    - 오류 유형별 상세 메시지 출력

    Args:
        args: (인덱스, 이미지 ID, 이미지 URL, 총 개수, verbose 플래그)

    Returns:
        결과 딕셔너리
    """
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

    except ValueError as e:
        # JSON 파싱 오류 또는 이미지 처리 오류
        error_msg = str(e)
        if "Stage1 JSON parse" in error_msg:
            print(f"[{idx+1}/{total}] {_id} -> Stage1 JSON 파싱 오류")
        elif "Stage2 JSON parse" in error_msg:
            print(f"[{idx+1}/{total}] {_id} -> Stage2 JSON 파싱 오류")
        elif "Stage1 response missing" in error_msg or "Stage2 response missing" in error_msg:
            print(f"[{idx+1}/{total}] {_id} -> LLM 응답 형식 오류: {error_msg[:150]}")
        elif "Failed to load image" in error_msg:
            print(f"[{idx+1}/{total}] {_id} -> 이미지 로드 오류")
        else:
            print(f"[{idx+1}/{total}] {_id} -> ValueError: {error_msg[:150]}")
        result["needs_review"] = True
        result["review_message"] = f"ValueError: {error_msg[:300]}"

    except RuntimeError as e:
        # API 호출 오류
        error_msg = str(e)
        print(f"[{idx+1}/{total}] {_id} -> API 오류: {error_msg[:150]}")
        result["needs_review"] = True
        result["review_message"] = f"RuntimeError: {error_msg[:300]}"

    except KeyError as e:
        # 딕셔너리 키 접근 오류 - 이전에 발생했던 문제
        error_msg = str(e)
        print(f"[{idx+1}/{total}] {_id} -> KeyError (JSON 구조 오류): {error_msg}")
        result["needs_review"] = True
        result["review_message"] = f"KeyError: {error_msg}"

    except Exception as e:
        # 기타 예상치 못한 오류
        error_msg = str(e)
        error_type = type(e).__name__
        print(f"[{idx+1}/{total}] {_id} -> {error_type}: {error_msg[:150]}")
        result["needs_review"] = True
        result["review_message"] = f"{error_type}: {error_msg[:300]}"

    return result


@traceable(name="Main Pipeline V9 Parallel", run_type="chain")
def main():
    """메인 파이프라인 V9 - 병렬 처리 버전"""
    # 병렬 처리 설정
    MAX_WORKERS = 20  # 동시 처리할 이미지 수 (API 제한에 따라 조절)
    VERBOSE = False   # 병렬 처리 시 verbose는 False 권장

    print("=" * 60)
    print("Two-Stage Prompt Agent V9 (OpenCV 위치 분석 + LLM)")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Max Workers: {MAX_WORKERS}")
    print(f"CV in_hole threshold: {CV_CONFIG['in_hole_threshold']}px")
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


# =========================
# CV 테스트 함수
# =========================
def test_cv_analysis(img_url: str):
    """단일 이미지에 대한 CV 분석 테스트"""
    print("=" * 60)
    print("CV Analysis Test")
    print("=" * 60)
    print(f"Image: {img_url}")
    print()

    result = analyze_image_cv(img_url)

    print(result["details"])
    print()
    print(f"Raw analysis: {json.dumps(result['analysis'], indent=2)}")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # CV 테스트 모드: python agent_v9.py test
        test_url = "https://cfiles.dacon.co.kr/competitions/236680/TEST_000.png"
        if len(sys.argv) > 2:
            test_url = sys.argv[2]
        test_cv_analysis(test_url)
    else:
        # 전체 실행
        main()
