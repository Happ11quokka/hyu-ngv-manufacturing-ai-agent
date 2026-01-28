"""
이미지 전처리 도구 모듈 (Focus 기반)
반도체 결함 검사 AI Agent가 필요에 따라 호출할 수 있는 맞춤형 전처리 도구

Focus 영역별 도구:
1. preprocess_focus_leads - 리드 전체 영역에 집중 (recheck_leads_focus 대응)
2. preprocess_focus_body - 바디/정렬 영역에 집중 (recheck_body_alignment 대응)
3. preprocess_focus_lead_tips - 리드 끝/홀 연결부에 집중 (patch_recheck_leads 대응)
4. preprocess_full_enhanced - 전체 이미지 종합 강화 (dual_model_check 대응)
5. preprocess_custom - 사용자 지정 영역/옵션 전처리
"""

import os
import cv2
import numpy as np
import base64
import requests
from typing import Optional, Tuple, Dict, Any
from langchain_core.tools import tool


# =========================
# 영역별 크롭 설정 (이미지 비율 기준)
# =========================
FOCUS_REGIONS = {
    # 리드 전체 영역: 바디 아래부터 홀까지
    "leads": {
        "top": 0.20,      # 바디 하단부터
        "bottom": 0.95,   # 홀 포함
        "left": 0.25,
        "right": 0.75,
    },
    # 바디 영역: 검은색 바디 컴포넌트
    "body": {
        "top": 0.0,
        "bottom": 0.45,   # 바디 + 약간의 리드 상단
        "left": 0.20,
        "right": 0.80,
    },
    # 리드 끝부분/홀 연결 영역: 리드가 홀에 들어가는 부분
    "lead_tips": {
        "top": 0.55,      # 리드 중간부터
        "bottom": 1.0,    # 이미지 하단까지
        "left": 0.20,
        "right": 0.80,
    },
    # 전체 이미지 (크롭 없음)
    "full": {
        "top": 0.0,
        "bottom": 1.0,
        "left": 0.0,
        "right": 1.0,
    },
    # 좌측 리드만
    "left_lead": {
        "top": 0.20,
        "bottom": 0.95,
        "left": 0.20,
        "right": 0.45,
    },
    # 중앙 리드만
    "center_lead": {
        "top": 0.20,
        "bottom": 0.95,
        "left": 0.35,
        "right": 0.65,
    },
    # 우측 리드만
    "right_lead": {
        "top": 0.20,
        "bottom": 0.95,
        "left": 0.55,
        "right": 0.80,
    },
}

# Focus별 기본 전처리 설정
FOCUS_PRESETS = {
    "leads": {
        "enhance_contrast": True,
        "clip_limit": 2.5,
        "sharpen": True,
        "sharpen_amount": 1.0,
        "denoise": False,
    },
    "body": {
        "enhance_contrast": True,
        "clip_limit": 2.0,
        "sharpen": False,
        "denoise": False,
    },
    "lead_tips": {
        "enhance_contrast": True,
        "clip_limit": 3.0,        # 더 강한 대비 (홀과 리드 구분)
        "sharpen": True,
        "sharpen_amount": 1.5,
        "denoise": False,
    },
    "full": {
        "enhance_contrast": True,
        "clip_limit": 2.0,
        "sharpen": True,
        "sharpen_amount": 0.8,
        "denoise": True,
        "denoise_strength": 5,
    },
}


# =========================
# 유틸리티 함수
# =========================

def load_image_from_url(url: str, max_retries: int = 3) -> np.ndarray:
    """URL에서 이미지를 로드하여 OpenCV 형식(BGR)으로 반환

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
                raise ValueError(f"cv2.imdecode returned None")

            # 이미지 크기 확인
            if img.shape[0] < 10 or img.shape[1] < 10:
                raise ValueError(f"Decoded image too small: {img.shape}")

            return img

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                import time
                time.sleep(0.5 * (attempt + 1))  # 점진적 대기
                continue

    raise ValueError(f"Failed to load image from URL after {max_retries} attempts: {url}, last error: {last_error}")


def load_image_from_path(path: str) -> np.ndarray:
    """로컬 경로에서 이미지를 로드하여 OpenCV 형식(BGR)으로 반환"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image from path: {path}")
    return img


def load_image(source: str) -> np.ndarray:
    """URL 또는 로컬 경로에서 이미지 로드"""
    if source.startswith(('http://', 'https://')):
        return load_image_from_url(source)
    else:
        return load_image_from_path(source)


def save_image(img: np.ndarray, output_path: str) -> str:
    """이미지를 파일로 저장"""
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    cv2.imwrite(output_path, img)
    return output_path


def image_to_base64(img: np.ndarray, format: str = "png") -> str:
    """OpenCV 이미지를 base64 문자열로 변환

    Args:
        img: OpenCV BGR 형식의 이미지 배열
        format: 출력 이미지 형식 ("png", "jpg", "jpeg")

    Returns:
        Base64 인코딩된 문자열

    Raises:
        ValueError: 이미지가 None이거나 인코딩 실패 시
    """
    if img is None:
        raise ValueError("Image is None, cannot encode to base64")

    if img.size == 0:
        raise ValueError("Image is empty, cannot encode to base64")

    # 이미지 유효성 검사
    if len(img.shape) < 2:
        raise ValueError(f"Invalid image shape: {img.shape}")

    # 이미지가 너무 작은 경우 확인
    if img.shape[0] < 1 or img.shape[1] < 1:
        raise ValueError(f"Image dimensions too small: {img.shape}")

    # PNG 인코딩 파라미터 설정 (압축률 최적화)
    encode_params = []
    if format.lower() == "png":
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 6]  # 압축률 0-9
    elif format.lower() in ["jpg", "jpeg"]:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]  # 품질 0-100

    # imencode는 (success, buffer) 튜플 반환
    success, buffer = cv2.imencode(f'.{format}', img, encode_params)

    if not success:
        raise ValueError(f"cv2.imencode failed for format: {format}, image shape: {img.shape}")

    if buffer is None or len(buffer) == 0:
        raise ValueError(f"cv2.imencode returned empty buffer for format: {format}")

    return base64.b64encode(buffer).decode('utf-8')


def base64_to_data_url(base64_str: str, format: str = "png") -> str:
    """base64 문자열을 data URL로 변환"""
    return f"data:image/{format};base64,{base64_str}"


def get_output(img: np.ndarray, output_path: Optional[str]) -> str:
    """이미지를 파일로 저장하거나 data URL로 반환

    Args:
        img: OpenCV BGR 형식의 이미지 배열
        output_path: 저장할 파일 경로 (None이면 data URL 반환)

    Returns:
        파일 경로 또는 data URL

    Raises:
        ValueError: 이미지가 None이거나 처리 실패 시
    """
    if img is None:
        raise ValueError("Cannot output None image")

    if img.size == 0:
        raise ValueError("Cannot output empty image")

    if output_path:
        return save_image(img, output_path)
    else:
        try:
            base64_str = image_to_base64(img)
            return base64_to_data_url(base64_str)
        except Exception as e:
            raise ValueError(f"Failed to convert image to data URL: {e}")


# =========================
# 전처리 핵심 함수
# =========================

def _crop_region(img: np.ndarray, region: Dict[str, float]) -> np.ndarray:
    """지정된 영역으로 이미지 크롭"""
    h, w = img.shape[:2]
    top = int(h * region["top"])
    bottom = int(h * region["bottom"])
    left = int(w * region["left"])
    right = int(w * region["right"])
    return img[top:bottom, left:right].copy()


def _enhance_contrast(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """CLAHE를 사용한 대비 강화"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def _sharpen(img: np.ndarray, amount: float = 1.5) -> np.ndarray:
    """Unsharp Masking 샤프닝"""
    blurred = cv2.GaussianBlur(img, (3, 3), 1.0)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


def _denoise(img: np.ndarray, strength: float = 10) -> np.ndarray:
    """Non-local Means 노이즈 제거"""
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def _detect_edges(img: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Canny 에지 검출"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def _apply_preprocessing(
    img: np.ndarray,
    region_name: str = "full",
    custom_region: Optional[Dict[str, float]] = None,
    enhance_contrast: bool = True,
    clip_limit: float = 2.0,
    sharpen: bool = False,
    sharpen_amount: float = 1.5,
    denoise: bool = False,
    denoise_strength: float = 10,
    detect_edges: bool = False,
) -> np.ndarray:
    """통합 전처리 파이프라인"""
    result = img.copy()

    # 1. 노이즈 제거 (먼저 수행)
    if denoise:
        result = _denoise(result, denoise_strength)

    # 2. 대비 강화
    if enhance_contrast:
        result = _enhance_contrast(result, clip_limit)

    # 3. 샤프닝
    if sharpen:
        result = _sharpen(result, sharpen_amount)

    # 4. 영역 크롭
    if custom_region:
        result = _crop_region(result, custom_region)
    elif region_name in FOCUS_REGIONS:
        result = _crop_region(result, FOCUS_REGIONS[region_name])

    # 5. 에지 검출 (선택적, 마지막에 수행)
    if detect_edges:
        result = _detect_edges(result)

    return result


# =========================
# Focus 기반 LangChain Tools
# =========================

@tool
def preprocess_focus_leads(
    image_source: str,
    output_path: Optional[str] = None,
    enhance_level: str = "normal"
) -> str:
    """
    리드(금속 다리) 영역에 집중한 전처리를 수행합니다.

    [용도] 리드의 형태, 정렬, 변형 여부를 검사할 때 사용합니다.
    - 리드가 똑바로 뻗어있는지
    - 리드 간 간격이 균일한지
    - 리드가 구부러지거나 꺾였는지

    이 도구는 에이전트의 "recheck_leads_focus" 액션과 함께 사용됩니다.

    Args:
        image_source: 이미지 URL 또는 로컬 파일 경로
        output_path: 저장할 파일 경로 (None이면 base64 data URL 반환)
        enhance_level: 강화 수준 ("light", "normal", "strong")

    Returns:
        전처리된 이미지 (파일 경로 또는 data URL)
    """
    clip_limits = {"light": 1.5, "normal": 2.5, "strong": 3.5}
    clip_limit = clip_limits.get(enhance_level, 2.5)

    img = load_image(image_source)
    result = _apply_preprocessing(
        img,
        region_name="leads",
        enhance_contrast=True,
        clip_limit=clip_limit,
        sharpen=True,
        sharpen_amount=1.0,
    )
    return get_output(result, output_path)


@tool
def preprocess_focus_body(
    image_source: str,
    output_path: Optional[str] = None,
    check_alignment: bool = True
) -> str:
    """
    바디(검은색 컴포넌트) 영역에 집중한 전처리를 수행합니다.

    [용도] 바디의 위치, 기울기, 손상 여부를 검사할 때 사용합니다.
    - 바디가 중앙에 위치하는지
    - 바디가 기울어졌는지 (tilt)
    - 바디 표면에 손상이나 이물질이 있는지

    이 도구는 에이전트의 "recheck_body_alignment" 액션과 함께 사용됩니다.

    Args:
        image_source: 이미지 URL 또는 로컬 파일 경로
        output_path: 저장할 파일 경로 (None이면 base64 data URL 반환)
        check_alignment: True면 정렬 확인용 대비 강화 적용

    Returns:
        전처리된 이미지 (파일 경로 또는 data URL)
    """
    img = load_image(image_source)
    result = _apply_preprocessing(
        img,
        region_name="body",
        enhance_contrast=check_alignment,
        clip_limit=2.0,
        sharpen=False,
    )
    return get_output(result, output_path)


@tool
def preprocess_focus_lead_tips(
    image_source: str,
    output_path: Optional[str] = None,
    high_contrast: bool = True
) -> str:
    """
    리드 끝부분과 홀 연결 영역에 집중한 전처리를 수행합니다.

    [용도] 리드가 홀에 제대로 들어갔는지 검사할 때 사용합니다.
    - 리드 끝이 홀 안에 있는지 (connected vs floating)
    - 리드가 홀을 벗어났는지 (missed_hole)
    - 리드가 수직 라인에 붙었는지 (attached_to_line)

    이 도구는 에이전트의 "patch_recheck_leads" 액션과 함께 사용됩니다.

    Args:
        image_source: 이미지 URL 또는 로컬 파일 경로
        output_path: 저장할 파일 경로 (None이면 base64 data URL 반환)
        high_contrast: True면 홀과 리드 구분을 위한 강한 대비 적용

    Returns:
        전처리된 이미지 (파일 경로 또는 data URL)
    """
    img = load_image(image_source)
    result = _apply_preprocessing(
        img,
        region_name="lead_tips",
        enhance_contrast=True,
        clip_limit=3.0 if high_contrast else 2.0,
        sharpen=True,
        sharpen_amount=1.5,
    )
    return get_output(result, output_path)


@tool
def preprocess_full_enhanced(
    image_source: str,
    output_path: Optional[str] = None,
    denoise: bool = True
) -> str:
    """
    전체 이미지에 종합적인 전처리를 수행합니다.

    [용도] 전체적인 품질 향상이 필요하거나 종합 검사할 때 사용합니다.
    - 이미지 품질이 좋지 않을 때
    - 전체 컴포넌트를 한번에 분석할 때
    - 불확실한 판단을 재검토할 때

    이 도구는 에이전트의 "dual_model_check" 액션과 함께 사용됩니다.

    Args:
        image_source: 이미지 URL 또는 로컬 파일 경로
        output_path: 저장할 파일 경로 (None이면 base64 data URL 반환)
        denoise: True면 노이즈 제거 적용

    Returns:
        전처리된 이미지 (파일 경로 또는 data URL)
    """
    img = load_image(image_source)
    result = _apply_preprocessing(
        img,
        region_name="full",
        enhance_contrast=True,
        clip_limit=2.0,
        sharpen=True,
        sharpen_amount=0.8,
        denoise=denoise,
        denoise_strength=5,
    )
    return get_output(result, output_path)


@tool
def preprocess_single_lead(
    image_source: str,
    lead_position: str,
    output_path: Optional[str] = None
) -> str:
    """
    특정 리드 하나에 집중한 전처리를 수행합니다.

    [용도] 개별 리드의 상태를 자세히 검사할 때 사용합니다.
    - 특정 리드가 이상해 보일 때
    - 좌/중/우 리드 중 하나만 집중 분석할 때

    Args:
        image_source: 이미지 URL 또는 로컬 파일 경로
        lead_position: 리드 위치 ("left", "center", "right")
        output_path: 저장할 파일 경로 (None이면 base64 data URL 반환)

    Returns:
        전처리된 이미지 (파일 경로 또는 data URL)
    """
    region_map = {
        "left": "left_lead",
        "center": "center_lead",
        "right": "right_lead",
    }
    region_name = region_map.get(lead_position, "center_lead")

    img = load_image(image_source)
    result = _apply_preprocessing(
        img,
        region_name=region_name,
        enhance_contrast=True,
        clip_limit=2.5,
        sharpen=True,
        sharpen_amount=1.2,
    )
    return get_output(result, output_path)


@tool
def preprocess_custom(
    image_source: str,
    output_path: Optional[str] = None,
    top_ratio: float = 0.0,
    bottom_ratio: float = 1.0,
    left_ratio: float = 0.0,
    right_ratio: float = 1.0,
    enhance_contrast: bool = True,
    clip_limit: float = 2.0,
    sharpen: bool = False,
    sharpen_amount: float = 1.5,
    denoise: bool = False,
    denoise_strength: float = 10,
    detect_edges: bool = False,
) -> str:
    """
    사용자 지정 영역과 옵션으로 전처리를 수행합니다.

    [용도] 미리 정의된 focus 영역이 맞지 않을 때 직접 설정합니다.

    Args:
        image_source: 이미지 URL 또는 로컬 파일 경로
        output_path: 저장할 파일 경로 (None이면 base64 data URL 반환)
        top_ratio: 상단 시작점 비율 (0.0~1.0)
        bottom_ratio: 하단 끝점 비율 (0.0~1.0)
        left_ratio: 좌측 시작점 비율 (0.0~1.0)
        right_ratio: 우측 끝점 비율 (0.0~1.0)
        enhance_contrast: 대비 강화 적용 여부
        clip_limit: CLAHE 클립 제한값 (1.0~4.0)
        sharpen: 샤프닝 적용 여부
        sharpen_amount: 샤프닝 강도 (0.5~3.0)
        denoise: 노이즈 제거 적용 여부
        denoise_strength: 노이즈 제거 강도 (5~20)
        detect_edges: 에지 검출 적용 여부

    Returns:
        전처리된 이미지 (파일 경로 또는 data URL)
    """
    custom_region = {
        "top": top_ratio,
        "bottom": bottom_ratio,
        "left": left_ratio,
        "right": right_ratio,
    }

    img = load_image(image_source)
    result = _apply_preprocessing(
        img,
        custom_region=custom_region,
        enhance_contrast=enhance_contrast,
        clip_limit=clip_limit,
        sharpen=sharpen,
        sharpen_amount=sharpen_amount,
        denoise=denoise,
        denoise_strength=denoise_strength,
        detect_edges=detect_edges,
    )
    return get_output(result, output_path)


@tool
def detect_edges_for_analysis(
    image_source: str,
    focus_region: str = "full",
    output_path: Optional[str] = None,
    sensitivity: str = "normal"
) -> str:
    """
    에지 검출을 수행하여 리드 형태를 분석합니다.

    [용도] 리드의 윤곽과 방향을 명확하게 볼 때 사용합니다.
    - 리드가 교차했는지 (crossed)
    - 리드가 엉켰는지 (tangled)
    - 리드 형태가 변형되었는지

    Args:
        image_source: 이미지 URL 또는 로컬 파일 경로
        focus_region: 집중 영역 ("full", "leads", "lead_tips")
        output_path: 저장할 파일 경로 (None이면 base64 data URL 반환)
        sensitivity: 감도 ("low", "normal", "high")

    Returns:
        에지 검출된 이미지 (파일 경로 또는 data URL)
    """
    thresholds = {
        "low": (80, 200),
        "normal": (50, 150),
        "high": (30, 100),
    }
    low, high = thresholds.get(sensitivity, (50, 150))

    img = load_image(image_source)

    # 먼저 영역 크롭
    if focus_region in FOCUS_REGIONS:
        img = _crop_region(img, FOCUS_REGIONS[focus_region])

    # 대비 강화 후 에지 검출
    enhanced = _enhance_contrast(img, clip_limit=2.0)
    result = _detect_edges(enhanced, low, high)

    return get_output(result, output_path)


# =========================
# 도구 목록 (에이전트에서 사용)
# =========================

# Focus 기반 주요 도구
FOCUS_TOOLS = [
    preprocess_focus_leads,       # recheck_leads_focus 대응
    preprocess_focus_body,        # recheck_body_alignment 대응
    preprocess_focus_lead_tips,   # patch_recheck_leads 대응
    preprocess_full_enhanced,     # dual_model_check 대응
]

# 추가 유틸리티 도구
UTILITY_TOOLS = [
    preprocess_single_lead,       # 개별 리드 분석
    preprocess_custom,            # 커스텀 전처리
    detect_edges_for_analysis,    # 에지 검출 분석
]

# 전체 도구 목록
PREPROCESSING_TOOLS = FOCUS_TOOLS + UTILITY_TOOLS


# =========================
# 테스트 코드
# =========================

if __name__ == "__main__":
    import sys

    test_image = "./dev_images/DEV_000.png"
    output_dir = "./preprocessed_images"

    if not os.path.exists(test_image):
        print(f"테스트 이미지를 찾을 수 없습니다: {test_image}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Focus 기반 이미지 전처리 도구 테스트")
    print("=" * 60)

    # 1. Focus Leads (recheck_leads_focus 대응)
    print("\n1. preprocess_focus_leads (리드 영역 집중)...")
    result = preprocess_focus_leads.invoke({
        "image_source": test_image,
        "output_path": f"{output_dir}/focus_leads.png",
        "enhance_level": "normal"
    })
    print(f"   저장됨: {result}")

    # 2. Focus Body (recheck_body_alignment 대응)
    print("\n2. preprocess_focus_body (바디 영역 집중)...")
    result = preprocess_focus_body.invoke({
        "image_source": test_image,
        "output_path": f"{output_dir}/focus_body.png"
    })
    print(f"   저장됨: {result}")

    # 3. Focus Lead Tips (patch_recheck_leads 대응)
    print("\n3. preprocess_focus_lead_tips (리드 끝/홀 연결 집중)...")
    result = preprocess_focus_lead_tips.invoke({
        "image_source": test_image,
        "output_path": f"{output_dir}/focus_lead_tips.png",
        "high_contrast": True
    })
    print(f"   저장됨: {result}")

    # 4. Full Enhanced (dual_model_check 대응)
    print("\n4. preprocess_full_enhanced (전체 종합 강화)...")
    result = preprocess_full_enhanced.invoke({
        "image_source": test_image,
        "output_path": f"{output_dir}/full_enhanced.png",
        "denoise": True
    })
    print(f"   저장됨: {result}")

    # 5. Single Lead
    print("\n5. preprocess_single_lead (중앙 리드만)...")
    result = preprocess_single_lead.invoke({
        "image_source": test_image,
        "lead_position": "center",
        "output_path": f"{output_dir}/single_lead_center.png"
    })
    print(f"   저장됨: {result}")

    # 6. Edge Detection
    print("\n6. detect_edges_for_analysis (리드 에지 검출)...")
    result = detect_edges_for_analysis.invoke({
        "image_source": test_image,
        "focus_region": "leads",
        "output_path": f"{output_dir}/edges_leads.png",
        "sensitivity": "normal"
    })
    print(f"   저장됨: {result}")

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print(f"결과 이미지: {output_dir}/")
    print("=" * 60)

    print("\n[에이전트 연동 가이드]")
    print("- recheck_leads_focus    -> preprocess_focus_leads")
    print("- recheck_body_alignment -> preprocess_focus_body")
    print("- patch_recheck_leads    -> preprocess_focus_lead_tips")
    print("- dual_model_check       -> preprocess_full_enhanced")
