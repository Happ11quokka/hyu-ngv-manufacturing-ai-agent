"""
반자동 YOLO 라벨링 스크립트

1. OpenCV로 홀/리드 검출
2. YOLO 형식 라벨 생성
3. 시각화 이미지 저장 (수동 검수용)
"""

import os
import cv2
import numpy as np
from pathlib import Path

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEV_IMAGES_DIR = DATA_DIR / "dev_images"
OUTPUT_DIR = PROJECT_ROOT / "data" / "yolo_dataset"

# 클래스 정의
CLASSES = {
    0: "hole",      # 홀 (3개)
    1: "lead_tip",  # 리드 끝점 (3개)
    2: "body",      # 부품 몸체
}


def detect_holes_improved(img: np.ndarray) -> list:
    """개선된 홀 검출 - 이미지 하단 영역에서 원형 검출"""
    h, w = img.shape[:2]

    # 홀 검색 영역 (하단 60-95%)
    top = int(h * 0.60)
    bottom = int(h * 0.95)
    left = int(w * 0.10)
    right = int(w * 0.90)
    roi = img[top:bottom, left:right]

    # 그레이스케일 + 블러
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 어두운 영역 (홀) 찾기
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

    # 모폴로지 연산
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 컨투어 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    holes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 2000:  # 홀 크기 범위
            # 원형도 체크
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.5:  # 원형에 가까운 것만
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    cx = x + bw // 2 + left
                    cy = y + bh // 2 + top
                    radius = max(bw, bh) // 2
                    holes.append({
                        "x": cx, "y": cy,
                        "w": bw + 4, "h": bh + 4,  # 여유 마진
                        "radius": radius
                    })

    # x 좌표로 정렬
    holes.sort(key=lambda h: h["x"])

    # 상위 3개만 (가장 큰 것 또는 가장 원형인 것)
    return holes[:3]


def detect_lead_tips_improved(img: np.ndarray) -> list:
    """개선된 리드 끝점 검출 - 금속 색상 기반"""
    h, w = img.shape[:2]

    # 리드 검색 영역 (중간-하단)
    top = int(h * 0.40)
    bottom = int(h * 0.85)
    left = int(w * 0.15)
    right = int(w * 0.85)
    roi = img[top:bottom, left:right]

    # HSV 변환
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 금속(은색/회색) 마스크 - 낮은 채도, 높은 명도
    lower = np.array([0, 0, 140])
    upper = np.array([180, 80, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # 모폴로지 연산
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 컨투어 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lead_tips = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # 세로로 긴 형태 (리드 특성) 또는 일정 크기 이상
        if bh > bw * 0.3 or area > 100:
            # 컨투어에서 가장 아래쪽 점 (리드 끝점)
            bottom_point = tuple(cnt[cnt[:, :, 1].argmax()][0])

            lead_tips.append({
                "x": x + left,
                "y": y + top,
                "w": bw + 4,
                "h": bh + 4,
                "tip_x": bottom_point[0] + left,
                "tip_y": bottom_point[1] + top,
                "area": area
            })

    # x 좌표로 정렬
    lead_tips.sort(key=lambda t: t["x"])

    # 3개 영역으로 그룹화
    if len(lead_tips) >= 3:
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

        # 각 그룹에서 가장 큰 영역 선택
        result = []
        for key in ["left", "center", "right"]:
            if grouped[key]:
                best = max(grouped[key], key=lambda t: t["area"])
                result.append(best)

        return result

    return lead_tips[:3]


def detect_body(img: np.ndarray) -> dict:
    """부품 몸체 검출 - 상단의 검은 사각형"""
    h, w = img.shape[:2]

    # 상단 영역
    top = int(h * 0.05)
    bottom = int(h * 0.45)
    left = int(w * 0.20)
    right = int(w * 0.80)
    roi = img[top:bottom, left:right]

    # 그레이스케일
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 어두운 영역 (몸체)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # 컨투어 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 가장 큰 컨투어
        largest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)

        return {
            "x": x + left,
            "y": y + top,
            "w": bw,
            "h": bh
        }

    # 기본값 (상단 중앙)
    return {
        "x": int(w * 0.25),
        "y": int(h * 0.10),
        "w": int(w * 0.50),
        "h": int(h * 0.30)
    }


def to_yolo_format(bbox: dict, img_w: int, img_h: int, class_id: int) -> str:
    """바운딩 박스를 YOLO 형식으로 변환"""
    cx = (bbox["x"] + bbox["w"] / 2) / img_w
    cy = (bbox["y"] + bbox["h"] / 2) / img_h
    w = bbox["w"] / img_w
    h = bbox["h"] / img_h

    # 범위 클리핑
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    w = max(0.01, min(1, w))
    h = max(0.01, min(1, h))

    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def visualize_labels(img: np.ndarray, holes: list, leads: list, body: dict) -> np.ndarray:
    """라벨 시각화"""
    vis = img.copy()

    # 홀 (빨간색)
    for i, hole in enumerate(holes):
        x, y, w, h = hole["x"], hole["y"], hole["w"], hole["h"]
        cv2.rectangle(vis, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 0, 255), 2)
        cv2.putText(vis, f"H{i}", (x - 5, y - h//2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # 리드 (초록색)
    for i, lead in enumerate(leads):
        x, y, w, h = lead["x"], lead["y"], lead["w"], lead["h"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, f"L{i}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 몸체 (파란색)
    if body:
        x, y, w, h = body["x"], body["y"], body["w"], body["h"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(vis, "Body", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    return vis


def process_image(img_path: Path, output_labels_dir: Path, output_vis_dir: Path):
    """이미지 처리 및 라벨 생성"""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to load: {img_path}")
        return

    h, w = img.shape[:2]
    img_id = img_path.stem

    # 검출
    holes = detect_holes_improved(img)
    leads = detect_lead_tips_improved(img)
    body = detect_body(img)

    # YOLO 라벨 생성
    labels = []

    # 홀 (class 0)
    for hole in holes:
        # 중심 기준 바운딩 박스로 변환
        bbox = {
            "x": hole["x"] - hole["w"]//2,
            "y": hole["y"] - hole["h"]//2,
            "w": hole["w"],
            "h": hole["h"]
        }
        labels.append(to_yolo_format(bbox, w, h, 0))

    # 리드 (class 1)
    for lead in leads:
        bbox = {"x": lead["x"], "y": lead["y"], "w": lead["w"], "h": lead["h"]}
        labels.append(to_yolo_format(bbox, w, h, 1))

    # 몸체 (class 2)
    if body:
        labels.append(to_yolo_format(body, w, h, 2))

    # 라벨 파일 저장
    label_path = output_labels_dir / f"{img_id}.txt"
    with open(label_path, "w") as f:
        f.write("\n".join(labels))

    # 시각화 저장
    vis = visualize_labels(img, holes, leads, body)
    vis_path = output_vis_dir / f"{img_id}_labeled.png"
    cv2.imwrite(str(vis_path), vis)

    print(f"[{img_id}] 홀: {len(holes)}, 리드: {len(leads)}, 몸체: {'O' if body else 'X'}")


def main():
    # 출력 디렉토리 생성
    images_dir = OUTPUT_DIR / "images"
    labels_dir = OUTPUT_DIR / "labels"
    vis_dir = OUTPUT_DIR / "visualizations"

    for d in [images_dir, labels_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # classes.txt 생성
    classes_path = OUTPUT_DIR / "classes.txt"
    with open(classes_path, "w") as f:
        for class_id, class_name in CLASSES.items():
            f.write(f"{class_name}\n")

    # 이미지 처리
    print("=" * 50)
    print("YOLO 라벨 자동 생성")
    print("=" * 50)

    for img_path in sorted(DEV_IMAGES_DIR.glob("*.png")):
        # 원본 이미지 복사
        import shutil
        shutil.copy(img_path, images_dir / img_path.name)

        # 라벨 생성
        process_image(img_path, labels_dir, vis_dir)

    print()
    print(f"완료!")
    print(f"- 이미지: {images_dir}")
    print(f"- 라벨: {labels_dir}")
    print(f"- 시각화: {vis_dir}")
    print()
    print("다음 단계:")
    print("1. 시각화 이미지를 확인하고 잘못된 라벨 수정")
    print("2. YOLO 학습 진행")


if __name__ == "__main__":
    main()
