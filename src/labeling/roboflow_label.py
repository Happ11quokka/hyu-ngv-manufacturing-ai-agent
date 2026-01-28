"""
Roboflow Workflow를 사용한 dev 이미지 라벨링
"""

import os
from pathlib import Path
from inference_sdk import InferenceHTTPClient
import cv2

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEV_IMAGES_DIR = DATA_DIR / "dev_images"
OUTPUT_DIR = DATA_DIR / "yolo_dataset"

# Roboflow 설정
API_URL = "https://serverless.roboflow.com"
API_KEY = "4jFqretpF99k2W3RnV1S"
WORKSPACE_NAME = "ngv-ra5w3"
WORKFLOW_ID = "find-dark-circular-holes-silver-metal-leads-and-black-rectangular-bodies"

# 클래스 매핑 (워크플로우 결과 -> YOLO 클래스 ID)
CLASS_MAPPING = {
    "hole": 0,
    "dark-circular-hole": 0,
    "lead": 1,
    "lead_tip": 1,
    "silver-metal-lead": 1,
    "body": 2,
    "black-rectangular-body": 2,
}

# 클래스 정의
CLASSES = {
    0: "hole",
    1: "lead_tip",
    2: "body",
}


def to_yolo_format(bbox: dict, img_w: int, img_h: int, class_id: int) -> str:
    """바운딩 박스를 YOLO 형식으로 변환"""
    # bbox는 x, y, width, height 또는 x1, y1, x2, y2 형식일 수 있음
    if "x" in bbox and "y" in bbox and "width" in bbox and "height" in bbox:
        cx = (bbox["x"] + bbox["width"] / 2) / img_w
        cy = (bbox["y"] + bbox["height"] / 2) / img_h
        w = bbox["width"] / img_w
        h = bbox["height"] / img_h
    elif "x1" in bbox:
        cx = ((bbox["x1"] + bbox["x2"]) / 2) / img_w
        cy = ((bbox["y1"] + bbox["y2"]) / 2) / img_h
        w = (bbox["x2"] - bbox["x1"]) / img_w
        h = (bbox["y2"] - bbox["y1"]) / img_h
    else:
        # center x, y, width, height 형식
        cx = bbox.get("center_x", bbox.get("x", 0)) / img_w
        cy = bbox.get("center_y", bbox.get("y", 0)) / img_h
        w = bbox.get("w", bbox.get("width", 0)) / img_w
        h = bbox.get("h", bbox.get("height", 0)) / img_h

    # 범위 클리핑
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    w = max(0.01, min(1, w))
    h = max(0.01, min(1, h))

    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def visualize_labels(img_path: Path, labels: list, output_path: Path):
    """라벨 시각화"""
    img = cv2.imread(str(img_path))
    if img is None:
        return

    h, w = img.shape[:2]
    colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0)}
    names = {0: "hole", 1: "lead", 2: "body"}

    for label in labels:
        parts = label.split()
        class_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])

        # YOLO 형식 -> 픽셀 좌표
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)

        color = colors.get(class_id, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, names.get(class_id, str(class_id)), (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(str(output_path), img)


def process_workflow_result(result: dict, img_w: int, img_h: int) -> list:
    """워크플로우 결과를 YOLO 라벨로 변환"""
    labels = []

    print(f"  Result keys: {result.keys()}")

    # 결과 구조 파싱 (다양한 형식 지원)
    predictions = []

    # 직접 predictions 키가 있는 경우
    if "predictions" in result:
        preds = result["predictions"]
        if isinstance(preds, list):
            predictions.extend(preds)
        elif isinstance(preds, dict) and "predictions" in preds:
            predictions.extend(preds["predictions"])

    # output 키가 있는 경우
    if "output" in result:
        output = result["output"]
        if isinstance(output, dict):
            if "predictions" in output:
                predictions.extend(output["predictions"])
            # 여러 출력이 있는 경우
            for key, value in output.items():
                if isinstance(value, dict) and "predictions" in value:
                    predictions.extend(value["predictions"])
                elif isinstance(value, list):
                    predictions.extend(value)

    # 직접 리스트인 경우
    if isinstance(result, list):
        predictions.extend(result)

    # 다른 키 탐색
    for key in result:
        if key not in ["predictions", "output"] and isinstance(result[key], dict):
            if "predictions" in result[key]:
                predictions.extend(result[key]["predictions"])

    print(f"  Found {len(predictions)} predictions")

    for pred in predictions:
        if not isinstance(pred, dict):
            continue

        # 클래스 이름 추출
        class_name = pred.get("class", pred.get("label", pred.get("class_name", ""))).lower()

        # 클래스 ID 매핑
        class_id = None
        for key, cid in CLASS_MAPPING.items():
            if key in class_name:
                class_id = cid
                break

        if class_id is None:
            print(f"    Unknown class: {class_name}")
            continue

        # 바운딩 박스 추출
        bbox = {}
        if "x" in pred and "y" in pred and "width" in pred and "height" in pred:
            bbox = {
                "x": pred["x"] - pred["width"]/2,  # center to corner
                "y": pred["y"] - pred["height"]/2,
                "width": pred["width"],
                "height": pred["height"]
            }
        elif "x1" in pred:
            bbox = {"x1": pred["x1"], "y1": pred["y1"], "x2": pred["x2"], "y2": pred["y2"]}
        elif "bbox" in pred:
            bbox = pred["bbox"]

        if bbox:
            label = to_yolo_format(bbox, img_w, img_h, class_id)
            labels.append(label)
            print(f"    {class_name} -> class {class_id}: {label}")

    return labels


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

    # Roboflow 클라이언트 초기화
    client = InferenceHTTPClient(
        api_url=API_URL,
        api_key=API_KEY
    )

    print("=" * 50)
    print("Roboflow Workflow 라벨링")
    print("=" * 50)
    print(f"Workspace: {WORKSPACE_NAME}")
    print(f"Workflow: {WORKFLOW_ID}")
    print()

    # 이미지 처리
    import shutil

    for img_path in sorted(DEV_IMAGES_DIR.glob("*.png")):
        print(f"Processing: {img_path.name}")

        # 이미지 크기 확인
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Failed to load image")
            continue

        img_h, img_w = img.shape[:2]

        try:
            # Roboflow 워크플로우 실행
            result = client.run_workflow(
                workspace_name=WORKSPACE_NAME,
                workflow_id=WORKFLOW_ID,
                images={"image": str(img_path)},
                use_cache=True
            )

            print(f"  API response type: {type(result)}")

            # 결과가 리스트인 경우 첫 번째 요소 사용
            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            # 라벨 생성
            labels = process_workflow_result(result, img_w, img_h)

            # 라벨 파일 저장
            label_path = labels_dir / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(labels))

            # 이미지 복사
            shutil.copy(img_path, images_dir / img_path.name)

            # 시각화 저장
            if labels:
                visualize_labels(img_path, labels, vis_dir / f"{img_path.stem}_labeled.png")

            print(f"  Saved {len(labels)} labels")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 50)
    print("완료!")
    print(f"- 이미지: {images_dir}")
    print(f"- 라벨: {labels_dir}")
    print(f"- 시각화: {vis_dir}")


if __name__ == "__main__":
    main()
