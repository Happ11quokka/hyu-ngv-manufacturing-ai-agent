"""
Roboflow 업로드를 위한 데이터 준비
YOLO 형식 데이터를 Roboflow 업로드용 구조로 정리
"""

import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
YOLO_DATASET_DIR = PROJECT_ROOT / "data" / "yolo_dataset"
UPLOAD_DIR = PROJECT_ROOT / "data" / "roboflow_upload"


def prepare_upload_structure():
    """Roboflow 업로드용 디렉토리 구조 생성"""

    # 출력 디렉토리 생성
    upload_images = UPLOAD_DIR / "images"
    upload_labels = UPLOAD_DIR / "labels"

    for d in [upload_images, upload_labels]:
        d.mkdir(parents=True, exist_ok=True)

    # 이미지와 라벨 복사
    src_images = YOLO_DATASET_DIR / "images"
    src_labels = YOLO_DATASET_DIR / "labels"

    count = 0
    for img_path in sorted(src_images.glob("*.png")):
        img_name = img_path.stem
        label_path = src_labels / f"{img_name}.txt"

        if label_path.exists():
            # 이미지 복사
            shutil.copy(img_path, upload_images / img_path.name)
            # 라벨 복사
            shutil.copy(label_path, upload_labels / label_path.name)
            count += 1
            print(f"  복사: {img_name}")

    # classes.txt 복사
    classes_src = YOLO_DATASET_DIR / "classes.txt"
    if classes_src.exists():
        shutil.copy(classes_src, UPLOAD_DIR / "classes.txt")

    # data.yaml 생성 (YOLO 형식)
    yaml_content = f"""# YOLO Dataset Configuration
# For Roboflow Upload

train: {upload_images}
val: {upload_images}

nc: 3
names: ['hole', 'lead_tip', 'body']
"""

    with open(UPLOAD_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\n완료! {count}개 이미지-라벨 쌍 준비됨")
    print(f"업로드 경로: {UPLOAD_DIR}")
    print()
    print("=" * 50)
    print("Roboflow 업로드 방법:")
    print("=" * 50)
    print("1. https://app.roboflow.com 접속")
    print("2. 'Create New Project' 클릭")
    print("3. Project Type: 'Object Detection' 선택")
    print("4. 'Upload' 탭에서 images 폴더의 이미지들 드래그")
    print("5. 'Upload Annotations' 클릭 후 labels 폴더 선택")
    print("6. Annotation Format: 'YOLO v5 PyTorch' 선택")
    print()
    print("또는 CLI 사용:")
    print("  pip install roboflow")
    print("  roboflow upload <경로>")

    return count


def create_train_val_split(train_ratio=0.8):
    """Train/Val 분할 (선택적)"""
    import random

    images = list((UPLOAD_DIR / "images").glob("*.png"))
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    print(f"\nTrain/Val 분할:")
    print(f"  Train: {len(train_images)}개")
    print(f"  Val: {len(val_images)}개")

    # 분할 정보 저장
    with open(UPLOAD_DIR / "train.txt", "w") as f:
        for img in train_images:
            f.write(f"{img.name}\n")

    with open(UPLOAD_DIR / "val.txt", "w") as f:
        for img in val_images:
            f.write(f"{img.name}\n")

    return train_images, val_images


if __name__ == "__main__":
    print("=" * 50)
    print("Roboflow 업로드 데이터 준비")
    print("=" * 50)

    count = prepare_upload_structure()

    if count > 5:
        create_train_val_split()
