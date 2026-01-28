"""
간단한 라벨 수정 도구

키보드 단축키:
- 1,2,3: 왼쪽/중앙/오른쪽 홀 추가 (마우스 클릭 위치)
- q,w,e: 왼쪽/중앙/오른쪽 리드 추가 (마우스 클릭 위치)
- d: 가장 가까운 바운딩 박스 삭제
- s: 저장
- n: 다음 이미지
- p: 이전 이미지
- r: 현재 이미지 리셋 (자동 검출 다시)
- ESC: 종료
"""

import cv2
import numpy as np
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "yolo_dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

# 클래스 색상
COLORS = {
    0: (0, 0, 255),    # hole - 빨강
    1: (0, 255, 0),    # lead - 초록
    2: (255, 0, 0),    # body - 파랑
}

CLASS_NAMES = {0: "hole", 1: "lead", 2: "body"}

# 기본 바운딩 박스 크기 (픽셀)
DEFAULT_SIZES = {
    0: (20, 20),  # hole
    1: (15, 30),  # lead
    2: (90, 60),  # body
}


class LabelEditor:
    def __init__(self):
        self.images = sorted(IMAGES_DIR.glob("*.png"))
        self.current_idx = 0
        self.labels = []  # [(class_id, cx, cy, w, h), ...]
        self.mouse_pos = (0, 0)
        self.img = None
        self.img_h = 224
        self.img_w = 224

    def load_image(self):
        """현재 이미지와 라벨 로드"""
        img_path = self.images[self.current_idx]
        self.img = cv2.imread(str(img_path))
        self.img_h, self.img_w = self.img.shape[:2]

        # 라벨 로드
        label_path = LABELS_DIR / f"{img_path.stem}.txt"
        self.labels = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        self.labels.append((class_id, cx, cy, w, h))

    def save_labels(self):
        """라벨 저장"""
        img_path = self.images[self.current_idx]
        label_path = LABELS_DIR / f"{img_path.stem}.txt"

        with open(label_path, "w") as f:
            for label in self.labels:
                class_id, cx, cy, w, h = label
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        print(f"Saved: {label_path.name}")

    def draw(self):
        """이미지에 라벨 그리기"""
        vis = self.img.copy()

        # 라벨 그리기
        for i, (class_id, cx, cy, w, h) in enumerate(self.labels):
            x1 = int((cx - w/2) * self.img_w)
            y1 = int((cy - h/2) * self.img_h)
            x2 = int((cx + w/2) * self.img_w)
            y2 = int((cy + h/2) * self.img_h)

            color = COLORS.get(class_id, (255, 255, 255))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{CLASS_NAMES.get(class_id, '?')}{i}",
                       (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 현재 마우스 위치 표시
        cv2.circle(vis, self.mouse_pos, 3, (255, 255, 0), -1)

        # 정보 표시
        img_name = self.images[self.current_idx].name
        info = f"{self.current_idx+1}/{len(self.images)} - {img_name}"
        cv2.putText(vis, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 라벨 개수
        counts = {}
        for c, _, _, _, _ in self.labels:
            counts[c] = counts.get(c, 0) + 1
        count_str = f"H:{counts.get(0,0)} L:{counts.get(1,0)} B:{counts.get(2,0)}"
        cv2.putText(vis, count_str, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return vis

    def add_label(self, class_id):
        """마우스 위치에 라벨 추가"""
        cx = self.mouse_pos[0] / self.img_w
        cy = self.mouse_pos[1] / self.img_h
        default_w, default_h = DEFAULT_SIZES.get(class_id, (20, 20))
        w = default_w / self.img_w
        h = default_h / self.img_h

        self.labels.append((class_id, cx, cy, w, h))
        print(f"Added {CLASS_NAMES.get(class_id)} at ({self.mouse_pos[0]}, {self.mouse_pos[1]})")

    def delete_nearest(self):
        """가장 가까운 라벨 삭제"""
        if not self.labels:
            return

        mx, my = self.mouse_pos
        min_dist = float("inf")
        min_idx = -1

        for i, (_, cx, cy, _, _) in enumerate(self.labels):
            px = cx * self.img_w
            py = cy * self.img_h
            dist = np.sqrt((mx - px)**2 + (my - py)**2)
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        if min_idx >= 0 and min_dist < 50:  # 50픽셀 이내만
            removed = self.labels.pop(min_idx)
            print(f"Deleted {CLASS_NAMES.get(removed[0])} at idx {min_idx}")

    def mouse_callback(self, event, x, y, flags, param):
        """마우스 콜백"""
        self.mouse_pos = (x, y)

    def run(self):
        """에디터 실행"""
        cv2.namedWindow("Label Editor")
        cv2.setMouseCallback("Label Editor", self.mouse_callback)

        self.load_image()

        print("\n=== Label Editor ===")
        print("1,2,3: Add hole (left/center/right)")
        print("q,w,e: Add lead (left/center/right)")
        print("b: Add body")
        print("d: Delete nearest")
        print("s: Save")
        print("n/p: Next/Prev image")
        print("r: Reset (re-detect)")
        print("ESC: Exit")
        print()

        while True:
            vis = self.draw()
            cv2.imshow("Label Editor", vis)

            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                break
            elif key == ord("n"):  # Next
                self.current_idx = (self.current_idx + 1) % len(self.images)
                self.load_image()
            elif key == ord("p"):  # Prev
                self.current_idx = (self.current_idx - 1) % len(self.images)
                self.load_image()
            elif key == ord("s"):  # Save
                self.save_labels()
            elif key == ord("d"):  # Delete
                self.delete_nearest()
            elif key in [ord("1"), ord("2"), ord("3")]:  # Hole
                self.add_label(0)
            elif key in [ord("q"), ord("w"), ord("e")]:  # Lead
                self.add_label(1)
            elif key == ord("b"):  # Body
                self.add_label(2)
            elif key == ord("r"):  # Reset
                self.load_image()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    editor = LabelEditor()
    editor.run()
