# V10 분석 파이프라인 상세 문서

## 개요

V10은 **YOLO 객체 검출**과 **GPT-4o 언어 모델**을 결합한 하이브리드 분석 파이프라인입니다.
제조 이미지에서 부품(body), 리드(lead), 홀(hole)을 검출하고, 조립 상태의 정상/불량을 판정합니다.

### V7 대비 변경사항

| 버전 | Stage 0 | Stage 1 | Stage 2 |
|------|---------|---------|---------|
| V7 | 없음 | GPT-4o 관찰 | GPT-4o 판단 |
| **V10** | **YOLO 검출** | GPT-4o 관찰 (YOLO 힌트 포함) | GPT-4o 판단 |

---

## 전체 파이프라인 흐름

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            V10 전체 파이프라인                                │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │   이미지 URL  │
                              └──────┬───────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │  STAGE 0: YOLO 검출            │
                    │  (Roboflow API)                │
                    └────────────────┬───────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │  정렬 분석                      │
                    │  (Lead-Hole Alignment)         │
                    └────────────────┬───────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │  STAGE 1: GPT-4o 관찰          │
                    │  (YOLO 힌트 + 이미지)          │
                    └────────────────┬───────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │  STAGE 2: GPT-4o 판단          │
                    │  (점수 기반 분류)              │
                    └────────────────┬───────────────┘
                                     │
                          ┌──────────┴──────────┐
                          │                     │
                    confidence < 0.85?    confidence >= 0.85?
                          │                     │
                          ▼                     │
                    ┌────────────────┐          │
                    │  조건부 RECHECK │          │
                    │  (최대 4회)     │          │
                    └───────┬────────┘          │
                            │                   │
                            ▼                   │
                    ┌────────────────┐          │
                    │  가중치 투표    │◄─────────┘
                    └───────┬────────┘
                            │
                            ▼
                    ┌────────────────┐
                    │  최종 판정      │
                    │  0(정상)/1(불량)│
                    └────────────────┘
```

---

## Stage 0: YOLO 검출

### 목적
이미지에서 **hole**, **lead**, **body**의 위치를 빠르게 검출합니다.

### 사용 기술
- **Roboflow Serverless API**
- **Workflow ID**: `find-dark-circular-holes-silver-metal-leads-and-black-rectangular-bodies`

### 입력
```
이미지 URL (예: https://example.com/image.png)
```

### 출력
```python
{
    "holes": [
        {"x": 중심x, "y": 중심y, "w": 너비, "h": 높이, "confidence": 0.95},
        # ... 약 15-20개
    ],
    "leads": [
        {"x": 중심x, "y": 중심y, "w": 너비, "h": 높이, "confidence": 0.92},
        # ... 3개
    ],
    "body": {
        "x": 중심x, "y": 중심y, "w": 너비, "h": 높이, "confidence": 0.98
    },
    "detection_success": True
}
```

### 클래스 매핑

| Roboflow 클래스 | 내부 클래스 ID | 설명 |
|-----------------|---------------|------|
| dark-circular-hole | 0 | 홀 (구멍) |
| silver-metal-lead | 1 | 리드 (금속 다리) |
| black-rectangular-body | 2 | 부품 몸체 |

---

## 정렬 분석 (Lead-Hole Alignment)

### 목적
각 리드가 대응하는 홀에 정확히 삽입되었는지 **수치적으로** 분석합니다.

### 알고리즘

```python
def analyze_lead_hole_alignment(yolo_result):
    # 1. 하단 3개 홀 선택 (y좌표가 가장 큰 것)
    bottom_holes = sorted(holes, key=y, reverse=True)[:3]

    # 2. 홀을 x좌표로 정렬 (좌→우)
    bottom_holes.sort(key=x)  # [left_hole, center_hole, right_hole]

    # 3. 리드를 x좌표로 정렬
    sorted_leads.sort(key=x)  # [left_lead, center_lead, right_lead]

    # 4. 각 리드-홀 쌍의 거리 계산
    for i, (lead, hole) in enumerate(zip(sorted_leads, bottom_holes)):
        lead_tip_y = lead.y + lead.h / 2  # 리드 하단
        distance = sqrt((lead.x - hole.x)² + (lead_tip_y - hole.y)²)

        # 5. 연결 여부 판정
        if distance < hole_radius * 1.5:
            reaches_hole = "yes"
        else:
            reaches_hole = "no"  # missed_hole!
```

### 출력
```python
{
    "left_lead":   {"reaches_hole": "yes", "distance": 12.5},
    "center_lead": {"reaches_hole": "yes", "distance": 8.3},
    "right_lead":  {"reaches_hole": "no",  "distance": 45.7},  # 문제!
    "alignment_issues": ["right_lead_missed_hole"]
}
```

---

## Stage 1: GPT-4o 관찰

### 목적
YOLO 검출 결과를 **힌트**로 제공하고, GPT-4o가 이미지를 직접 보면서 상세 관찰합니다.

### YOLO 힌트 주입

프롬프트에 다음 정보가 포함됩니다:
```
YOLO Detection Results (for reference):
- Detected 19 holes
- Detected 3 leads
- Body detected: Yes

Potential Issues Detected by YOLO:
  - right_lead_missed_hole

Lead-Hole Alignment Analysis:
  - left_lead: reaches_hole=yes
  - center_lead: reaches_hole=yes
  - right_lead: reaches_hole=no    ← 주의 필요!
```

### 관찰 항목

#### Body (몸체)
| 필드 | 허용 값 | 설명 |
|------|--------|------|
| tilt | none, mild, severe, unknown | 기울어짐 |
| rotation | none, mild, severe, unknown | 회전 |
| center_offset | centered, left_shift, right_shift, unknown | 중앙 정렬 |
| surface_mark | none, present, unknown | 표면 이물질 |

#### Leads (리드)
| 필드 | 허용 값 | 설명 |
|------|--------|------|
| shape | straightish, curved, severely_deformed, blob_like | 형태 |
| end_position | **in_hole**, **missed_hole**, attached_to_line, unknown | 끝점 위치 |
| contact_with_board | connected, **floating**, unknown | 기판 접촉 |
| touches_vertical_line | yes, no, unknown | 수직선 접촉 |

#### 전체 리드 상태
| 필드 | 허용 값 | 설명 |
|------|--------|------|
| lead_arrangement | parallel, **crossed**, **tangled**, splayed | 리드 배치 |
| all_leads_reach_holes | yes, **no**, unknown | 전체 연결 여부 |

### 출력 JSON 구조

```json
{
  "body": {
    "tilt": {"value": "none", "evidence": "Body appears level"},
    "surface_mark_unusual_blob": {"value": "none", "evidence": "Clean surface"}
  },
  "leads": {
    "left_lead": {
      "shape": "straightish",
      "end_position": "in_hole",
      "contact_with_board": "connected",
      "evidence": "Lead goes straight into left hole"
    },
    "center_lead": { ... },
    "right_lead": {
      "shape": "curved",
      "end_position": "missed_hole",      // YOLO 힌트 확인됨!
      "contact_with_board": "floating",
      "evidence": "Lead bends right, misses the hole"
    },
    "lead_arrangement": {"value": "parallel", "evidence": "..."},
    "all_leads_reach_holes": {"value": "no", "evidence": "Right lead missed"}
  },
  "yolo_verification": {
    "yolo_alignment_issues_confirmed": ["right_lead_missed_hole"]
  }
}
```

---

## Stage 2: GPT-4o 판단

### 목적
Stage 1의 관찰 결과를 기반으로 **점수를 계산**하고 정상/불량을 판정합니다.

### 점수 체계

#### CRITICAL (+5점) - 즉시 불량
| 조건 | 점수 | 설명 |
|------|------|------|
| end_position == "missed_hole" | +5 | 리드가 홀을 벗어남 |
| end_position == "attached_to_line" | +5 | 리드가 수직선에 붙음 |
| touches_vertical_line == "yes" | +5 | 수직 트레이스 접촉 |
| contact_with_board == "floating" | +5 | 리드가 떠있음 |
| lead_arrangement == "crossed" | +5 | 리드 교차 |
| lead_arrangement == "tangled" | +5 | 리드 엉킴 |
| all_leads_reach_holes == "no" | +5 | 전체 연결 실패 |

#### HIGH (+3~4점)
| 조건 | 점수 |
|------|------|
| body.tilt == "severe" | +3 |
| center_lead.shape == "blob_like" | +4 |
| lead_overlap_or_clumping == "present" | +3 |
| shape == "severely_deformed" | +4 |

#### MEDIUM (+2점)
| 조건 | 점수 |
|------|------|
| symmetry == "asymmetric" | +2 |

#### LOW (+1점, 주의)
| 조건 | 점수 | 비고 |
|------|------|------|
| surface_mark == "present" | +1 | 오탐 많음 |

### 판정 기준

```
총점 >= 3  →  abnormal (불량)
총점 < 3   →  normal (정상)

예외: CRITICAL 조건 1개라도 해당 → 즉시 abnormal
```

### 출력
```json
{
  "label": "abnormal",
  "confidence": 0.92,
  "key_reasons": [
    "right_lead end_position == missed_hole (+5)",
    "all_leads_reach_holes == no (+5)"
  ],
  "triggered_checks": "none",
  "termination_reason": "high_confidence"
}
```

---

## 조건부 Recheck

### Recheck 트리거 조건

| 조건 | 설명 |
|------|------|
| `triggered_checks != "none"` | Stage 2가 명시적으로 요청 |
| `confidence < 0.85` | 낮은 확신도 |
| Suspicious reason 존재 | blob_like, asymmetric 등 |
| 판정 충돌 | 원본 vs recheck 결과가 다름 |

### Recheck 시퀀스

최대 4회까지, 다음 순서로 시도:

| 순서 | Focus Action | 전처리 내용 |
|------|--------------|------------|
| 1 | `recheck_leads_focus` | 리드 영역 강조, 대비 증가 |
| 2 | `patch_recheck_leads` | 리드 끝점 영역 확대 |
| 3 | `recheck_body_alignment` | 몸체 영역 강조 |
| 4 | `dual_model_check` | 전체 이미지 개선 |

### Recheck 흐름

```
원본 판정 (abnormal, conf=0.75)
        │
        ▼ (confidence < 0.85 → recheck 필요)

Recheck #1: recheck_leads_focus
        │ 결과: normal, conf=0.82
        │
        ▼ (원본과 다름 → 추가 recheck)

Recheck #2: patch_recheck_leads
        │ 결과: abnormal, conf=0.88
        │
        ▼

가중치 투표로 최종 결정
```

---

## 가중치 투표

### 가중치 계산 공식

```python
weight = base_weight
       × original_bonus      # 원본이면 ×1.5
       × critical_bonus      # critical reason 있으면 ×2.0
       × suspicious_bonus    # suspicious reason 있으면 ×1.2
       × confidence          # 확신도 반영
```

### 가중치 상수

| 요소 | 가중치 |
|------|--------|
| 기본 (recheck) | 1.0 |
| 원본 판정 | 1.5 |
| Critical reason | 2.0 |
| Suspicious reason | 1.2 |

### 투표 예시

```
결과 1 (원본):   abnormal, conf=0.90, critical reason
  → weight = 1.5 × 2.0 × 0.90 = 2.70

결과 2 (recheck): normal, conf=0.82
  → weight = 1.0 × 0.82 = 0.82

결과 3 (recheck): abnormal, conf=0.75
  → weight = 1.0 × 0.75 = 0.75

abnormal_weight = 2.70 + 0.75 = 3.45
normal_weight   = 0.82

최종 판정: abnormal (3.45 > 0.82)
최종 confidence: 3.45 / (3.45 + 0.82) = 0.81
```

### 동점 처리

```
|abnormal_weight - normal_weight| < 0.1 → 동점
→ 추가 recheck 수행
→ 모든 focus 사용 후에도 동점 → abnormal (보수적 판단)
```

---

## API 호출 구조

### Roboflow API (Stage 0)

```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="YOUR_API_KEY"
)

result = client.run_workflow(
    workspace_name="ngv-ra5w3",
    workflow_id="find-dark-circular-holes-silver-metal-leads-and-black-rectangular-bodies",
    images={"image": img_url},
    use_cache=True
)
```

### GPT-4o API (Stage 1, 2)

```python
payload = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": USER_PROMPT},
            {"type": "image_url", "image_url": {"url": img_url}}
        ]}
    ],
    "temperature": 0.1
}
```

---

## 성능 특성

| 항목 | 값 |
|------|-----|
| Stage 0 (YOLO) | ~0.5초 |
| Stage 1 (GPT-4o) | ~3-5초 |
| Stage 2 (GPT-4o) | ~2-3초 |
| Recheck 1회 | ~5-8초 |
| 전체 (recheck 없음) | ~6-10초 |
| 전체 (recheck 2회) | ~15-20초 |

### 병렬 처리

```python
MAX_WORKERS = 20  # 동시 처리 이미지 수
```

---

## 파일 구조

```
src/agent/
├── agent_v7.py          # V7: GPT-4o only
├── agent_v10.py         # V10: YOLO + GPT-4o (현재 버전)
│
src/labeling/
├── roboflow_label.py    # Roboflow workflow로 라벨링
├── prepare_roboflow_upload.py  # 업로드 데이터 준비
│
src/preprocessing/
├── image_preprocessing_tools.py  # 전처리 도구
│
data/
├── yolo_dataset/        # 라벨링된 데이터
│   ├── images/
│   ├── labels/
│   └── visualizations/
└── roboflow_upload/     # Roboflow 업로드용
```

---

## 향후 개선 방향

1. **파인튜닝된 YOLO 모델 적용**
   - 현재: Roboflow Workflow 사용
   - 개선: 도메인 특화 모델 학습 후 `ROBOFLOW_MODEL_ID` 설정

2. **Stage 1 경량화**
   - YOLO 검출 결과가 충분히 신뢰할 수 있으면 GPT-4o 호출 생략
   - 비용 및 속도 개선

3. **Confidence 캘리브레이션**
   - 실제 정답 데이터와 비교하여 confidence threshold 조정
