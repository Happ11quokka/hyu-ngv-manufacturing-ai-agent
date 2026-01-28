# dAIso Agent

**Defect Analysis & Inspection System with OpenAI**

반도체 부품 불량 검출을 위한 Two-Stage Prompting 기반 AI Agent

---

## 📄 프로젝트 개요

DAISO Agent는 반도체 부품(TO-220 패키지) 이미지를 분석하여 불량 여부를 자동으로 판정하는 AI 에이전트입니다. Two-Stage Prompting과 가중치 투표 시스템을 통해 높은 정확도의 불량 탐지를 수행합니다.

![Pipeline Architecture](assets/pipeline_architecture.png)

---

## 🏗️ 시스템 아키텍처

### Two-Stage Prompting

| Stage       | 역할        | 입력       | 출력               |
| ----------- | ----------- | ---------- | ------------------ |
| **Stage 1** | 이미지 관찰 | 이미지 URL | 관찰 JSON          |
| **Stage 2** | 불량 판정   | 관찰 JSON  | 라벨, 신뢰도, 근거 |

### 조건부 툴 호출 (V6.1)

다음 조건에서 이미지 전처리 도구를 자동 호출하여 재검증:

```
CONFIDENCE < 0.85
OR TRIGGERED_CHECKS != "NONE"
OR HAS_SUSPICIOUS_REASONS()
OR NUM_RESULTS < 2
```

### 가중치 투표 시스템

| 결과 유형       | 가중치 |
| --------------- | ------ |
| 원본 판정       | 1.5x   |
| Critical 근거   | 2.0x   |
| Suspicious 근거 | 1.2x   |
| 일반 Recheck    | 1.0x   |

---

## 🔄 워크플로우

1. **이미지 입력**: DACON에서 제공하는 dev.csv의 이미지 URL 로드
2. **Stage 1 관찰**: OpenAI GPT-4o로 이미지 관찰 (리드, 본체, 정렬 상태)
3. **Stage 2 판정**: 관찰 결과 기반 불량 여부 결정
4. **조건 평가**: 재검증 필요 여부 판단 (confidence, suspicious reasons 등)
5. **툴 호출**: 필요시 이미지 전처리 후 재검증
6. **결과 수집**: 모든 Stage 2 결과 수집
7. **가중치 투표**: 결과별 가중치 적용 후 최종 라벨 결정
8. **출력**: submission.csv에 최종 라벨 저장

---

## 🛠️ 이미지 전처리 도구

4가지 전처리 도구로 다양한 관점에서 이미지 분석:

![Preprocessing Tools](assets/preprocessing_tools.png)

| 도구                         | 기능             | Focus               |
| ---------------------------- | ---------------- | ------------------- |
| `preprocess_focus_leads`     | 리드 영역 강조   | 리드 형태, 휨, 간격 |
| `preprocess_focus_body`      | 본체 영역 강조   | 본체 정렬, 위치     |
| `preprocess_focus_lead_tips` | 리드 끝단 강조   | 홀 도달 여부        |
| `preprocess_full_enhanced`   | 전체 이미지 향상 | 종합 분석           |

---

## 🔍 주요 기능

### 1. Critical & Suspicious 근거 분류

**Critical Reasons** (확실한 불량 근거):

- `bent_lead`, `missing_lead`, `broken`, `short_circuit`, `lifted_lead`

**Suspicious Reasons** (추가 검증 필요):

- `blob_like`, `short_like`, `asymmetric`, `overlap`, `surface_mark`

### 2. LangSmith 트레이싱

모든 API 호출과 파이프라인 실행이 LangSmith에 기록됩니다:

- `Main Pipeline V6.1` → `Classify Agent` → `Stage1/2` → `LLM API Call`

### 3. 보수적 판정

동점(가중치 차이 < 0.1) 시 **abnormal(1)**로 판정하여 불량 누락 방지

---

## 📁 프로젝트 구조

```
hyu-hyundai-ngv-ai-agent-hackathon/
├── src/
│   ├── agent/
│   │   ├── agent_v6.py          # 메인 에이전트 (V6.1)
│   │   └── agent_with_langsmith_*.py  # 이전 버전들
│   └── preprocessing/
│       └── image_preprocessing_tools.py  # 이미지 전처리 도구
├── data/
│   ├── dev.csv                  # 개발 데이터셋
│   ├── dev_test.csv             # 테스트 데이터셋
│   └── dev_images/              # 개발 이미지 (라벨 포함)
├── assets/
│   ├── pipeline_architecture.png    # 아키텍처 다이어그램
│   └── preprocessing_tools.png      # 전처리 도구 예시
├── docs/
│   └── luxia_api_reference.md   # Luxia Cloud API 문서
├── preprocessed_images/         # 전처리 결과 이미지 (디버깅용)
├── requirements.txt
├── .env                         # API 키 설정
├── submission.csv               # 제출 파일
└── README.md
```

---

## ⚙️ 기술 스택

| 분류                 | 기술                            |
| -------------------- | ------------------------------- |
| **LLM**              | OpenAI GPT-4o (via Luxia Cloud) |
| **Tracing**          | LangSmith                       |
| **Image Processing** | OpenCV, Pillow                  |
| **Framework**        | Python, Pandas                  |
| **API**              | Luxia Cloud API                 |

---

## 🚀 실행 방법

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일 생성:

```env
LUXIA_API_KEY=your_luxia_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=semiconductor-defect-detection
```

### 3. 에이전트 실행

```bash
python src/agent/agent_v6.py
```

---

## 📊 결과

실행 후 `submission.csv`에 결과가 저장됩니다:

```csv
id,label
DEV_000,0
DEV_001,1
DEV_002,0
...
```

---

## 👥 팀 정보

| 역할 | 이름   | GitHub                                           |
| ---- | ------ | ------------------------------------------------ |
| 팀원 | 임동현 | [@Happ11quokka](https://github.com/Happ11quokka) |
| 팀원 | 서문경 | [@Munkyeong-Suh](https://github.com/Munkyeong-Suh) |

---

## 📝 라이선스

MIT License

---

## 📚 참고 문헌

본 프로젝트에서 사용한 방법론의 이론적 근거가 되는 논문들입니다.

### Chain-of-Thought Prompting (Two-Stage Reasoning)

| 논문                                                                                                      | 저자         | 학회/연도    | 핵심 내용                                                                                                                          |
| --------------------------------------------------------------------------------------------------------- | ------------ | ------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) | Wei et al.   | NeurIPS 2022 | 중간 추론 단계를 생성하면 LLM의 복잡한 추론 능력이 향상됨을 증명                                                                   |
| [Multimodal Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2302.00923)              | Zhang et al. | ACL 2023     | **Two-Stage Framework**: 1단계에서 rationale 생성, 2단계에서 답변 추론. 본 프로젝트의 Stage1(관찰)/Stage2(판정) 구조의 이론적 근거 |

### Self-Consistency & Voting (가중치 투표 시스템)

| 논문                                                                                                                 | 저자        | 학회/연도           | 핵심 내용                                                                                   |
| -------------------------------------------------------------------------------------------------------------------- | ----------- | ------------------- | ------------------------------------------------------------------------------------------- |
| [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)          | Wang et al. | ICLR 2023           | 다양한 추론 경로를 샘플링한 후 **다수결 투표**로 최종 답변 선택. GSM8K에서 +17.9% 성능 향상 |
| [Universal Self-Consistency for Large Language Model Generation](https://arxiv.org/abs/2311.17311)                   | Chen et al. | arXiv 2023          | Self-Consistency를 다양한 생성 태스크로 확장                                                |
| [Mirror-Consistency: Harnessing Inconsistency in Majority Voting](https://aclanthology.org/2024.findings-emnlp.135/) | -           | EMNLP 2024 Findings | 소수 의견도 정보를 포함할 수 있음을 고려한 개선된 투표 방식                                 |

### Confidence Calibration (신뢰도 기반 재검증)

| 논문                                                                                                                      | 저자         | 학회/연도  | 핵심 내용                                                                        |
| ------------------------------------------------------------------------------------------------------------------------- | ------------ | ---------- | -------------------------------------------------------------------------------- |
| [A Survey of Confidence Estimation and Calibration in Large Language Models](https://arxiv.org/abs/2311.08298)            | Geng et al.  | TMLR 2024  | LLM의 신뢰도 추정 및 calibration 방법론 종합 서베이                              |
| [Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation](https://arxiv.org/abs/2306.13063) | Xiong et al. | ICLR 2024  | LLM이 자신의 불확실성을 표현할 수 있는지 실증 평가                               |
| [SETS: Leveraging Self-Verification and Self-Correction for Improved Test-Time Scaling](https://arxiv.org/abs/2501.19306) | -            | arXiv 2025 | Self-Verification으로 calibration 향상. 본 프로젝트의 조건부 recheck 로직의 근거 |

### Vision-Language Models for Defect Detection

| 논문                                                                                                                          | 저자        | 학회/연도         | 핵심 내용                                        |
| ----------------------------------------------------------------------------------------------------------------------------- | ----------- | ----------------- | ------------------------------------------------ |
| [The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)](https://arxiv.org/abs/2309.17421)                             | Yang et al. | arXiv 2023        | GPT-4V의 시각적 이해 능력 종합 평가              |
| [GPT-4V(ision) as a Generalist Evaluator for Vision-Language Tasks](https://arxiv.org/abs/2311.01361)                         | -           | arXiv 2023        | GPT-4V를 다양한 비전-언어 태스크 평가자로 활용   |
| [Few-shot Learning for Defect Detection in Manufacturing](https://www.tandfonline.com/doi/full/10.1080/00207543.2024.2316279) | -           | IJPR 2024         | 제조업 결함 탐지를 위한 Few-shot 학습 방법론     |
| [LogicQA: Logical Anomaly Detection with Vision Language Models](https://aclanthology.org/2025.acl-industry.29/)              | -           | ACL 2025 Industry | VLM(GPT-4o, Gemini 등)을 활용한 논리적 이상 탐지 |

### 방법론 요약

```
본 프로젝트 적용 방법론:

1. Two-Stage Prompting (Zhang et al., 2023)
   Stage 1: 이미지 관찰 → 구조화된 JSON 출력
   Stage 2: JSON 기반 판정 → 라벨 + 신뢰도 + 근거

2. Self-Consistency Voting (Wang et al., 2023)
   - 다수결 투표 대신 가중치 투표 적용
   - Critical 근거(2.0x), Suspicious 근거(1.2x), 원본(1.5x)

3. Confidence-based Verification (Xiong et al., 2024)
   - Confidence < 0.85 → 자동 재검증 트리거
   - 불확실한 판정에 대해 추가 검증 수행
```

---

## 🔗 참고 자료

- [DACON 현대자동차그룹 x AI 해커톤](https://dacon.io/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Luxia Cloud API](https://luxia.cloud/)
