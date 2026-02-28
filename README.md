# dAIso Agent

**Defect Analysis & Inspection System with OpenAI**

<img width="1162" height="206" alt="image" src="https://github.com/user-attachments/assets/3f496fce-08a6-4d5e-b288-abc66d861da6" />

반도체 부품 불량 검출을 위한 Two-Stage Prompting 기반 AI Agent

> [!IMPORTANT]
> **DACON 현대자동차그룹 x AI 해커톤 — 최우수상 수상**

[![Award](https://img.shields.io/badge/DACON-최우수상-FFD700?logo=trophy&logoColor=white)](https://dacon.io/)
[![Live Demo](https://img.shields.io/badge/Demo-Hugging%20Face-FF9D00?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/promise42da/dAIso)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![GPT-4o](https://img.shields.io/badge/GPT--4o-412991?logo=openai&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Roboflow](https://img.shields.io/badge/Roboflow-6706CE?logo=roboflow&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?logo=gradio&logoColor=white)
![LangSmith](https://img.shields.io/badge/LangSmith-1C3C3C?logo=langchain&logoColor=white)

---

## 프로젝트 개요

dAIso Agent는 반도체 부품(TO-220 패키지) 이미지를 분석하여 불량 여부를 자동 판정하는 AI 에이전트입니다. YOLO 기반 컴포넌트 탐지와 GPT-4o Two-Stage Prompting을 결합하고, 신뢰도 기반 조건부 Recheck 및 가중치 투표로 최종 판정의 정확도를 높입니다.

---

## 담당 역할

**임동현** — AI Pipeline, Demo, Data (2인 팀)

| 영역 | 기여 내용 |
|------|-----------|
| AI Pipeline | Two-Stage Prompting 설계, 조건부 Recheck 로직, 가중치 투표 시스템 |
| Object Detection | Roboflow 데이터셋 구축, YOLO Workflow 설계 |
| 이미지 전처리 | OpenCV 기반 4종 전처리 도구 개발 |
| Demo | Gradio 웹 데모 개발 + Hugging Face Spaces 배포 |
| Data | 자동 라벨링 파이프라인, 라벨 편집기 개발 |
| Tracing | LangSmith 연동을 통한 프롬프트 추적 및 디버깅 |

---

## Demo

> **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/promise42da/dAIso)** — 실시간으로 분석 결과를 확인하고 AI Assistant와 대화할 수 있습니다.

![Pipeline Architecture](assets/pipeline_architecture.png)

---

## 시스템 아키텍처

### Two-Stage Prompting

| Stage | 역할 | 입력 | 출력 |
|-------|------|------|------|
| **Stage 0** | 컴포넌트 탐지 (YOLO) | 이미지 | Bounding Box + Confidence |
| **Stage 1** | 이미지 관찰 (GPT-4o) | 이미지 + YOLO 힌트 | 구조화된 관찰 JSON |
| **Stage 2** | 불량 판정 (GPT-4o) | 관찰 JSON | 라벨, 신뢰도, 근거 |

### 조건부 Recheck

신뢰도가 85% 미만이거나, 의심 근거가 존재하거나, 결과 수가 부족한 경우 4가지 전처리 도구를 활용하여 자동으로 재검증을 수행합니다.

### 가중치 투표 시스템

| 결과 유형 | 가중치 |
|-----------|--------|
| 원본 판정 | 1.5x |
| Critical 근거 | 2.0x |
| Suspicious 근거 | 1.2x |
| 일반 Recheck | 1.0x |

---

## 이미지 전처리 도구

4가지 전처리 도구로 다양한 관점에서 이미지 분석:

![Preprocessing Tools](assets/preprocessing_tools.png)

| 도구 | 기능 | Focus |
|------|------|-------|
| `preprocess_focus_leads` | 리드 영역 강조 | 리드 형태, 휨, 간격 |
| `preprocess_focus_body` | 본체 영역 강조 | 본체 정렬, 위치 |
| `preprocess_focus_lead_tips` | 리드 끝단 강조 | 홀 도달 여부 |
| `preprocess_full_enhanced` | 전체 이미지 향상 | 종합 분석 |

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| **LLM** | GPT-4o (via Luxia Cloud Bridge API) |
| **Object Detection** | Roboflow Workflow API |
| **Tracing** | LangSmith |
| **Image Processing** | OpenCV, Pillow |
| **Framework** | Python, Pandas |
| **Demo UI** | Gradio (Hugging Face Spaces) |

---

## 프로젝트 구조

```
dAIso-Agent/
├── src/
│   ├── agent/
│   │   └── agent_v10.py              # 최종 에이전트
│   ├── preprocessing/
│   │   └── image_preprocessing_tools.py
│   └── labeling/
│       ├── auto_label.py             # 자동 라벨링
│       ├── label_editor.py           # 라벨 편집기
│       └── roboflow_label.py         # Roboflow 라벨링
├── dAIso/                            # Hugging Face Demo
│   ├── app.py                        # Gradio 앱
│   ├── requirements.txt
│   └── data/
│       ├── dummy_results.json
│       └── dev_images/
├── data/
│   ├── dev.csv
│   └── dev_images/
├── assets/
│   ├── pipeline_architecture.png
│   └── preprocessing_tools.png
├── requirements.txt
└── README.md
```

---

## 실행 방법

```bash
# 환경 설정
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`.env` 파일 생성:

```env
LUXIA_API_KEY=your_luxia_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=semiconductor-defect-detection
```

```bash
# 에이전트 실행
python src/agent/agent_v10.py

# 로컬 데모 실행
cd dAIso && pip install -r requirements.txt && python app.py
```

---

## 팀 정보

| 이름 | 역할 | GitHub |
|------|------|--------|
| **임동현** | AI Pipeline, Demo, Data | [@Happ11quokka](https://github.com/Happ11quokka) |
| 서문경 | Data Analysis | [@Munkyeong-Suh](https://github.com/Munkyeong-Suh) |

---

## 참고 문헌

<details>
<summary>논문 목록 (8편)</summary>

### Chain-of-Thought Prompting (Two-Stage Reasoning)

| 논문 | 저자 | 학회/연도 | 핵심 내용 |
|------|------|-----------|-----------|
| [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) | Wei et al. | NeurIPS 2022 | 중간 추론 단계를 생성하면 LLM의 복잡한 추론 능력이 향상됨을 증명 |
| [Multimodal Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2302.00923) | Zhang et al. | ACL 2023 | Two-Stage Framework: 1단계에서 rationale 생성, 2단계에서 답변 추론 |

### Self-Consistency & Voting (가중치 투표 시스템)

| 논문 | 저자 | 학회/연도 | 핵심 내용 |
|------|------|-----------|-----------|
| [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) | Wang et al. | ICLR 2023 | 다양한 추론 경로를 샘플링한 후 다수결 투표로 최종 답변 선택 |
| [Mirror-Consistency: Harnessing Inconsistency in Majority Voting](https://aclanthology.org/2024.findings-emnlp.135/) | - | EMNLP 2024 | 소수 의견도 정보를 포함할 수 있음을 고려한 개선된 투표 방식 |

### Confidence Calibration (신뢰도 기반 재검증)

| 논문 | 저자 | 학회/연도 | 핵심 내용 |
|------|------|-----------|-----------|
| [A Survey of Confidence Estimation and Calibration in Large Language Models](https://arxiv.org/abs/2311.08298) | Geng et al. | TMLR 2024 | LLM의 신뢰도 추정 및 calibration 방법론 종합 서베이 |
| [Can LLMs Express Their Uncertainty?](https://arxiv.org/abs/2306.13063) | Xiong et al. | ICLR 2024 | LLM이 자신의 불확실성을 표현할 수 있는지 실증 평가 |

### Vision-Language Models for Defect Detection

| 논문 | 저자 | 학회/연도 | 핵심 내용 |
|------|------|-----------|-----------|
| [The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)](https://arxiv.org/abs/2309.17421) | Yang et al. | arXiv 2023 | GPT-4V의 시각적 이해 능력 종합 평가 |
| [LogicQA: Logical Anomaly Detection with Vision Language Models](https://aclanthology.org/2025.acl-industry.29/) | - | ACL 2025 Industry | VLM을 활용한 논리적 이상 탐지 |

</details>

---

## 참고 자료

- [DACON 현대자동차그룹 x AI 해커톤](https://dacon.io/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Luxia Cloud API](https://luxia.cloud/)
- [Roboflow](https://roboflow.com/)

## 라이선스

MIT License
