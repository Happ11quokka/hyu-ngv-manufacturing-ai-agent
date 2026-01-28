---
title: dAIso - Semiconductor Defect Detection
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.36.1
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# dAIso - Semiconductor Defect Detection Demo

**Agent V10: YOLO + Two-Stage GPT-4o Pipeline**

This is an interactive demo showcasing the semiconductor defect detection system developed for the Hyundai NGV AI Agent Hackathon 2026.

## Features

### 1. Dashboard
- Overall analysis statistics
- Classification distribution (Normal vs Abnormal)
- Defect type breakdown
- Performance metrics

### 2. Inspection Results
- Individual image analysis details
- YOLO detection results (holes, leads, body)
- Stage 1: GPT-4o visual observations
- Stage 2: Decision making and defect scoring
- Recheck information

### 3. Visualizations
- Classification pie charts
- Defect type bar charts
- Confidence score distributions
- Processing time analysis
- YOLO detection performance

### 4. AI Assistant (Chatbot)
Ask questions about the analysis results in Korean or English:
- "전체 분석 요약해줘"
- "어떤 결함이 가장 많아?"
- "DEV_005 결과 알려줘"
- "YOLO 탐지 성능은?"

## Pipeline Architecture

```
Image Input
    │
    ▼
┌─────────────────────┐
│  Stage 0: YOLO      │  → Component Detection
│  (Roboflow API)     │  → Bounding boxes
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Stage 1: GPT-4o    │  → Visual Observation
│  (Observation)      │  → Body/Lead analysis
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Stage 2: GPT-4o    │  → Defect scoring
│  (Decision)         │  → Classification
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Conditional        │  → Confidence check
│  Recheck            │  → Weighted voting
└─────────────────────┘
```

## Tech Stack

- **LLM**: GPT-4o (via Luxia Cloud Bridge API)
- **Object Detection**: Roboflow Workflow API
- **UI**: Gradio
- **Visualization**: Matplotlib

## Team

- Dong-Hyeon Lim
- Munkyeong Suh

---

Hyundai NGV AI Agent Hackathon 2026
