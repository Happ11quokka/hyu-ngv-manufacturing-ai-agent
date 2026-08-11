"""
Stage 0 (YOLO / Roboflow) Real API Smoke Test
================================================
포트폴리오 정리 과정에서 Agent V10 파이프라인의 Stage 0(YOLO 객체 검출)을
실제 20장의 dev 이미지(data/dev.csv)에 대해 Roboflow Workflow API로
**실측** 호출한 스모크 테스트 스크립트입니다.

주의 (정직성 표기):
- Stage 0 (YOLO/Roboflow)만 실제 API를 호출합니다. Roboflow API 키는
  src/agent/agent_v10.py 에 해커톤 당시 공개용으로 커밋된 기본값을 그대로 사용합니다.
- Stage 1/2 (GPT-4o via Luxia Cloud Bridge)는 해커톤 주최측이 발급한
  LUXIA_API_KEY가 있어야 호출 가능하며, 이 키는 대회 종료 후 회수되어
  본 포트폴리오 정리 환경에서는 재현할 수 없습니다. 따라서 이 스크립트는
  Stage 0 + 결정론적 정렬 분석(analyze_lead_hole_alignment)까지만 실측합니다.
  Stage 1/2 판단 로직은 eval/test_decision_logic.py 에서 별도로
  (실제 이미지 없이) 로직 단위 테스트로 검증합니다.

실행 방법:
    pip install inference-sdk pandas
    python eval/test_stage0_smoke.py

출력:
    eval/results/stage0_smoke_test_results.json
"""
import sys
import os
import json
import time
import importlib.util

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "agent"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "preprocessing"))

import pandas as pd  # noqa: E402

# agent_v10.py를 재구현하지 않고 그대로 import하여 실제 코드 경로를 검증합니다.
spec = importlib.util.spec_from_file_location(
    "agent_v10", os.path.join(PROJECT_ROOT, "src", "agent", "agent_v10.py")
)
agent_v10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_v10)


def main():
    print(f"Roboflow key(masked)={agent_v10.ROBOFLOW_API_KEY[:4]}***  "
          f"workspace={agent_v10.ROBOFLOW_WORKSPACE}  "
          f"workflow={agent_v10.ROBOFLOW_WORKFLOW_ID}")

    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "dev.csv"))
    print(f"dev.csv rows: {len(df)}")

    results = []
    for _, row in df.iterrows():
        _id, url = row["id"], row["img_url"]
        t0 = time.time()
        try:
            yolo_result = agent_v10.stage0_yolo_detect(url, verbose=False)
            alignment = (
                agent_v10.analyze_lead_hole_alignment(yolo_result)
                if yolo_result.get("detection_success")
                else None
            )
            elapsed = time.time() - t0
            rec = {
                "id": _id,
                "detection_success": yolo_result.get("detection_success"),
                "n_holes": len(yolo_result.get("holes", [])),
                "n_leads": len(yolo_result.get("leads", [])),
                "body_detected": yolo_result.get("body") is not None,
                "alignment_issues": alignment.get("alignment_issues") if alignment else None,
                "elapsed_sec": round(elapsed, 2),
                "error": yolo_result.get("error"),
            }
        except Exception as e:
            rec = {"id": _id, "detection_success": False, "error": str(e)}
        print(rec)
        results.append(rec)

    out_path = os.path.join(PROJECT_ROOT, "eval", "results", "stage0_smoke_test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n_success = sum(1 for r in results if r.get("detection_success"))
    print(f"\n=== SUMMARY: {n_success}/{len(results)} images detected successfully ===")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
