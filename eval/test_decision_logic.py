"""
Decision Engine Logic Test (deterministic, no LLM call)
==========================================================
Agent V10의 판단 로직 중 LLM 호출 없이 순수 Python으로 동작하는
결정론적 함수 2개를 검증합니다.

    1. vote_decision()               — 가중치 기반 투표 집계
    2. analyze_lead_hole_alignment() — YOLO 검출 결과 기반 리드-홀 정렬 분석

Stage 1/2 (GPT-4o) 자체는 LLM 호출이라 로컬에서 결정론적으로 재현할 수 없으므로,
이 테스트는 "LLM이 특정 결과를 반환했다고 가정했을 때 이후 로직이 문서(docs/V10_
PIPELINE.md)에 기술된 대로 동작하는가"를 검증하는 유닛 테스트입니다.

실행 방법:
    python eval/test_decision_logic.py
"""
import sys
import os
import importlib.util

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "agent"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "preprocessing"))

spec = importlib.util.spec_from_file_location(
    "agent_v10", os.path.join(PROJECT_ROOT, "src", "agent", "agent_v10.py")
)
agent_v10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_v10)


def test_vote_decision_docs_example():
    """docs/V10_PIPELINE.md '투표 예시' 절과 동일한 입력으로 검증."""
    print("TEST 1: docs/V10_PIPELINE.md worked example")
    results = [
        {"label": "abnormal", "confidence": 0.90, "key_reasons": ["right_lead end_position == missed_hole"]},
        {"label": "normal", "confidence": 0.82, "key_reasons": []},
        {"label": "abnormal", "confidence": 0.75, "key_reasons": []},
    ]
    metadata = [{"is_original": True}, {"is_original": False}, {"is_original": False}]
    label, conf, _, is_tie = agent_v10.vote_decision(results, result_metadata=metadata)
    print(f"  -> label={label}, confidence={conf:.3f}, is_tie={is_tie}")
    assert label == "abnormal"
    assert abs(conf - 0.81) < 0.01, f"confidence mismatch: got {conf}"
    print("  PASS\n")


def test_vote_decision_unanimous_normal():
    print("TEST 2: unanimous normal, no critical/suspicious reasons -> normal")
    results = [{"label": "normal", "confidence": 0.95, "key_reasons": []}]
    metadata = [{"is_original": True}]
    label, conf, _, _ = agent_v10.vote_decision(results, result_metadata=metadata)
    print(f"  -> label={label}, confidence={conf:.3f}")
    assert label == "normal"
    print("  PASS\n")


def test_vote_decision_tie_detection():
    print("TEST 3: near-equal weights -> is_tie correctly flagged")
    results = [
        {"label": "abnormal", "confidence": 0.60, "key_reasons": []},
        {"label": "normal", "confidence": 0.90, "key_reasons": []},
    ]
    metadata = [{"is_original": True}, {"is_original": False}]
    # original weight = 1.5 * 0.60 = 0.90 ; recheck weight = 1.0 * 0.90 = 0.90 -> tie
    label, conf, _, is_tie = agent_v10.vote_decision(results, result_metadata=metadata)
    print(f"  -> label={label}, confidence={conf:.3f}, is_tie={is_tie}")
    assert is_tie is True
    print("  PASS (is_tie correctly detected; note: exact-tie label pick is float-order")
    print("   dependent inside vote_decision() itself. The documented 'abnormal on tie'")
    print("   behavior is enforced one level up, in classify_agent()'s")
    print("   'while is_tie: do more rechecks' loop — agent_v10.py around lines 957-978.)\n")


def test_alignment_perfect_case():
    print("TEST 4: analyze_lead_hole_alignment on synthetic, perfectly-aligned YOLO output")
    synthetic_yolo = {
        "holes": [
            {"x": 100, "y": 400, "w": 20, "h": 20, "confidence": 0.9},
            {"x": 150, "y": 400, "w": 20, "h": 20, "confidence": 0.9},
            {"x": 200, "y": 400, "w": 20, "h": 20, "confidence": 0.9},
        ],
        "leads": [
            {"x": 100, "y": 390, "w": 5, "h": 20, "confidence": 0.9},
            {"x": 150, "y": 390, "w": 5, "h": 20, "confidence": 0.9},
            {"x": 200, "y": 390, "w": 5, "h": 20, "confidence": 0.9},
        ],
        "body": {"x": 150, "y": 100, "w": 100, "h": 60, "confidence": 0.9},
        "detection_success": True,
    }
    alignment = agent_v10.analyze_lead_hole_alignment(synthetic_yolo)
    print(f"  -> alignment_issues={alignment['alignment_issues']}")
    assert alignment["alignment_issues"] == []
    print("  PASS\n")


if __name__ == "__main__":
    test_vote_decision_docs_example()
    test_vote_decision_unanimous_normal()
    test_vote_decision_tie_detection()
    test_alignment_perfect_case()
    print("ALL LOGIC TESTS PASSED")
