from PIL import Image
from backend.rag.ingest import build_vectorstore_from_pdf
from backend.fusion.multimodal_pipeline import run_all_pipelines

vs = build_vectorstore_from_pdf("data/test.pdf", "checkpoint_phase2")
img = Image.open("data/test_figure.png").convert("RGB")

result = run_all_pipelines(img, "What does this figure show?", vs)

for key in ["pipeline_a_vision_only", "pipeline_b_rag_only", "pipeline_c_multimodal_rag"]:
    ans = result[key]["answer"]
    assert len(ans) > 10, f"{key} returned too-short answer: {ans}"
    print(f"[{key}] {ans[:120]}...")
def evaluate_result(result):
    scores = {}

    for key in ["pipeline_a_vision_only", "pipeline_b_rag_only", "pipeline_c_multimodal_rag"]:
        answer = result[key]["answer"]

        scores[key] = {
            "length": len(answer),
            "uses_visual": "[VISUAL]" in answer,
            "uses_paper": "[PAPER" in answer or "[PAPER:" in answer
        }

    return scores


# RUN EVALUATION
scores = evaluate_result(result)
print("\nEvaluation:")
for k, v in scores.items():
    print(f"{k}: {v}")
# Check Pipeline C evidence structure
evidence = result["pipeline_c_multimodal_rag"]["evidence"]
assert "visual_description" in evidence
assert "ocr_text" in evidence
assert "retrieved_passages" in evidence
assert len(evidence["retrieved_passages"]) == 4

print("Phase 2 PASSED")