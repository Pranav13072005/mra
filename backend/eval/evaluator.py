import json
import time
from groq import Groq
from collections import defaultdict
from backend.core.config import settings

client = Groq(api_key=settings.groq_api_key)

# ---------------------------------------------------------------------------
# LLM-AS-JUDGE PROMPT
# Fixed. Applied identically to all three pipelines. Never change during eval.
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are evaluating the quality of an AI-generated answer to a question about a scientific figure.

Question: {question}
Reference Answer: {reference_answer}
Generated Answer: {generated_answer}

Score the generated answer on THREE dimensions.
For each, output a float between 0.0 and 1.0.
Output ONLY a valid JSON object — no explanation, no markdown, no prefix.

{{
  "correctness": <float>,
  "groundedness": <float>,
  "hallucination": <float>
}}

Scoring criteria:
- correctness:   How factually accurate is the answer compared to the reference?
                 0.0 = completely wrong, 1.0 = fully correct
- groundedness:  Are the claims in the answer supported by the question context
                 (retrieved passages, visual description)?
                 0.0 = fully hallucinated, 1.0 = fully grounded in evidence
- hallucination: Does the answer contain specific claims that contradict or are
                 absent from both the reference and the question context?
                 0.0 = no hallucination, 1.0 = severe hallucination

IMPORTANT: hallucination is scored 0=none, 1=severe (opposite direction from others).
"""


def judge(question: str, reference: str, generated: str) -> dict:
    """
    LLM-as-judge: scores one generated answer on correctness, groundedness, hallucination.
    Returns dict with three float scores, or zeros on parse failure.
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference_answer=reference,
        generated_answer=generated
    )
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60
        )
        raw = resp.choices[0].message.content.strip()
        scores = json.loads(raw)
        return {
            "correctness": float(scores.get("correctness", 0.0)),
            "groundedness": float(scores.get("groundedness", 0.0)),
            "hallucination": float(scores.get("hallucination", 0.0)),
            "hallucination_free": round(1.0 - float(scores.get("hallucination", 0.0)), 4)
        }
    except Exception as e:
        print(f"[judge] Parse error: {e} | raw: {raw}")
        return {"correctness": 0.0, "groundedness": 0.0,
                "hallucination": 0.0, "hallucination_free": 1.0}


# ---------------------------------------------------------------------------
# FAILURE MODE DETECTION
# ---------------------------------------------------------------------------

def detect_failure_modes(
    item: dict,
    result: dict,
    scores_a: dict,
    scores_b: dict,
    scores_c: dict
) -> dict:
    """
    Classify failure modes per v2 spec taxonomy.

    FM-1 Retrieval Noise:
      Pipeline C correctness < Pipeline A correctness by >= 0.1
      AND figure not mentioned in retrieved passages

    FM-2 Visual Misinterpretation:
      Pipeline C correctness < Pipeline B correctness by >= 0.1
      (suggests caption error; flag for manual review)

    FM-3 Modality Conflict:
      Directly from eval_set.json modality_conflict field
    """
    delta_c_vs_a = scores_c["correctness"] - scores_a["correctness"]
    delta_c_vs_b = scores_c["correctness"] - scores_b["correctness"]

    passages = result["pipeline_c_multimodal_rag"]["evidence"].get("retrieved_passages", [])
    figure_mentioned = any(
        any(kw in p.lower() for kw in ["figure", "fig.", "fig ", "shown in", "plot"])
        for p in passages
    )

    return {
        "retrieval_noise": delta_c_vs_a < -0.1 and not figure_mentioned,
        "visual_misinterpretation": delta_c_vs_b < -0.1,
        "modality_conflict": item.get("modality_conflict", False)
    }


# ---------------------------------------------------------------------------
# BENCHMARK RUNNER
# ---------------------------------------------------------------------------

def run_benchmark(eval_set_path: str, output_path: str):
    from PIL import Image
    from backend.rag.ingest import build_vectorstore_from_pdf
    from backend.fusion.multimodal_pipeline import run_all_pipelines

    with open(eval_set_path) as f:
        eval_set = json.load(f)

    results = []

    for i, item in enumerate(eval_set):
        print(f"\n[{i+1}/{len(eval_set)}] {item['id']} | type={item['question_type']}")

        # Build session vectorstore from paper
        vs = build_vectorstore_from_pdf(item["paper_pdf"], f"eval_{item['id']}")
        img = Image.open(item["figure_path"]).convert("RGB")

        # Run all three pipelines
        result = run_all_pipelines(img, item["question"], vs)

        # Score all three pipelines
        scores_a = judge(item["question"], item["reference_answer"],
                         result["pipeline_a_vision_only"]["answer"])
        scores_b = judge(item["question"], item["reference_answer"],
                         result["pipeline_b_rag_only"]["answer"])
        scores_c = judge(item["question"], item["reference_answer"],
                         result["pipeline_c_multimodal_rag"]["answer"])

        # Detect failure modes
        failure_flags = detect_failure_modes(item, result, scores_a, scores_b, scores_c)

        record = {
            "id": item["id"],
            "question_type": item["question_type"],
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "pipeline_a_answer": result["pipeline_a_vision_only"]["answer"],
            "pipeline_b_answer": result["pipeline_b_rag_only"]["answer"],
            "pipeline_c_answer": result["pipeline_c_multimodal_rag"]["answer"],
            "scores_a": scores_a,
            "scores_b": scores_b,
            "scores_c": scores_c,
            "failure_flags": failure_flags
        }
        results.append(record)

        print(f"  A: correct={scores_a['correctness']:.2f} ground={scores_a['groundedness']:.2f} halluc_free={scores_a['hallucination_free']:.2f}")
        print(f"  B: correct={scores_b['correctness']:.2f} ground={scores_b['groundedness']:.2f} halluc_free={scores_b['hallucination_free']:.2f}")
        print(f"  C: correct={scores_c['correctness']:.2f} ground={scores_c['groundedness']:.2f} halluc_free={scores_c['hallucination_free']:.2f}")
        print(f"  Failure flags: {failure_flags}")

        # Rate limit buffer
        time.sleep(1.5)

    # ---------------------------------------------------------------------------
    # AGGREGATION
    # ---------------------------------------------------------------------------
    by_type = defaultdict(lambda: {"a": defaultdict(list), "b": defaultdict(list), "c": defaultdict(list)})
    for r in results:
        t = r["question_type"]
        for metric in ["correctness", "groundedness", "hallucination_free"]:
            by_type[t]["a"][metric].append(r["scores_a"][metric])
            by_type[t]["b"][metric].append(r["scores_b"][metric])
            by_type[t]["c"][metric].append(r["scores_c"][metric])

    def mean(lst): return round(sum(lst) / len(lst), 3) if lst else 0.0

    print("\n" + "="*80)
    print("RESULTS TABLE — CORRECTNESS")
    print(f"{'Type':<22} {'A:Vision':>10} {'B:RAG':>8} {'C:MM-RAG':>10} {'Winner':>8}")
    print("-"*65)
    for qtype in ["factual", "visual_trend", "cross_verification"]:
        a = mean(by_type[qtype]["a"]["correctness"])
        b = mean(by_type[qtype]["b"]["correctness"])
        c = mean(by_type[qtype]["c"]["correctness"])
        winner = ["A", "B", "C"][[a, b, c].index(max(a, b, c))]
        print(f"{qtype:<22} {a:>10.3f} {b:>8.3f} {c:>10.3f} {winner:>8}")

    print("\nRESULTS TABLE — GROUNDEDNESS")
    print(f"{'Type':<22} {'A:Vision':>10} {'B:RAG':>8} {'C:MM-RAG':>10}")
    print("-"*55)
    for qtype in ["factual", "visual_trend", "cross_verification"]:
        a = mean(by_type[qtype]["a"]["groundedness"])
        b = mean(by_type[qtype]["b"]["groundedness"])
        c = mean(by_type[qtype]["c"]["groundedness"])
        print(f"{qtype:<22} {a:>10.3f} {b:>8.3f} {c:>10.3f}")

    print("\nFAILURE MODE SUMMARY")
    fm1 = sum(1 for r in results if r["failure_flags"]["retrieval_noise"])
    fm2 = sum(1 for r in results if r["failure_flags"]["visual_misinterpretation"])
    fm3 = sum(1 for r in results if r["failure_flags"]["modality_conflict"])
    print(f"  FM-1 Retrieval Noise:          {fm1}/{len(results)} items")
    print(f"  FM-2 Visual Misinterpretation: {fm2}/{len(results)} items")
    print(f"  FM-3 Modality Conflict:        {fm3}/{len(results)} items")
    print("="*80)

    # Save full results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    run_benchmark(
        "data/benchmark/eval_set.json",
        "data/benchmark/results.json"
    )