import gradio as gr
import requests
import base64
import io
from PIL import Image

API_BASE = "http://localhost:8000"
_session_id = None


def upload_files(pdf_path, figure_path):
    global _session_id
    if pdf_path is None or figure_path is None:
        return None, "Upload both a PDF and a figure image!"

    with open(pdf_path, 'rb') as pf, open(figure_path, 'rb') as ff:
        resp = requests.post(
            f"{API_BASE}/upload",
            files={
                "pdf": ("paper.pdf", pf, "application/pdf"),
                "figure": ("figure.png", ff, "image/png")
            }
        )
    if resp.status_code != 200:
        return None, f"Upload error: {resp.text}"

    data = resp.json()
    _session_id = data["session_id"]

    img_bytes = base64.b64decode(data["figure_preview_base64"])
    preview = Image.open(io.BytesIO(img_bytes))
    return preview, f"Ready. Session: {_session_id[:8]}..."


def ask_question(question):
    global _session_id
    if not _session_id:
        return "Upload files first.", "", "", ""
    if not question.strip():
        return "Enter a question.", "", "", ""

    resp = requests.post(
        f"{API_BASE}/ask",
        json={"session_id": _session_id, "question": question}
    )
    if resp.status_code != 200:
        return f"Error: {resp.text}", "", "", ""

    data = resp.json()

    a = data["pipeline_a_vision_only"]
    b = data["pipeline_b_rag_only"]
    c = data["pipeline_c_multimodal_rag"]

    vision_out = f"{a['answer']}\n\n⏱ {a['latency']}s"
    rag_out = f"{b['answer']}\n\n⏱ {b['latency']}s"
    mm_out = f"{c['answer']}\n\n⏱ {c['latency']}s"

    ev = c["evidence"]
    evidence_out = (
        f"**Visual Description (LLaVA Caption):**\n{ev['visual_description']}\n\n"
        f"**OCR Text Extracted from Figure:**\n{ev['ocr_text']}\n\n"
        f"**Top Retrieved Paper Passages:**\n\n"
        + "\n---\n".join(ev['retrieved_passages'][:2])
    )

    return vision_out, rag_out, mm_out, evidence_out


with gr.Blocks(title="Multimodal Research Assistant") as demo:
    gr.Markdown("""
    # 🔬 Multimodal Research Assistant
    Upload a paper PDF + one of its figures. Ask any question.
    Three pipelines answer in parallel — see which one wins and why.
    
    **Pipeline A:** LLaVA vision-only &nbsp;|&nbsp; **Pipeline B:** RAG text-only &nbsp;|&nbsp; **Pipeline C:** Multimodal RAG (this system)
    """)

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Paper PDF", file_types=[".pdf"])
            figure_input = gr.File(label="Figure Image", file_types=["image"])
            upload_btn = gr.Button("Upload & Process", variant="primary")
            figure_preview = gr.Image(label="Figure Preview", height=280)
            status_box = gr.Textbox(label="Status", interactive=False, value="No files uploaded.")

        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g. Does this graph support the paper's main claim?",
                lines=2
            )
            ask_btn = gr.Button("Ask All Three Pipelines", variant="primary")

            with gr.Row():
                vision_out = gr.Textbox(
                    label="Pipeline A — Vision Only (LLaVA)",
                    lines=7
                )
                rag_out = gr.Textbox(
                    label="Pipeline B — RAG Only (Text)",
                    lines=7
                )

            mm_out = gr.Textbox(
                label="Pipeline C — Multimodal RAG [YOUR SYSTEM]",
                lines=9
            )
            evidence_out = gr.Textbox(
                label="Evidence Used by Pipeline C",
                lines=7
            )

    upload_btn.click(
        fn=upload_files,
        inputs=[pdf_input, figure_input],
        outputs=[figure_preview, status_box]
    )
    ask_btn.click(
        fn=ask_question,
        inputs=[question_input],
        outputs=[vision_out, rag_out, mm_out, evidence_out]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())