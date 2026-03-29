from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

MODEL_ID = "Salesforce/blip2-flan-t5-xl"

_processor = None
_model = None


def _load():
    global _processor, _model
    if _model is not None:
        return
    print(f"[vision] Loading {MODEL_ID}...")
    _processor = Blip2Processor.from_pretrained(MODEL_ID)
    _model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    print("[vision] Model loaded.")


def _run(image: Image.Image, prompt: str) -> str:
    _load()
    inputs = _processor(
        images=image,
        text=f"Question: {prompt} Answer:",
        return_tensors="pt"
    )
    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )
    return _processor.decode(output[0], skip_special_tokens=True).strip()


def vision_only_answer(image: Image.Image, question: str) -> str:
    return _run(image, question)


def generate_figure_caption(image: Image.Image) -> str:
    return _run(image, "Describe this scientific figure including axes, labels, and trends.")