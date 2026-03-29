from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

_ocr_instance = None


def get_ocr() -> PaddleOCR:
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _ocr_instance


def extract_text_from_figure(img: Image.Image, confidence_threshold: float = 0.7) -> str:
    """
    Extract all text visible in a scientific figure via PaddleOCR.
    Covers: axis labels, tick values, legend text, annotations, titles.
    Only returns detections above confidence_threshold (default 0.7).
    Returns pipe-delimited string of detected text segments.
    Returns 'No text detected in figure' if nothing found above threshold.
    """
    ocr = get_ocr()
    img_np = np.array(img)
    result = ocr.ocr(img_np, cls=True)

    texts = []
    if result and result[0]:
        for line in result[0]:
            text, confidence = line[1][0], line[1][1]
            if confidence >= confidence_threshold:
                texts.append(text.strip())

    return " | ".join(texts) if texts else "No text detected in figure"