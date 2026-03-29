import io
import base64
from PIL import Image
from backend.core.config import settings


class ImageProcessor:

    @staticmethod
    def load_from_bytes(image_bytes: bytes) -> Image.Image:
        """Load and validate image from bytes. Raises ValueError on corrupt input."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
            img = Image.open(io.BytesIO(image_bytes))  # reopen after verify
            return img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid or corrupt image: {e}")

    @staticmethod
    def resize_if_needed(img: Image.Image, max_size: int = None) -> Image.Image:
        """Resize to fit within max_size x max_size, preserving aspect ratio."""
        max_size = max_size or settings.max_image_size
        w, h = img.size
        if max(w, h) <= max_size:
            return img
        scale = max_size / max(w, h)
        return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    @staticmethod
    def to_base64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()

    def process(self, image_bytes: bytes) -> dict:
        img = self.load_from_bytes(image_bytes)
        original_size = img.size
        img = self.resize_if_needed(img)
        return {
            "image": img,
            "original_size": original_size,
            "processed_size": img.size,
            "base64": self.to_base64(img)
        }