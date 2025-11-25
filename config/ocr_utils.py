import pytesseract
from PIL import Image
import re

def extract_ocr_from_image(image):
    """
    Extract text from an image using Tesseract OCR and clean it.
    Returns:
        text (str): Cleaned OCR text
        confidence (float): Average confidence score (if available)
    """
    try:
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        text_blocks = [ocr_data["text"][i] for i in range(len(ocr_data["text"])) if len(ocr_data["text"][i].strip()) > 2]
        text = " ".join(text_blocks)

        # Clean up unwanted characters
        text = re.sub(r"[^A-Za-z0-9\s.,:;!?%()'’\-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Compute average confidence
        conf_scores = [int(c) for c in ocr_data["conf"] if c.isdigit() and int(c) >= 0]
        avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0

        return text, avg_conf

    except Exception as e:
        return "", 0.0
