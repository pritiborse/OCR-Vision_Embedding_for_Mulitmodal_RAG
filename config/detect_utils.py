import fitz  # PyMuPDF
from PIL import Image
import os
import io
from config.ocr_utils import extract_ocr_from_image

MAX_PAGES = None  # Process all pages


def extract_images_and_captions(pdf_path, output_dir="outputs/hybrid_index/extracted_images"):
    """
    Extracts images from a PDF and associates nearby text captions.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    image_data = []

    for page_num, page in enumerate(pdf_document, start=1):
        if MAX_PAGES and page_num > MAX_PAGES:
            break

        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            try:
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
            except Exception:
                continue

            # Skip tiny images
            if image.width < 50 or image.height < 50:
                continue

            # Save image
            image_filename = f"page_{page_num}_img_{img_index}.png"
            image_path = os.path.join(output_dir, image_filename)
            image.save(image_path)

            # Get caption from nearby text
            caption = find_nearby_text(page, img, xref)

            # Extract OCR text from image
            ocr_text, ocr_conf = extract_ocr_from_image(image)

            # Combine caption and OCR for better context
            combined_text = f"{caption} {ocr_text}".strip()

            image_data.append({
                "page": page_num,
                "image_path": image_path,
                "caption": caption,
                "ocr_text": ocr_text,
                "combined_text": combined_text,
                "ocr_conf": ocr_conf
            })

    pdf_document.close()
    return image_data


def find_nearby_text(page, img_info, img_xref, margin=100):
    """Finds text blocks close to image."""
    text_parts = []
    blocks = page.get_text("dict")["blocks"]

    # Try to get image bbox
    try:
        image_list = page.get_images(full=True)
        image_name = None
        for xref, smask, width, height, bpc, cs, alt, name, *_ in image_list:
            if xref == img_xref:
                image_name = name
                break
        
        if image_name:
            bbox = page.get_image_bbox(image_name)
        else:
            return ""
    except Exception:
        return ""

    # Collect text near image
    for block in blocks:
        if "lines" in block:
            block_bbox = block["bbox"]
            # Check if block is near image (above, below, or sides)
            vertical_dist = min(abs(block_bbox[1] - bbox[3]), abs(block_bbox[3] - bbox[1]))
            horizontal_dist = min(abs(block_bbox[0] - bbox[2]), abs(block_bbox[2] - bbox[0]))
            
            if vertical_dist < margin or horizontal_dist < margin:
                block_text = " ".join(
                    [span["text"] for line in block["lines"] for span in line["spans"]]
                )
                text_parts.append(block_text.strip())

    return " ".join(text_parts).strip()