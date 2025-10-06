import os
import pypdfium2 as pdfium
from typing import List

def pdf_to_images(
        pdf_path: str,
        output_path: str,
        format: str="jpeg",
        scale: float=2.0
) -> List[str]:
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'")
        return []
    
    os.makedirs(output_path, exist_ok=True)

    generated_files = []

    try:
        print(f"Loading PDF from {pdf_path}")

        pdf_document = pdfium.PdfDocument(pdf_path)
        n_pages = len(pdf_document)
        print(f"Found {n_pages} pages in the document")

        for page_number in range(n_pages):
            page = pdf_document.get_page(page_number)
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            output_filename = f"page_{page_number + 1:03d}.{format}"
            output_filepath = os.path.join(output_path, output_filename)

            if format.lower() == "jpeg":
                pil_image.save(output_filepath, format=format, quality=90)
            else:
                pil_image.save(output_filepath, format=format)

            generated_files.append(output_filepath)

            page.close()

        print("\nConversin completed!")
        return generated_files
    
    except Exception as e:
        print(f"An unexpected error occured during conversion: {e}")
        return []
    
# --- CONVERT PDF -> JPEG --- 
PDF_FILE = "TÀI LIỆU ĐÀO TẠO AN TOÀN THÔNG TIN 2025.pdf"
OUTPUT_FOLDER = "./converted_images"

if __name__ == "__main__":
    if PDF_FILE == "TÀI LIỆU ĐÀO TẠO AN TOÀN THÔNG TIN …" and not os.path.exists(PDF_FILE):
        print("""
              ACTION REQUIRED:
              f"Please make sure the PDF path provided is valid!
              """)
    else:
        converted_images = pdf_to_images(
            pdf_path=PDF_FILE,
            output_path=OUTPUT_FOLDER,
            format="png",
            scale=4.0
        )

        if converted_images:
            print(f"\nSuccessfully created {len(converted_images)} images in the {OUTPUT_FOLDER}")
        else:
            print("\nImage conversion failed")