import os
import json
from dotenv import load_dotenv

from agentic_doc.parse import parse
from agentic_doc.utils import viz_parsed_document
from agentic_doc.config import VisualizationConfig


# Load API key
load_dotenv()
API_KEY = os.getenv("LANDING_API_KEY")

# Directories
INPUT_FOLDER = './converted_images'
OUTPUT_DIR = './landingai_extracted_images'
DOC_DIR = './parsed_docs'

# Save the parsed JSON docs, and save visualized images
def process_images(folder, doc_dir, output_dir, viz_config=None):
    os.makedirs(doc_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for filename in os.listdir(folder):
        if not filename.lower().endswith(('.png', '.jpeg', '.jpg', '.webp')):
            continue

        image_path = os.path.join(folder, filename)
        print(f"Processing: {filename}")

        # Parse image using Landing AI API
        result = parse(image_path)
        doc = result[0]

        # --- Save parsed document as JSON ---
        json_path = os.path.join(doc_dir, f"{os.path.splitext(filename)[0]}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(doc.model_dump(), f, ensure_ascii=False, indent=2)
        #Recommended: Save the file in pickle format for regeneration
        #with open(json_path.replace(".json", ".pkl"), "wb") as f:
        #   pickle.dump(doc, f)
        

        # --- Generate visualization ---
        viz_parsed_document(
            image_path,
            doc,
            output_dir=output_dir,
            viz_config=viz_config
        )

        count += 1

    return f"{count} images processed. JSON saved in {doc_dir}, visualizations saved in {output_dir}."


# --- RUN THE PROGRAM ---
if __name__ == "__main__":
    viz_config = VisualizationConfig(text_bg_opacity=0.8)
    print(process_images(INPUT_FOLDER, DOC_DIR, OUTPUT_DIR, viz_config))
