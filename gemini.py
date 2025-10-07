import os
import cv2
import json
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Constants
API_KEY = os.getenv("GEMINI_API_KEY")
INPUT_FOLDER = "./converted_images"
OUTPUT_FOLDER = "./gemini_extracted_images"
JSON_FOLDER = "./gemini_json_results"

# Pricing (USD per 1M tokens for gemini-2.5-flash)
PRICE_INPUT = 0.075  # per 1M input tokens
PRICE_OUTPUT = 0.30  # per 1M output tokens

# Prompts
PROMPT = """
Please analyze the uploaded image and do the following:

1. Identify and draw bounding boxes around three categories:
   * Text regions (all paragraphs, headings, labels, and sentences)
   * Table regions (structured rows/columns with certification details)
   * Image regions
   Double check to make sure that the bounded region is correct
"""
OUTPUT_PROMPT = "Return just bounding box coordinates and labels: text, image, or table."


def get_gemini_client(api_key: str) -> genai.Client:
    """Initialize and return a Gemini client."""
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing in the environment.")
    return genai.Client(api_key=api_key)


def inference(client: genai.Client, image: Image.Image, prompt: str, temperature: float = 0.5):
    """Send image and prompt to Gemini model for inference, returning text and token usage."""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image],
            config=types.GenerateContentConfig(temperature=temperature)
        )

        text = response.text or ""

        # Token usage info
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0)
        output_tokens = getattr(usage, "candidates_token_count", 0)

        return text, input_tokens, output_tokens

    except Exception as e:
        print(f"Gemini API call failed: {e}")
        return "", 0, 0


def read_image(filepath: str) -> Image.Image:
    """Load an image file and return a PIL Image."""
    image = cv2.imread(filepath)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {filepath}")
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def clean_results(response_text: str) -> list[dict]:
    """Extract JSON content from Gemini model response."""
    results = response_text.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(results)
    except json.JSONDecodeError:
        print("JSON parsing failed, returning empty list.")
        return []


def draw_boxes(pil_image: Image.Image, detections: list[dict]) -> Image.Image:
    """Draw labeled bounding boxes on an image."""
    image_rgb = np.array(pil_image)
    h, w = image_rgb.shape[:2]
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    for obj in detections:
        if "box_2d" not in obj or "label" not in obj:
            continue

        y1, x1, y2, x2 = obj["box_2d"]
        label = obj["label"]

        # Normalize coordinates (assuming 1000-based normalization)
        y1, x1, y2, x2 = [coord / 1000 for coord in (y1, x1, y2, x2)]
        y1, y2 = int(y1 * h), int(y2 * h)
        x1, x2 = int(x1 * w), int(x2 * w)

        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, label, (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    image_rgb_out = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb_out)


def process_images(input_folder: str, output_folder: str, json_folder: str, prompt: str, output_prompt: str):
    """Process all images in the input folder and save annotated images + JSON results."""
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)

    client = get_gemini_client(API_KEY)

    total_input_tokens = 0
    total_output_tokens = 0

    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)

        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            print(f"Skipping non-image file: {filename}")
            continue

        print(f"Processing {filename}...")

        try:
            pil_img = read_image(filepath)
            raw_response, in_tokens, out_tokens = inference(client, pil_img, prompt + output_prompt)
            detections = clean_results(raw_response)

            total_input_tokens += in_tokens
            total_output_tokens += out_tokens

            # Save JSON results
            json_path = os.path.join(json_folder, f"{os.path.splitext(filename)[0]}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(detections, f, indent=4, ensure_ascii=False)

            # Draw and save annotated image
            output_img = draw_boxes(pil_img, detections)
            output_path = os.path.join(output_folder, filename)
            output_img.save(output_path)

            print(f"Saved annotated image to {output_path}")
            print(f"Saved JSON results to {json_path}")
            print(f"Tokens — Input: {in_tokens}, Output: {out_tokens}\n")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    # Compute estimated cost
    cost_input = (total_input_tokens / 1_000_000) * PRICE_INPUT
    cost_output = (total_output_tokens / 1_000_000) * PRICE_OUTPUT
    total_cost = cost_input + cost_output

    print("\n📈 --- Usage Summary ---")
    print(f"Total input tokens: {total_input_tokens:,}")
    print(f"Total output tokens: {total_output_tokens:,}")
    print(f"Estimated input cost:  ${cost_input:.6f}")
    print(f"Estimated output cost: ${cost_output:.6f}")
    print(f"Estimated total cost: ${total_cost:.6f}")

if __name__ == "__main__":
    print("🚀 Starting Gemini image analysis pipeline...")
    process_images(INPUT_FOLDER, OUTPUT_FOLDER, JSON_FOLDER, PROMPT, OUTPUT_PROMPT)
    print("🎉 Processing complete!")
