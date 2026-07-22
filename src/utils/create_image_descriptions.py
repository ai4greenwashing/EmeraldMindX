import os
import tempfile
import argparse
from pdf2image import convert_from_path
from dotenv import load_dotenv
from ollama import chat 
from google import genai
import re
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


GEMINI_MODEL="gemini"
OLLAMA_MODEL = "gemma3"
OLLAMA_URL   = "http://localhost:11434"

def process_images_with_ollama(image_paths, system_prompt):
    for image_path in image_paths:
        out_txt = os.path.splitext(image_path)[0] + ".txt"

        messages = [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": "",  "images": [image_path]},
        ]

        try:
            resp = chat(model=OLLAMA_MODEL, messages=messages)
            content = resp["message"]["content"].strip()

            with open(out_txt, "w") as f:
                f.write(content)

            print(f"Written output to {out_txt}")

        except Exception as e:
            print(f"Failed on {image_path}: {e}")

    for img_path in image_paths:
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        extract_id_and_descriptions(txt_path)
    # remove images
    for img_path in image_paths:
        os.remove(img_path)
        print(f"Removed {img_path}")

def convert_pdf_to_images(pdf_path, output_dir, dpi=300):
    os.makedirs(output_dir, exist_ok=True)
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]

    with tempfile.TemporaryDirectory() as temp_dir:
        images = convert_from_path(
            pdf_path,
            output_folder=temp_dir,
            fmt='jpeg',
            dpi=dpi,
            thread_count=8
        )

        image_paths = []
        for idx, img in enumerate(images, start=1):
            output_filename = f"{pdf_basename}_page_{idx}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            img.save(output_path, 'JPEG')
            print(f"Saved {output_path}")
            image_paths.append(output_path)

    return image_paths, pdf_basename

def extract_id_and_descriptions(txt_path: str):
    print(f"[DEBUG] Processing {txt_path}")
    try:
        with open(txt_path, "r") as f:
            raw = f.read()
            if not raw.strip():
                print(f"File is empty: {txt_path}")
                return
    except Exception as e:
        print(f"Could not read {txt_path}: {e}")
        return

    pattern = re.compile(
        r'"id"\s*:\s*(\d+)\s*,[^}]*?"description"\s*:\s*"([^"]+)"',
        re.DOTALL
    )

    matches = pattern.findall(raw)
    if not matches:
        print(f"No matches found in {txt_path}")
        return

    summary_lines = [f"{vid}: {desc.strip()}" for vid, desc in matches]

    out_path = txt_path.replace(".txt", ".summary.txt")
    with open(out_path, "w") as f:
        f.write("\n\n".join(summary_lines))

    print(f"Summary written to {out_path}")

def process_images_with_gemini(image_paths, system_prompt):
    for image_path in image_paths:
        output_path = os.path.splitext(image_path)[0] + ".txt"
        if os.path.exists(output_path):
            print(f"Output file already exists: {output_path}, skipping {image_path}")
            continue
        my_file = client.files.upload(file=image_path)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[my_file, system_prompt]
        )

        with open(output_path, "w") as f:
            f.write(response.text)
        print(f"Written output to {output_path}")
    for img_path in image_paths:
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        extract_id_and_descriptions(txt_path)
    # remove images
    for img_path in image_paths:
        os.remove(img_path)
        print(f"Removed {img_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert PDF pages to JPEG images.")
    parser.add_argument("--model", choices=["ollama", "gemini"], default="gemini", help="Choose the model backend.")
    parser.add_argument("--pdf_path", required=True, help="Path to the input PDF file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output JPEG images.")
    args = parser.parse_args()

    system_prompt = """You are an expert analyst of environmental, social, and governance (ESG) disclosures. You will receive ONE JPEG that represents an entire PDF page from a corporate sustainability report. The page may contain multiple charts, tables, infographics, or photos with overlaid text.

Detect every semantically distinct visual element in reading order and extract all quantitative and textual information.  Return a **single valid JSON** object with the exact schema below-no free text, comments, or Markdown fences.

{
  "visuals": [
    {
      "id": <integer>,
      "visual_type": "<chart | table | infographic | photo | other>",
      "title": "<string | null>",
      "description": "<description text>",
      "series": [
        {
          "series_name": "<string | null>",
          "x_label": "<string | null>",
          "y_label": "<string | null>",
          "unit": "<string | null>",
          "data_points": [
            { "x": "<string | number>", "y": <number> }
          ]
        }
      ] | null,
      "table": {
        "headers": ["<string>", ...] | null,
        "rows": [
          { "<header_1>": "<value>", "<header_2>": "<value>", ... }
        ] | null
      } | null,
      "caption_text": "<string | null>",
      "raw_text": "<string | null>"
    }
  ]
}

Rules:
1. Use null for any field that is missing or unreadable.
2. Preserve the key order exactly as given.
3. Provide a detailed description of the visual element and its information in the "description" field.
4. Output only the JSON object-no additional text."""
    model=args.model
    if model == "ollama":
        print("Using Ollama backend")
        process_images = process_images_with_ollama
    else:
        print("Using Gemini backend")
        process_images = process_images_with_gemini
    image_paths, _ = convert_pdf_to_images(args.pdf_path, args.output_dir) # can change to ollama backend
    print(f"Processing {len(image_paths)} images with {model}...")
    process_images(image_paths, system_prompt)

if __name__ == "__main__":
    main()