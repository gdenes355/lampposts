import os
import csv
import cv2
import rasterio
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Schema asks for label_text ONLY for candidate lamp posts
class CandidateLampPost(BaseModel):
    ymin: int = Field(description="Minimum y coordinate (normalized 0-1000)")
    xmin: int = Field(description="Minimum x coordinate (normalized 0-1000)")
    ymax: int = Field(description="Maximum y coordinate (normalized 0-1000)")
    xmax: int = Field(description="Maximum x coordinate (normalized 0-1000)")
    label_text: str = Field(description="The exact text letters read inside this label, e.g. 'L.', 'L.P'")

class LampPostResponse(BaseModel):
    lamp_posts: list[CandidateLampPost] = Field(description="List of candidate lamp post detections")

def main():
    with open("tiles.txt", "r") as f:
        tiles_dir = f.read().strip()

    os.makedirs("out", exist_ok=True)

    # Original targeted prompt
    prompt = (
        "Here is a geotiff tile of a 1880s town plan map. "
        "Search ONLY for potential lamp posts labeled 'L', 'L.' or 'L.P'. "
        "Do not output general map text, numbers, or non-lamp post markings. "
        "For each candidate found, return its bounding box and transcribe the exact text read."
    )

    master_csv_path = os.path.join("out", "all_lamp_posts_master.csv")
    master_exists = os.path.exists(master_csv_path)
    
    with open(master_csv_path, "a", newline="") as master_csv_file:
        master_writer = csv.writer(master_csv_file)
        if not master_exists:
            master_writer.writerow(["file_name", "detected_text", "pixel_x", "pixel_y", "bng_easting", "bng_northing"])

        for filename in os.listdir(tiles_dir):
            if not filename.lower().endswith(('.tif', '.tiff')):
                continue

            file_path = os.path.join(tiles_dir, filename)
            
            try:
                pil_img = Image.open(file_path)
                img_w, img_h = pil_img.size
            except Exception as e:
                print(f"Error opening {filename}: {e}")
                continue

            cv_img = cv2.imread(file_path)
            if cv_img is None:
                continue

            print(f"Processing tile with Gemini 3.6 Flash: {filename}...")

            try:
                with rasterio.open(file_path) as dataset:
                    transform = dataset.transform
            except Exception as e:
                print(f"Warning: Could not read geotransform from {filename}: {e}")
                transform = None

            detected_points = []

            try:
                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=[pil_img, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=LampPostResponse,
                        thinking_config=types.ThinkingConfig(
                            thinking_level="HIGH"
                        )
                    )
                )

                # Print exact token usage breakdown for this tile
                if response.usage_metadata:
                    in_tok = response.usage_metadata.prompt_token_count
                    out_tok = response.usage_metadata.candidates_token_count
                    tot_tok = response.usage_metadata.total_token_count
                    print(f"  [Tokens Used] Input: {in_tok} | Output (JSON+Thinking): {out_tok} | Total: {tot_tok}")

                if response.text:
                    result = LampPostResponse.model_validate_json(response.text)
                    for bbox in result.lamp_posts:
                        raw_label = bbox.label_text.strip()
                        
                        # Original Python Safety Check: Discard candidates that lack an 'L'
                        if "L" not in raw_label.upper():
                            print(f"  [Python Filter] Dropped false positive candidate: '{raw_label}'")
                            continue

                        px_xmin = int(bbox.xmin * img_w / 1000)
                        px_ymin = int(bbox.ymin * img_h / 1000)
                        px_xmax = int(bbox.xmax * img_w / 1000)
                        px_ymax = int(bbox.ymax * img_h / 1000)

                        center_x = (px_xmin + px_xmax) // 2
                        center_y = (px_ymin + px_ymax) // 2

                        detected_points.append((center_x, center_y, raw_label))

            except Exception as e:
                print(f"Error processing {filename}: {e}")

            # Write outputs
            base_name = os.path.splitext(filename)[0]
            tile_csv_path = os.path.join("out", f"{base_name}.csv")
            
            with open(tile_csv_path, "w", newline="") as tile_csv_file:
                tile_writer = csv.writer(tile_csv_file)
                tile_writer.writerow(["file_name", "detected_text", "pixel_x", "pixel_y", "bng_easting", "bng_northing"])

                for px_x, px_y, text_label in detected_points:
                    if transform:
                        easting, northing = rasterio.transform.xy(transform, px_y, px_x)
                        easting_val = round(easting, 2)
                        northing_val = round(northing, 2)
                    else:
                        easting_val, northing_val = None, None

                    row = [filename, text_label, px_x, px_y, easting_val, northing_val]
                    tile_writer.writerow(row)
                    master_writer.writerow(row)
                    
                    # Annotate in red locally (zero API tokens used)
                    cv2.circle(cv_img, (px_x, px_y), 25, (0, 0, 255), 3)

            master_csv_file.flush()

            out_png_path = os.path.join("out", f"{base_name}_out.png")
            cv2.imwrite(out_png_path, cv_img)
            print(f" Saved: {out_png_path} and {tile_csv_path}\n")

    print("Processing complete!")

if __name__ == "__main__":
    main()