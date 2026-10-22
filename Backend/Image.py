import csv
import re
import time
from pathlib import Path

import cv2
import pytesseract
import requests

# Change this if Tesseract is installed elsewhere
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

base_path = Path(__file__).resolve().parent
folder = base_path / "dataset_auto_simple"
images_folder = folder / "images"
labels_file = folder / "labels.csv"

if labels_file.exists():
    print("Continuing the existing collection.")

images_folder.mkdir(parents=True, exist_ok=True)

url = "https://www.indianrail.gov.in/enquiry/captchaDraw.png"

with open(labels_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "text"])

    for i in range(1500):
        try:
            # 1. Download the image
            response = requests.get(url, timeout=20)
            response.raise_for_status()

            filename = f"{i}.png"
            image_path = images_folder / filename
            image_path.write_bytes(response.content)

            # 2. Open the downloaded image
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

            if image is None:
                raise ValueError("Downloaded file is not an image.")


            # 3. Prepare the image for OCR
            image[image[:, :, 3] == 0] = [255, 255, 255, 255]
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            image = cv2.resize(image, None, fx=4, fy=4)
            image = cv2.copyMakeBorder(
                image, 20, 20, 20, 20,
                cv2.BORDER_CONSTANT, value=255
            )


            text = pytesseract.image_to_string(
                image,
                config="--psm 7 -c tessedit_char_whitelist=0123456789+-=?"
            )
            text = "".join(text.split())

            if re.fullmatch(r"\d{1,2}[+-]\d{1,2}=\?", text):
                writer.writerow([filename, text])
                file.flush()
                print(filename, text)
            else:
                print(filename, "needs manual labeling:", repr(text))

        except Exception as error:
            print("Stopped:", error)
            break

        finally:
            time.sleep(2)



# import csv
# import re
# import time
# from pathlib import Path

# import cv2
# import pytesseract
# import requests


# # Use the Tesseract path that worked on your computer
# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )

# base_path = Path(__file__).resolve().parent
# folder = base_path / "dataset_auto_simple"
# images_folder = folder / "images"
# labels_file = folder / "labels.csv"

# images_folder.mkdir(parents=True, exist_ok=True)

# url = "https://www.indianrail.gov.in/enquiry/captchaDraw.png"

# # Continue after the highest existing image number
# existing_numbers = [
#     int(path.stem)
#     for path in images_folder.glob("*.png")
#     if path.stem.isdigit()
# ]

# start_index = max(existing_numbers, default=-1) + 1

# write_header = (
#     not labels_file.exists()
#     or labels_file.stat().st_size == 0
# )

# print(f"Starting from image number: {start_index}")

# # Append to the CSV instead of replacing existing labels
# with open(
#     labels_file,
#     "a",
#     newline="",
#     encoding="utf-8"
# ) as file:
#     writer = csv.writer(file)

#     if write_header:
#         writer.writerow(["filename", "text"])

#     for i in range(start_index, 1500):
#         try:
#             # 1. Download the image
#             response = requests.get(url, timeout=20)
#             response.raise_for_status()

#             filename = f"{i}.png"
#             image_path = images_folder / filename

#             # Check downloaded bytes before saving
#             import numpy as np

#             image = cv2.imdecode(
#                 np.frombuffer(response.content, dtype=np.uint8),
#                 cv2.IMREAD_UNCHANGED
#             )

#             if image is None:
#                 raise ValueError("Downloaded response is not an image.")

#             if image.ndim != 3 or image.shape[2] != 4:
#                 raise ValueError("Expected a four-channel CAPTCHA image.")

#             image_path.write_bytes(response.content)

#             # 2. Replace transparency with white
#             image[image[:, :, 3] == 0] = [255, 255, 255, 255]

#             # 3. Convert to grayscale and enlarge
#             image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
#             image = cv2.resize(image, None, fx=4, fy=4)

#             # 4. Add white padding
#             image = cv2.copyMakeBorder(
#                 image,
#                 20, 20, 20, 20,
#                 cv2.BORDER_CONSTANT,
#                 value=255
#             )

#             # 5. Read the expression using OCR
#             text = pytesseract.image_to_string(
#                 image,
#                 config=(
#                     "--psm 7 "
#                     "-c tessedit_char_whitelist=0123456789+-=?"
#                 ),
#                 timeout=15
#             )

#             text = "".join(text.split())

#             # 6. Save provisionally valid image-label pairs
#             if re.fullmatch(r"\d{1,2}[+-]\d{1,2}=\?", text):
#                 writer.writerow([filename, text])
#                 file.flush()
#                 print(filename, text)

#             else:
#                 print(
#                     filename,
#                     "needs manual labeling:",
#                     repr(text)
#                 )

#         except requests.exceptions.Timeout:
#             print(f"{i}: download timed out; skipping this attempt.")

#         except requests.exceptions.HTTPError as error:
#             print(f"Server error: {error}")
#             print("Stopping. Existing progress is saved.")
#             break

#         except requests.exceptions.RequestException as error:
#             print(f"Connection error: {error}")
#             print("Stopping. Rerun later to resume.")
#             break

#         except Exception as error:
#             print(f"Stopped at attempt {i}: {error}")
#             break

#         finally:
#             time.sleep(2)

# print("Run finished. Existing images and labels are saved.")