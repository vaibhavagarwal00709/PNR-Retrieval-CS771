from pathlib import Path
import pickle
import re
import sys

import cv2


base_path = Path(__file__).resolve().parent

# Load our latest model
with open(base_path / "arithmetic_model_auto.sav", "rb") as file:
    model = pickle.load(file)


# Same segmentation function used during training
def segment(image, st):
    flag = 0
    flag2 = 0
    start = 0
    end = 0

    for i in range(st, 149):
        for j in range(30):
            if image[j, i] != 0:
                flag = 1

                if start == 0:
                    start = i - 2

                flag2 = 1

        if flag == 0 and flag2 == 1:
            end = i + 2
            break

        flag = 0

    return image[:, start:end], end


# Optional image path supplied through the terminal
image_path = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else base_path / "predict" / "0.png"
)

image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

if image is None:
    raise ValueError(f"Could not open: {image_path}")

if image.ndim != 3 or image.shape[2] != 4:
    raise ValueError("Expected a four-channel BGRA CAPTCHA image.")

# Same preprocessing as training
trans_mask = image[:, :, 3] == 0
image[trans_mask] = [255, 255, 255, 255]

image = cv2.bitwise_not(image)
image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
image = cv2.resize(image, (150, 30))

characters = "0123456789+-=?"
expression = ""
end = 0
found_equals = False

for _ in range(6):
    previous_end = end
    crop, end = segment(image, end)

    if crop.size == 0 or end <= previous_end:
        raise ValueError("Character segmentation failed.")

    crop = cv2.resize(crop, (25, 25))

    # Raw pixels, matching our training code
    features = crop.reshape(1, 625)

    label = int(model.predict(features)[0])
    character = characters[label]

    probabilities = model.predict_proba(features)[0]
    probability = probabilities[list(model.classes_).index(label)]

    print(
        f"Recognized: {character} | "
        f"Model probability: {probability:.2%}"
    )

    expression += character

    if character == "=":
        found_equals = True
        break

print("\nImage:", image_path)
print("Recognized expression:", expression)

match = re.fullmatch(r"(\d{1,2})([+-])(\d{1,2})=", expression)

if found_equals and match:
    first = int(match.group(1))
    operator = match.group(2)
    second = int(match.group(3))

    answer = first + second if operator == "+" else first - second

    print("Calculated answer:", answer)
else:
    print("Could not recognize a valid complete expression.")