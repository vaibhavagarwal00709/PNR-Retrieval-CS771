# import requests
# import pickle
# import cv2
# from flask import Flask, request, jsonify
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)
# s = requests.Session()
# loaded_model = pickle.load(open("models/model.sav", "rb"))


# def download_image(url, save_path):
#     try:
#         response = s.get(url, stream=True)
#         response.raise_for_status()
#         with open(save_path + ".png", "wb") as file:
#             for chunk in response.iter_content(chunk_size=8192):
#                 file.write(chunk)

#         print("Image downloaded successfully.")
#         return True

#     except requests.exceptions.RequestException as e:
#         print(f"Error occurred: {e}")
#         return False


# def segment(image, st):
#     flag = 0
#     flag2 = 0
#     start = 0
#     end = 0
#     for i in range(st, 149):
#         for j in range(30):
#             if image[j, i] != 0:
#                 flag = 1
#                 if start == 0:
#                     start = i - 2
#                 flag2 = 1
#         if flag == 0 and flag2 == 1:
#             end = i + 2
#             break
#         flag = 0
#     finalImage = image[:, start:end]
#     return finalImage, end


# def predict_number():
#     dfX = []
#     image_url = "https://www.indianrail.gov.in/enquiry/captchaDraw.png?1690016648505"
#     save_location = "predict/" + str(0)
#     download_image(image_url, save_location)
#     image = cv2.imread("predict/0.png", cv2.IMREAD_UNCHANGED)
#     trans_mask = image[:, :, 3] == 0
#     image[trans_mask] = [255, 255, 255, 255]
#     image = cv2.bitwise_not(image)
#     image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
#     image = cv2.resize(image, (150, 30))
#     end = 0
#     for i in range(6):
#         finalImage, end = segment(image, end)
#         finalImage = cv2.resize(finalImage, (25, 25))
#         dummy = []
#         finalImage = finalImage.flatten()
#         dummy.append(finalImage)
#         if loaded_model.predict(dummy) == [12]:
#             break
#         dfX.append(finalImage)
#     y_pred = loaded_model.predict(dfX)
#     plus = 0
#     dig1 = 0
#     dig = 0
#     for i in y_pred:
#         if i == 10 or i == 11:
#             if i == 10:
#                 plus = 1
#             dig1 = dig
#             dig = 0

#         else:
#             dig = dig * 10 + i
#     if plus:
#         dig1 += dig
#     else:
#         dig1 -= dig
#     return dig1


# def get_pnr_status(captcha, pnr_number):
#     base_url = f"https://www.indianrail.gov.in/enquiry/CommonCaptcha?inputCaptcha={captcha}&inputPnrNo={pnr_number}&inputPage=PNR&language=en"
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
#     }
#     response = s.get(base_url, headers=headers)
#     if response.status_code == 200:
#         return response.json()
#     else:
#         print(f"Failed to retrieve data. Status code: {response.status_code}")
#         s.close()
#         return None


# @app.route("/finpredict", methods=["GET"])
# def finpredict():
#     captcha = predict_number()  # Replace with your desired captcha value
#     pnr_number = int(
#         request.args.get("pnrnumber")
#     )  # Replace with your desired PNR number

#     json_data = get_pnr_status(captcha, pnr_number)

#     return json_data


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080, debug=True)





import requests
import pickle
import cv2

from pathlib import Path
from threading import Lock
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

base_path = Path(__file__).resolve().parent

s = requests.Session()

with open(base_path / "arithmetic_model_auto.sav", "rb") as file:
    loaded_model = pickle.load(file)

lookup_lock = Lock()


def download_image(url, save_path):
    try:
        response = s.get(url, stream=True, timeout=20)
        response.raise_for_status()

        with open(str(save_path) + ".png", "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print("Image downloaded successfully.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return False


# Original segmentation function
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

    finalImage = image[:, start:end]
    return finalImage, end


def predict_number():
    dfX = []

    image_url = (
        "https://www.indianrail.gov.in/enquiry/captchaDraw.png"
    )

    predict_folder = base_path / "predict"
    predict_folder.mkdir(exist_ok=True)

    save_location = predict_folder / "0"

    # Stop if downloading fails; don't use an old image
    if not download_image(image_url, save_location):
        raise ValueError("Could not download a fresh CAPTCHA.")

    image = cv2.imread(
        str(predict_folder / "0.png"),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise ValueError("Downloaded file is not a readable image.")

    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("Unexpected CAPTCHA image format.")

    # Same preprocessing as imageprocessing.py
    trans_mask = image[:, :, 3] == 0
    image[trans_mask] = [255, 255, 255, 255]

    image = cv2.bitwise_not(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    image = cv2.resize(image, (150, 30))

    end = 0
    found_equals = False

    for i in range(6):
        finalImage, end = segment(image, end)

        if finalImage.size == 0:
            raise ValueError("Could not segment the CAPTCHA.")

        finalImage = cv2.resize(finalImage, (25, 25))
        finalImage = finalImage.flatten()

        prediction = int(loaded_model.predict([finalImage])[0])

        if prediction == 12:  # "="
            found_equals = True
            break

        dfX.append(finalImage)

    if not found_equals or not dfX:
        raise ValueError("Could not recognize a complete expression.")

    y_pred = loaded_model.predict(dfX)

    # Original arithmetic-building logic, with validity checks
    plus = 0
    dig1 = 0
    dig = 0
    found_operator = False
    digit_count = 0

    for prediction in y_pred:
        prediction = int(prediction)

        if prediction == 10 or prediction == 11:
            if found_operator or digit_count == 0:
                raise ValueError("Invalid arithmetic expression.")

            found_operator = True
            plus = int(prediction == 10)

            dig1 = dig
            dig = 0
            digit_count = 0

        elif 0 <= prediction <= 9:
            dig = dig * 10 + prediction
            digit_count += 1

        else:
            raise ValueError("Unexpected character in expression.")

    if not found_operator or digit_count == 0:
        raise ValueError("Incomplete arithmetic expression.")

    if plus:
        dig1 += dig
    else:
        dig1 -= dig

    return int(dig1)


def get_pnr_status(captcha, pnr_number):
    url = "https://www.indianrail.gov.in/enquiry/CommonCaptcha"

    params = {
        "inputCaptcha": captcha,
        "inputPnrNo": pnr_number,
        "inputPage": "PNR",
        "language": "en",
    }

    headers = {
        # "User-Agent": "Mozilla/5.0"
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    response = s.get(
        url,
        params=params,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()
    return response.json()

def format_pnr_status(data):
    if not isinstance(data, dict):
        raise ValueError("Unexpected Railway response format.")

    # Preserve Railway's error message instead of showing empty details
    if data.get("errorMessage"):
        return {"error": data["errorMessage"]}

    passengers = []

    for index, passenger in enumerate(data.get("passengerList") or []):
        passengers.append({
            "passenger": index + 1,
            "booking_status": passenger.get("bookingStatusDetails"),
            "current_status": passenger.get("currentStatusDetails"),
            "booking_Coach_Id":passenger.get("bookingCoachId"),
        })

    return {
        "pnr": data.get("pnrNumber"),
        "train": {
            "name": data.get("trainName"),
            "number": data.get("trainNumber"),
        },
        "journey": {
            "from": data.get("boardingPoint"),
            "to": data.get("destinationStation"),
            "date": data.get("dateOfJourney"),
            "class": data.get("journeyClass"),
            "quota": data.get("quota"),
        },
        "passengers": passengers,
    }


@app.route("/finpredict", methods=["GET"])
def finpredict():
    # Keep PNR as text and validate it before contacting Railway
    pnr_number = request.args.get("pnrnumber", "").strip()

    if len(pnr_number) != 10 or not (
        pnr_number.isascii() and pnr_number.isdigit()
    ):
        return jsonify({
            "error": "Please enter a valid 10-digit PNR."
        }), 400

    try:
        with lookup_lock:
            captcha = predict_number()
            json_data = get_pnr_status(captcha, pnr_number)

        return jsonify(format_pnr_status(json_data))

    except requests.exceptions.RequestException as e:
        print(f"Railway request failed: {e}")
        return jsonify({
            "error": "Could not contact the Railway service."
        }), 502

    except ValueError as e:
        print(f"Image, prediction, or response error: {e}")
        return jsonify({
            "error": "Could not process the CAPTCHA or Railway response."
        }), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)