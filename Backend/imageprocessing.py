import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


base_path = Path(__file__).resolve().parent

dataset_path = base_path / "dataset_auto_simple"
path = str(dataset_path / "images")

with open(
    dataset_path / "labels.csv",
    newline="",
    encoding="utf-8-sig"
) as file:
    rows = list(csv.DictReader(file))

imageList = [row["filename"] for row in rows]
Labels = [row["text"] for row in rows]

print("Number of labeled images:", len(imageList))

if len(imageList) < 30:
    raise ValueError("Collect more labeled images before training.")


holdout_size = round(len(imageList) * 0.05)

Xremaining, Xtest, yremaining, ytest = train_test_split(
    imageList,
    Labels,
    test_size=holdout_size,
    random_state=42
)

Xtrain, Xval, ytrain, yval = train_test_split(
    Xremaining,
    yremaining,
    test_size=holdout_size,
    random_state=42
)

print("Training images:", len(Xtrain))
print("Validation images:", len(Xval))
print("Test images:", len(Xtest))

numTolabel = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "+",
    11: "-",
    12: "=",
    13: "?",
}
labelTonum = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "+": 10,
    "-": 11,
    "=": 12,
    "?": 13,
}


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



def ImgProcessing_Seg(imageFiles, imageLabels):
    dfX = []
    dfy = []

    for imageIndex in tqdm(range(len(imageFiles))):
        image = cv2.imread(
            os.path.join(path, imageFiles[imageIndex]),
            cv2.IMREAD_UNCHANGED
        )
        if image is None:
            raise ValueError(f"Could not read {imageFiles[imageIndex]}")

        # Original preprocessing, unchanged.
        trans_mask = image[:, :, 3] == 0
        image[trans_mask] = [255, 255, 255, 255]
        image = cv2.bitwise_not(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        image = cv2.resize(image, (150, 30))
        end = 0

        for i in range(len(imageLabels[imageIndex])):
            finalImage, end = segment(image, end)
            if finalImage.size == 0:
                raise ValueError(
                    f"Empty character crop in {imageFiles[imageIndex]}, position {i}"
                )
            finalImage = cv2.resize(finalImage, (25, 25))
            finalImage = finalImage.flatten()
            dfX.append(finalImage)
            dfy.append(labelTonum[imageLabels[imageIndex][i]])

    return np.array(dfX), np.array(dfy)


dfX, dfy = ImgProcessing_Seg(Xtrain, ytrain)
valX, valy = ImgProcessing_Seg(Xval, yval)


classifier = LogisticRegression(random_state=0, max_iter=4000)
classifier.fit(dfX, dfy)

train_predictions = classifier.predict(dfX)
val_predictions = classifier.predict(valX)


print("Training character accuracy:", accuracy_score(dfy, train_predictions) * 100)
print("Validation character accuracy:", accuracy_score(valy, val_predictions) * 100)
print("Test images reserved:", len(Xtest))


filename = base_path / "arithmetic_model_auto.sav"
with open(filename, "wb") as file:
    pickle.dump(classifier, file)

print("Model saved to:", filename)


