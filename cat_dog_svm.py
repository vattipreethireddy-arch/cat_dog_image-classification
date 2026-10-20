import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = [], []
size = (64, 64)

folders = {"train/cat": 0, "train/dog": 1}

for folder, label in folders.items():
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = os.listdir(folder)[:100]

    for file in files:
        path = os.path.join(folder, file)

        try:
            with Image.open(path) as image:
                img = image.convert("RGB").resize(size)
                X.append(np.array(img).flatten())
                y.append(label)
        except (OSError, ValueError):
            continue

if len(X) == 0:
    raise ValueError("No valid images found in train/cat or train/dog.")

X = np.array(X, dtype=np.float32) / 255.0
y = np.array(y)

if len(np.unique(y)) < 2:
    raise ValueError("Add valid images to both train/cat and train/dog.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = SVC(kernel="linear")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Images loaded:", len(X))
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("Training completed successfully!")
