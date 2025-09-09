

# Brain Tumor Detection using YOLOv8

This project demonstrates **brain tumor classification and detection** using the **YOLOv8 model**.
It uses a labeled dataset of brain MRI scans (glioma, meningioma, pituitary tumor, and no tumor) to train and test a detection pipeline.

##  Setup


### 1. Mount Google Drive & Extract Dataset

```python
from google.colab import drive
drive.mount('/content/drive')

import zipfile
zip_path = '/content/drive/MyDrive/Braintumor classification/archive.zip'
extract_path = '/content/drive/MyDrive/Braintumor classification'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
```

### 2. Install Dependencies

```bash
pip install ultralytics Augmentor
```

##  Dataset

* **Classes:**

  * `glioma` → 0
  * `meningioma` → 1
  * `notumor` → 2
  * `pituitary` → 3

Each image has a corresponding YOLO-format `.txt` annotation file with bounding boxes.

##  Data Preprocessing

* Dataset is split into **train** (80%) and **test** (20%) using `custom_train_test_split()`.
* Visualization of bounding boxes is included to confirm dataset correctness.

##  Training YOLOv8

To train from scratch:

```python
from ultralytics import YOLO

yolo_btd_model = YOLO("yolov8n.yaml")
yolo_btd_model.train(
    data="/content/drive/MyDrive/Braintumor classification/brain_tumor_dataset.yaml",
    epochs=25
)
```

##  Evaluation

After training, the best weights are saved as **`best.pt`**.

Run inference:

```python
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

model = YOLO("/content/drive/MyDrive/Braintumor classification/best.pt")

image_paths = [
    "/content/.../glioma/Tr-gl_0022.jpg",
    "/content/.../meningioma/Tr-me_0010.jpg",
    "/content/.../notumor/Tr-no_0010.jpg",
    "/content/.../pituitary/Tr-pi_0012.jpg"
]

results_list = [model(path) for path in image_paths]

for i, results in enumerate(results_list):
    res_img = results[0].plot()
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Result {i+1}")
    plt.axis("off")
    plt.show()
```

##  Output Example

The model overlays bounding boxes and class labels on MRI images:

* Glioma tumor detected
* Meningioma tumor detected
* Pituitary tumor detected
* No tumor case

## 🛠 Requirements

* Python 3.8+
* Google Colab / Jupyter Notebook
* Libraries: `ultralytics`, `matplotlib`, `opencv-python`, `sklearn`, `Augmentor`, `PIL`
