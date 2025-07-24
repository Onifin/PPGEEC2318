## 🧠 Model Details

This multi-class image classification model (v1) was developed by **Ian Antonio Fonseca de Araújo** and **João Vítor de Souza Lopes**, students at UFRN, and finalized in **July 2025**. It implements a **Convolutional Neural Network (CNN)** trained with the **Adam optimizer** (`learning rate = 0.0003`) using the **PyTorch** framework.

The model classifies images of animals into **10 categories** based on their visual features. Input images were resized to **128×128** pixels and normalized. Multiple CNN architectures were tested, with variations in depth, number of filters (`n_feature`), and padding to evaluate performance. The final model uses **four convolutional layers** and includes **dropout regularization** (`p = 0.3`).

Complete code and training history can be found in the [GitHub repository](https://github.com/Onifin/PPGEEC2318/tree/main). For support or issues, please open an issue in the repository.

---

## 🎯 Intended Use

The model was developed for **educational purposes**, specifically for the course **PPGEEC2318 - Aprendizado de Máquina** at UFRN.

**Intended users:**

* Graduate students
* Researchers exploring CNNs in image classification
* Educators demonstrating practical applications of deep learning in computer vision

---

## 📈 Factors

The dataset contains natural images of animals sourced from **Google Images**, divided into 10 classes. However, the number of images per class is **not balanced**.

To address this, **Class Weights** were used in the loss function during training, which increased the penalty for incorrect predictions in underrepresented classes. This ensures fairer treatment of all classes, especially:

* Cavalo (horse)
* Galinha (chicken)
* Aranha (spider)
* Cachorro (dog)

---

## 📊 Metrics

The model’s performance was evaluated using:

* **Accuracy:** proportion of correct predictions over all predictions
* **Confusion Matrix:** shows the correct and incorrect predictions per class
* **Loss Function:** CrossEntropyLoss with class weights applied
* (Optional metrics such as F1-score can be added in further evaluations)

---

## 📁 Evaluation Data

### **Dataset**

* Source: [Animals-10 Dataset on Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
* Total images: **26,179**
* Classes: Dog, Cat, Horse, Spider, Butterfly, Chicken, Sheep, Cow, Squirrel, Elephant
* Verified and labeled by humans

### **Split**

* **80% training:** 20,938 images
* **20% testing:** 5,241 images
* Each class was split individually to maintain proportional representation

### **Preprocessing**

* Resize to 128×128 pixels
* Normalization using torchvision transforms
* Conversion to tensor and application of data augmentation (`composer`) during training

---

## 🧪 Training Data

Below is the class distribution used for training (after 80/20 split):

| Class     | Training Samples |
| --------- | ---------------- |
| Cavalo    | 2,098            |
| Ovelha    | 1,456            |
| Elefante  | 1,156            |
| Gato      | 1,334            |
| Esquilo   | 1,489            |
| Galinha   | 2,478            |
| Aranha    | 3,856            |
| Gado      | 1,492            |
| Cachorro  | 3,890            |
| Borboleta | 1,689            |

All images were resized and normalized. No missing data or imputation was necessary. Class balancing was handled via **weighted loss function**, not oversampling.

---

## 📉 Quantitative Analyses

### 👁️ Arquiteturas Testadas

**1st Model:**

* 2 convolutional layers (padding = 0), `n_feature = 5`, output shape: 5×30×30
* Fully connected layers: 4500 → 10
* Performance: underfitting observed due to low capacity

**2nd Model:**

* 2 convolutional layers (padding = 1), `n_feature = 11`, output shape: 22×32×32
* Fully connected layers: 22,528 → 10
* Performance: improved feature extraction

**3rd Model (Final):**

* 4 convolutional layers (padding = 1), `n_feature = 32`, output shape: 256×8×8
* Fully connected layers: 16,384 → 10
* Performance: better generalization, stable training and validation loss

> Detailed graphs including training curves, confusion matrix, and validation metrics are included in the GitHub repository.

---

## 🧩 Ethical Considerations

* Dataset is **public** and collected from **Google Images**, curated by humans
* All data are used for **non-commercial and academic purposes** only
* No personally identifiable information (PII) is included in the dataset

---

## ⚠️ Caveats and Recommendations

* The dataset is **unbalanced**, though mitigated with class weights. Future versions could explore **data augmentation** or **SMOTE** techniques
* Model performance may vary on real-world images due to the synthetic nature of collection
* Deeper architectures or **transfer learning** (e.g., ResNet, VGG) could further enhance accuracy
* Evaluation could benefit from more metrics (F1, precision/recall, per-class accuracy)