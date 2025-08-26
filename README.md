🐾 Animal Species Detection (Animals-10 Dataset)

A deep learning project for classifying animal species from images using three architectures:
ZFNet (custom CNN)
VGG16 (transfer learning)
GoogLeNet / InceptionV3 (transfer learning)

Dataset: Animals-10
 (~28,000 images, 10 classes)

 📂 Repository Structure
 
 animal-species-detection/
│── PROJECT_1(ANIMAL_SPECIES)GROUP_5.ipynb   # Full workflow (Jupyter Notebook)
│── FINAL.SSA_REPORT.pdf                     # Final report
│── src/                                     # Python scripts
│   ├── train_models.py                      # Model training
│   ├── predict.py                           # Prediction pipeline
│   └── preprocess.py                        # Preprocessing & dataset handling
│── dataset/
│   └── download_dataset.py                  # Dataset download script
│── requirements.txt                         # Dependencies
│── README.md                                # Documentation
│── .gitignore

📊 Results

| Model     | Train Acc | Val Acc | Val Loss |
| --------- | --------: | ------: | -------: |
| ZFNet     |      0.84 |    0.74 |     0.94 |
| VGG16     |      0.95 |    0.88 |     0.39 |
| GoogLeNet |      0.98 |    0.96 |   0.1331 |

GoogLeNet achieved ~96% validation accuracy.

⚙️ Installation & Setup

git clone https://github.com/<your-username>/animal-species-detection.git
cd animal-species-detection
pip install -r requirements.txt

Download the dataset:

pip install kaggle
python dataset/download_dataset.py

🚀 Usage
Train a Model
python src/train_models.py --model googlenet

Predict on a New Image
python src/predict.py --img_path path/to/image.jpg

📜 Notes
PROJECT_1(ANIMAL_SPECIES)GROUP_5.ipynb: End-to-end workflow
FINAL.SSA_REPORT.pdf: Methodology, experiments, and results
