🐾 Animal Species Detection (Animals-10 Dataset)

This project applies deep learning models to classify animal species from images.
We implemented and compared three architectures:

ZFNet (custom CNN)

VGG16 (transfer learning)

GoogLeNet / InceptionV3 (transfer learning)

The dataset used is Animals-10
, which contains ~28,000 labeled images across 10 animal classes.

📂 Repository Structure
animal-species-detection/
│── PROJECT_1(ANIMAL_SPECIES)GROUP_5.ipynb   # Jupyter notebook (full workflow)
│── FINAL.SSA_REPORT.pdf                     # Final academic report
│── src/                                     # Clean Python scripts
│   ├── train_models.py                      # Model training
│   ├── predict.py                           # Prediction pipeline
│   └── preprocess.py                        # Preprocessing & dataset handling
│── dataset/
│   └── download_dataset.py                  # Script to download Animals-10 dataset
│── requirements.txt                         # Project dependencies
│── README.md                                # Main documentation
│── .gitignore                               # Ignore unnecessary files

📊 Results (from the report)
Model	Train Acc	Val Acc	Val Loss
ZFNet	0.84	0.74	0.94
VGG16	0.95	0.88	0.39
GoogLeNet	0.98	0.96	0.1331

👉 GoogLeNet achieved the best performance with ~96% validation accuracy.
Details and analysis are provided in FINAL.SSA_REPORT.pdf
.

⚙️ Installation

Clone the repo:

git clone https://github.com/<your-username>/animal-species-detection.git
cd animal-species-detection


Install dependencies:

pip install -r requirements.txt

📥 Dataset

This project uses the Animals-10 dataset from Kaggle.

Install Kaggle CLI:

pip install kaggle


Place your kaggle.json API key in ~/.kaggle/.

Run the dataset download script:

python dataset/download_dataset.py


➡️ Images will be extracted into: dataset/raw-img/

🚀 Usage
1. Train Models

Example: train GoogLeNet

python src/train_models.py --model googlenet


This will:

Load and preprocess the dataset (preprocess.py)

Train the selected model

Save the trained weights to models/

2. Predict New Images
python src/predict.py --img_path path/to/test_image.jpg


Output example:

Predicted Species: cat (98.3% confidence)

📒 Notebook

PROJECT_1(ANIMAL_SPECIES)GROUP_5.ipynb:
Full end-to-end workflow with preprocessing, training, evaluation, and predictions.
Recommended for experimentation and visualization.

📑 Report

FINAL.SSA_REPORT.pdf:
Academic-style report with methodology, experiments, results, and conclusions.

🙏 Acknowledgments

Dataset: Animals-10 by Alessio Corrado (Kaggle)

Models: Keras/TensorFlow implementations of ZFNet, VGG16, and InceptionV3
