# Animal Species Classification (Animals-10)

A TensorFlow/Keras project comparing **ZFNet**, **VGG16** (transfer learning), and **GoogLeNet/Inception** (transfer learning) to classify images from the **Animals-10** dataset. The dataset has ~28,000 labeled images across **10 classes**. :contentReference[oaicite:2]{index=2}

## Results (from the project report)
| Model        | Train Acc | Val Acc | Val Loss |
|--------------|-----------:|--------:|---------:|
| ZFNet        | 0.84       | 0.74    | 0.94     |
| VGG16        | 0.95       | 0.88    | 0.39     |
| GoogLeNet    | 0.98       | 0.96    | 0.1331   |

Numbers above are summarized from `FINAL.SSA_REPORT.pdf`. :contentReference[oaicite:3]{index=3}

## Repository Contents
- `PROJECT_1(ANIMAL_SPECIES)_GROUP_5_.ipynb` — full training notebook (data download, preprocessing, training, evaluation, prediction)
- `FINAL.SSA_REPORT.pdf` — report with methods, experiments, and analysis
- `assets/` — (optional) plots/screenshots
- `.gitignore` — ignores big/temporary files
- `README.md` — this file

## How to Run (Google Colab)
1. Open the notebook in Colab (File ➜ Open in Colab).
2. Set up Kaggle API in Colab to download Animals-10, then unzip:
```python
import os, shutil, zipfile, subprocess, json
# upload kaggle.json or use your Kaggle API token
# then:
!mkdir -p /root/.kaggle
!cp kaggle.json /root/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets download -d alessiocorrado99/animals10
!unzip -q animals10.zip -d /content/extracted_data
