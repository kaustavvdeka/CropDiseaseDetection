# 🌿 CropGuard NE — Plant Disease Detection for Northeast India

AI-powered plant disease detection using MobileNetV2 fine-tuned on PlantVillage dataset (~87,000 images, 38 classes).

---

## 📁 Project Structure

```
cropguard_ne/
├── model/
│   ├── train.py          ← Full training script (Phase 1 + Phase 2 fine-tuning)
│   └── predictor.py      ← Inference engine with NE India disease metadata
├── app/
│   └── app.py            ← Streamlit 3-page application
├── notebooks/
│   └── CropGuard_NE_Training.ipynb  ← Google Colab GPU notebook
├── models/               ← Created after training
│   ├── plantdisease_mobilenetv2.h5
│   ├── class_indices.json
│   └── training_history.json
├── plots/                ← Created after training
│   ├── training_curves.png
│   └── confusion_matrix.png
├── data/                 ← Created after download
│   └── plantvillage dataset/color/  ← 38 class folders
├── setup.py              ← One-click setup
└── requirements.txt
```

---

## 🚀 Quick Start

### Option A — Google Colab (Recommended, free GPU)

1. Open `notebooks/CropGuard_NE_Training.ipynb` in Google Colab
2. Set **Runtime → Change runtime type → T4 GPU**
3. Run all cells — takes ~25–40 minutes
4. Download `plantdisease_mobilenetv2.h5` + `class_indices.json` to `models/`
5. Run the Streamlit app locally

### Option B — Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up Kaggle API key
#    Visit https://kaggle.com/settings → Create New Token
#    Save as ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 3. One-click setup (download → train → launch app)
python setup.py

# OR step by step:
python setup.py --skip-train    # download only (then train separately)
python model/train.py           # training only
streamlit run app/app.py        # app only
```

### Option C — App without model (Claude AI fallback)

If you just want to test the app before training:

```bash
pip install streamlit anthropic pillow
export ANTHROPIC_API_KEY=your_key_here
streamlit run app/app.py --skip-train
```

The app will use Claude AI for disease analysis instead of the local model.

---

## 🧠 Model Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 backbone (ImageNet pre-trained, top 50 layers fine-tuned)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(512, ReLU) + L2(1e-4) regularization
    ↓
Dropout(0.4)
    ↓
Dense(256, ReLU) + L2(1e-4) regularization
    ↓
Dropout(0.3)
    ↓
Dense(38, Softmax)  ← output: 38 disease classes
```

**Training Strategy:**
- Phase 1 (10 epochs): Freeze MobileNetV2, train head only @ lr=1e-4
- Phase 2 (15 epochs): Unfreeze top 50 layers, fine-tune @ lr=1e-5
- EarlyStopping + ReduceLROnPlateau callbacks

**Expected Performance:**
| Metric | Value |
|--------|-------|
| Validation Accuracy | ~96% |
| Top-5 Accuracy | ~99.5% |
| Training Time (T4 GPU) | ~30 min |
| Training Time (CPU) | ~4–6 hours |

---

## 📊 Dataset

- **Source:** [PlantVillage on Kaggle](https://kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Total images:** ~87,000
- **Classes:** 38 (healthy + diseased states of 14+ crops)
- **Split used:** 80% train / 20% validation

---

## 🌿 38 Disease Classes

Potato Late Blight, Potato Early Blight, Tomato Bacterial Spot, Tomato Early Blight,
Tomato Late Blight, Tomato Leaf Mold, Tomato Septoria Leaf Spot, Tomato Target Spot,
Tomato TYLCV, Tomato Mosaic Virus, Apple Scab, Apple Black Rot, Apple Cedar Rust,
Corn Gray Leaf Spot, Corn Common Rust, Corn Northern Leaf Blight, Grape Black Rot,
Grape Esca, Grape Leaf Blight, Orange Huanglongbing, Peach Bacterial Spot,
Pepper Bacterial Spot, Squash Powdery Mildew, Strawberry Leaf Scorch,
+ all Healthy classes

---

## 🗺️ Northeast India Focus

The app provides specific context for:
- **Assam** — Tomato, Potato, Mustard
- **Meghalaya** — Potato (Late Blight is critical), Ginger, Turmeric
- **Manipur** — Tomato, Cabbage, Pea
- **Nagaland** — Potato, Ginger, Maize
- **Arunachal Pradesh** — Apple (high altitude), Maize
- **Tripura** — Tomato, Cabbage, Brinjal
- **Mizoram** — Ginger, Turmeric, Potato
- **Sikkim** — Apple, Cardamom, Maize

---

## 📞 Support

- ICAR-NRC Meghalaya: nrcmeghalaya.icar.gov.in
- KVK Assam Network: rkvy.nic.in
- NE India Agriculture Portal: nehepune.org
