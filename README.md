<div align="center">

# 🩻 RadioAge

### AI-Powered Pediatric Bone Age Assessment Tool

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In_Development-orange.svg)]()

**Predict bone age from hand X-rays in 15 seconds — with confidence intervals, visual explanations, and clinical disorder flags.**

*An open-source alternative to the ₹2,400/test TW3 method — accessible from any device, anywhere.*

[Try Demo](#demo) · [API Docs](#api-reference) · [How It Works](#-architecture) · [Results](#-results)

</div>

---

## 🇮🇳 The Problem — Why India Needs This

### For Hospitals & Clinics

India has approximately **27 crore children** (0-14 years) but only around **2,000 pediatric radiologists** — that's roughly **1 radiologist per 135,000 children.** In rural India, the number is effectively **zero**.

When a child shows signs of a growth disorder, the standard diagnostic step is **bone age assessment** — an X-ray of the left hand compared against a reference atlas to determine skeletal maturity. This comparison is:

- **Slow** — each assessment takes **10-15 minutes** of a trained radiologist's time, using the **Greulich-Pyle Atlas** (published in 1959, based on 1930s American children)
- **Subjective** — two radiologists reading the same X-ray disagree by **±6-12 months** ([published inter-observer variability](https://pmc.ncbi.nlm.nih.gov/articles/PMC4266871/))
- **Inaccessible** — rural Primary Health Centers have X-ray machines but **no radiologist to read them**. X-rays are sent to district hospitals, with **3-7 day wait times** for results
- **Biased** — the GP Atlas was developed on white American children from the 1930s-40s. Indian children's skeletal maturation differs by an average of **3-6 months** ([Indian Journal of Pediatrics](https://pmc.ncbi.nlm.nih.gov/articles/PMC8547392/))

The consequence of a wrong bone age reading is real: **Growth hormone therapy costs ₹2-5 lakh/year**, and the decision to start, continue, or stop this treatment depends directly on bone age assessment.

### For Sports — The Age Fraud Crisis

Age fraud is one of Indian sports' biggest challenges. **BCCI, AIFF, and SAI** all use bone age testing (TW3 method) to verify player ages in junior tournaments:

- **BCCI** tests ~900 cricketers annually for under-16 tournaments (Vijay Merchant Trophy) — with only **4 radiologists** serving **38 state associations**, the process takes **2 full months**
- **AIFF** has tested ~3,000 footballers using TW3, revealing widespread age manipulation
- Current TW3 test costs: **₹2,400 per test** and takes **3-4 days** for results

### For Legal & Forensic Purposes

Indian courts rely on bone age evidence in:
- **POCSO Act cases** — determining whether a victim/accused is a minor
- **Child labor cases** — verifying age when birth certificates are unavailable
- **Juvenile Justice Act** — deciding whether to try as adult or juvenile
- **Adoption cases** — estimating age of abandoned children without documentation

**In all these scenarios, the current process is slow, expensive, and inaccessible.**

---

## 💡 Our Solution

RadioAge is a **free, always-available AI tool** that automates bone age assessment:

```
Doctor/user uploads hand X-ray → 15 seconds → Complete assessment report
```

| Feature | What It Does | Why It Matters |
|---|---|---|
| **Bone Age Prediction** | EfficientNet-B3 predicts skeletal age from hand X-ray | Replaces 15-min manual atlas comparison |
| **Uncertainty Quantification** | Monte Carlo Dropout gives confidence band (e.g., "132 ± 8 months") | Doctor knows WHEN to trust the AI and when to double-check manually |
| **Visual Explanation (GradCAM)** | Heatmap overlay showing WHICH bones the model focused on | Doctor can verify the AI is looking at clinically relevant areas (carpals, epiphyseal plates) |
| **Clinical Disorder Flagging** | Compares predicted vs actual age, flags significant deviations | Automatic alert: "⚠️ Advanced bone age — evaluate for precocious puberty" |
| **PDF Report** | Professional report with images, predictions, flags, disclaimer | Printable for patient files, court submissions, or referral letters |
| **REST API** | Programmatic access via POST /predict | Integrable into hospital EMR systems, sports federation workflows |

### RadioAge vs Current Methods

| Method | Time | Cost | Uncertainty | Explainable | Available 24/7 | Open Source |
|---|---|---|---|---|---|---|
| **GP Atlas (Manual)** | 10-15 min | Radiologist's time | ❌ | ❌ | ❌ (needs radiologist) | ❌ |
| **TW3 Method (Manual)** | 7-8 min | ₹2,400/test | ❌ | ❌ | ❌ (4 radiologists for 38 states) | ❌ |
| **BoneXpert (Commercial)** | Instant | ₹288/test (BCCI rate) | ❌ | Limited | Paid license | ❌ |
| **RadioAge (Ours)** | **15 sec** | **₹0** | **✅ MC Dropout** | **✅ GradCAM** | **✅** | **✅** |

---

## 🏗️ Architecture

```
User uploads hand X-ray (PNG/JPG)
        │
        ▼
┌─── RadioAge Server ────────────────────────────────┐
│                                                     │
│   📥 Preprocessing                                  │
│   ├── Resize to 300×300                             │
│   ├── CLAHE contrast enhancement                    │
│   └── ImageNet normalization                        │
│                                                     │
│   🧠 EfficientNet-B3 (pre-trained, fine-tuned)     │
│   ├── Backbone: 1536-dim feature extraction         │
│   ├── Gender conditioning: concat as input feature  │
│   └── Regression head: Dense(1537→512→256→1)        │
│                                                     │
│   📊 Monte Carlo Dropout (20 forward passes)        │
│   ├── Mean prediction → bone age                    │
│   └── Standard deviation → uncertainty band         │
│                                                     │
│   🔍 GradCAM Heatmap                               │
│   └── Visual explanation overlay on X-ray           │
│                                                     │
│   🏥 Clinical Rules Engine                          │
│   ├── Compare predicted vs chronological age        │
│   ├── 3-tier severity: Normal / Monitor / Flag      │
│   └── Condition suggestions + specialist referral   │
│                                                     │
│   📄 PDF Report Generator                           │
│   └── Professional clinical report with all outputs │
│                                                     │
└─────────────────────────────────────────────────────┘
        │
        ▼
Response: age ± uncertainty, heatmap, flags, PDF
```

---

## 📊 Results

> *Results will be updated after training completion on the RSNA dataset.*

| Metric | Target | Status |
|---|---|---|
| **MAE** (Mean Absolute Error) | < 8 months | Training in progress |
| **Within ±12 months** | > 90% | Training in progress |
| **Inference time** (CPU, with uncertainty) | < 15 sec | ✅ Confirmed |
| **Model size** | ~47 MB | ✅ Confirmed |

**Benchmark context:** Human radiologist MAE on RSNA dataset = **7.3 months**. Our target is to match or beat human performance.

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch, torchvision |
| Model | EfficientNet-B3 (12M params, transfer learning from ImageNet) |
| Uncertainty | Monte Carlo Dropout — Bayesian approximation ([Gal & Ghahramani, 2016](https://arxiv.org/abs/1506.02142)) |
| Explainability | GradCAM — Gradient-weighted Class Activation Mapping ([Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)) |
| Image Processing | OpenCV (CLAHE), Pillow |
| API | FastAPI, Pydantic, Uvicorn |
| Frontend | Gradio |
| Reports | ReportLab (PDF generation) |
| Deployment | Docker, DigitalOcean |
| Training | Kaggle P100 GPU (free tier) |

---

## 📂 Project Structure

```
RadioAge/
├── src/
│   ├── data_loader.py           # Dataset loading + inspection
│   ├── preprocessing.py         # Image transforms, CLAHE, DataLoader
│   ├── model.py                 # EfficientNet-B3 + gender-conditioned head
│   ├── train.py                 # Training loop with LR scheduling
│   ├── uncertainty.py           # Monte Carlo Dropout inference
│   ├── inference.py             # End-to-end prediction pipeline
│   ├── gradcam_module.py        # GradCAM heatmap generation + overlay
│   ├── clinical_rules.py        # Disorder flagging engine
│   ├── report_generator.py      # PDF clinical report
│   └── evaluate.py              # Metrics + visualization
├── api/
│   ├── main.py                  # FastAPI endpoints
│   └── schemas.py               # Request/response models
├── app/
│   └── gradio_app.py            # Web interface
├── models/                      # Saved model weights (.pth)
├── data/                        # Dataset (gitignored)
├── images/                      # EDA plots, GradCAM outputs
├── notebooks/
│   └── 01_EDA.ipynb             # Exploratory data analysis
├── Dockerfile
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/4yushman/RadioAge.git
cd RadioAge
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### Download Dataset

```bash
# Kaggle CLI
pip install kaggle
kaggle datasets download -d kmader/rsna-bone-age -p data/

# Or download manually from:
# https://www.kaggle.com/datasets/kmader/rsna-bone-age
```

### Quick Inference

```python
from src.inference import predict

result = predict(
    image_path="path/to/hand_xray.png",
    gender="male",
    chronological_age_months=108
)

print(f"Predicted age: {result['age']} ± {result['uncertainty']} months")
print(f"Clinical flag: {result['flag']}")
```

### Start API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Docs: http://localhost:8000/docs
```

### Launch Web Interface

```bash
python app/gradio_app.py
# Opens at http://localhost:7860
```

### Docker

```bash
docker build -t radioage .
docker run -p 8000:8000 radioage
```

---

<a name="api-reference"></a>
## 🔌 API Reference

### `POST /predict`

```bash
curl -X POST https://radioage.me/predict \
  -F "image=@hand_xray.png" \
  -F "gender=male" \
  -F "chronological_age=108"
```

```json
{
  "predicted_age_months": 132.5,
  "predicted_age_years": 11.0,
  "uncertainty_months": 8.2,
  "confidence_range": {"low": 124.3, "high": 140.7},
  "disorder_flag": {
    "status": "ADVANCED",
    "severity": "HIGH",
    "flag": "🔴 Significantly advanced bone age",
    "possible_conditions": ["Precocious puberty", "Hyperthyroidism"],
    "recommendation": "Refer to pediatric endocrinologist"
  },
  "gradcam_image": "base64...",
  "report_pdf": "base64..."
}
```

### `GET /health`

```json
{"status": "ok", "model": "EfficientNet-B3", "version": "1.0.0"}
```

---

## 📊 Dataset

[**RSNA Pediatric Bone Age Challenge (2017)**](https://www.kaggle.com/datasets/kmader/rsna-bone-age)

| Attribute | Value |
|---|---|
| Total images | 14,236 hand X-rays |
| Training / Validation / Test | 12,611 / 1,425 / 200 |
| Image format | PNG |
| Labels | Bone age (months) + Gender |
| Age range | 1 – 228 months (0 – 19 years) |
| Institutions | Stanford, University of Colorado, UCLA |
| License | Academic research + education |

---

## ⚠️ Known Limitations

- **Population bias** — trained on primarily Western institution data. Indian children may show 3-6 months average deviation in skeletal maturity. Validation on Indian populations is needed before any clinical consideration.
- **Age extremes** — fewer samples for ages <2 and >17 years. Model accuracy decreases at these extremes.
- **Not TW3-equivalent** — our model learns patterns from data (GP-style holistic approach), not the specific 13-bone RUS scoring of TW3. Results may differ from formal TW3 assessment.
- **Not a medical device** — has not undergone FDA/CDSCO regulatory clearance.

## 🔮 Future Work

- [ ] Fine-tuning on Indian pediatric X-ray datasets (pending institutional collaboration with AIIMS/PGI)
- [ ] Indian population calibration layer using IAP-WHO growth references
- [ ] TW3-style RUS scoring module (individual bone maturity stages)
- [ ] Multi-ethnic validation study
- [ ] Bone fracture detection module (additional feature tab)
- [ ] Mobile-optimized inference (ONNX export)
- [ ] Integration with DICOM-compatible PACS systems

---

## ⚠️ Disclaimer

> **RadioAge is a research prototype developed for educational purposes as part of an M.Sc. AI & ML program at Jamia Millia Islamia, New Delhi. It is NOT a medical device and is NOT intended for clinical diagnosis, treatment planning, or any medical decision-making. It is NOT a replacement for the TW3 method or professional radiological assessment. Always consult a board-certified radiologist for bone age evaluation. The developers assume no liability for any clinical, legal, or sporting decisions made based on this tool's outputs.**

---

## 👥 Built Collaboratively

This project was built as a collaborative effort, with work split across model development and deployment:

| | Name | Responsibilities | GitHub |
|---|---|---|---|
| 🧑‍💻 | **Ayushman Maurya** | Uncertainty quantification, inference pipeline, FastAPI, Docker, deployment | [@4yushman](https://github.com/4yushman) |
| 🧑‍💻 | **Faraz** | Data pipeline, model training, GradCAM, clinical rules, PDF reports, evaluation | [@faraz-username](https://github.com/) |

All work was done through **feature branches, pull requests, and code reviews** — following industry-standard collaborative Git workflow.

**M.Sc. AI & ML** — Department of Computer Science, Jamia Millia Islamia, New Delhi

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- **RSNA** — Pediatric Bone Age Challenge dataset
- **Stanford, UC, UCLA** — contributing X-ray images
- **Google Brain** — EfficientNet architecture ([Tan & Le, 2019](https://arxiv.org/abs/1905.11946))

## 📚 References

1. Halabi SS et al. *The RSNA Pediatric Bone Age Machine Learning Challenge.* Radiology, 2018.
2. Tan M, Le QV. *EfficientNet: Rethinking Model Scaling for CNNs.* ICML, 2019.
3. Gal Y, Ghahramani Z. *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.* ICML, 2016.
4. Selvaraju RR et al. *Grad-CAM: Visual Explanations from Deep Networks.* ICCV, 2017.
5. Greulich WW, Pyle SI. *Radiographic Atlas of Skeletal Development of the Hand and Wrist.* Stanford University Press, 1959.
6. Tanner JM et al. *Assessment of Skeletal Maturity and Prediction of Adult Height (TW3 Method).* Saunders, 2001.
7. Khadilkar VV et al. *Indian Growth References from 0-18 Year Old Children.* Indian J Endocrinol Metab, 2019.

---

<div align="center">

**Built with ❤️ for Indian healthcare, sports integrity, and child welfare**

⭐ Star this repo if you find it useful

[⬆ Back to Top](#-radioage)

</div>
