## AI-Driven Generative System for Predicting Hull Biofouling and Optimising Maritime Maintenance 
An end-to-end pipeline for classifying marine biofouling on ship hulls using a CNN classifier, a rule-based maintenance optimizer, and a Stable Diffusion img2img generator.

---

## Project Structure

```
classification_dataset/
├── classification_dataset/
│   ├── train/               # Training images (clean, microfouling, mediumfouling, macrofouling)
│   ├── valid/               # Validation images
│   └── test/                # Test images
├── models/
│   └── final_cnn_model.keras
├── my_saved_sd_model/       # Local Stable Diffusion img2img model
├── app.py                   # Streamlit web application
├── train_image.py           # CNN training script
└── README.md
```

---

## Fouling Classes

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Clean | No marine growth |
| 1 | Microfouling | Thin biofilm / slime layer |
| 2 | Mediumfouling | Patchy barnacles and algae |
| 3 | Macrofouling | Dense barnacles, mussels, tubeworms |

---

## Setup

### Requirements

```bash
pip install tensorflow keras torch torchvision diffusers streamlit matplotlib pillow pandas
```

### Train the CNN

```bash
python "train_image.py"
```

- Input images: `classification_dataset/train/` and `classification_dataset/valid/`
- Output model: `models/cnn_model.h5`
- Image size: 128×128, batch size: 8, epochs: 15

---

## Running the App

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### App Features

- Upload a hull image and set environmental conditions (temperature, salinity, region, speed)
- CNN classifies the fouling level with confidence scores
- Rule-based maintenance optimizer recommends action (clean now, monitor, no action)
- 90-day cost projection including fuel drag penalties and cleaning costs
- Stable Diffusion img2img generates a simulated fouling image based on predicted class

---

## Maintenance Optimizer

The optimizer uses environmental inputs to estimate:

- Days until fouling progresses to the next class
- Extra fuel cost from drag over a 90-day horizon
- Net benefit of cleaning now vs waiting

Urgency levels: `NONE` → `LOW` → `MEDIUM` → `HIGH` → `CRITICAL`

---

## Regions Supported

| Code | Region |
|------|--------|
| `gl` | Great Lakes (freshwater) |
| `ec` | East Canada (Atlantic) |
| `wc` | West Canada (Pacific) |
| `ar` | Arctic (polar) |

---

## Model Details

- Architecture: Simple CNN (Conv2D → MaxPool → Conv2D → MaxPool → Flatten → Dense → Dropout → Softmax)
- Optimizer: Adam
- Loss: Categorical cross-entropy
- Stable Diffusion: img2img pipeline loaded from `my_saved_sd_model/`
