# 🎨 U-Net Colorization  
### *Automatic Colorization of Grayscale Vintage Photos using Deep Learning*  
**Author:** Oualid Allouch  
**Télécom Physique Strasbourg — Data & AI Engineer**  
📧 [oualid.allouch@etu.unistra.fr](mailto:oualid.allouch@etu.unistra.fr)  
🔗 [LinkedIn – Oualid Allouch]((https://www.linkedin.com/in/oualid-allouch-608b3738a/))

---

## 🇬🇧 Overview / 🇫🇷 Présentation

This project implements a **U-Net-based deep neural network** trained on **ImageNet-64** for automatic colorization of grayscale or vintage photos.  
It combines **perceptual loss (VGG16 features)** with a custom **OldPhotoMaker degradation pipeline**, reproducing realistic aging (grain, vignette, scratches) to make the model robust to historical photos.

Ce projet met en œuvre un **réseau de neurones U-Net** entraîné sur **ImageNet-64** pour la **colorisation automatique de photos anciennes en niveaux de gris**.  
Il associe une **perte perceptuelle (VGG16)** à un **pipeline de dégradation personnalisé (OldPhotoMaker)** simulant le vieillissement des clichés (grain, vignette, rayures).

---

## 🧠 Model Architecture / Architecture du Modèle

**U-Net Encoder-Decoder**
- Input: 1-channel grayscale image (64×64)
- Encoder: stacked convolutions with downsampling
- Decoder: skip connections + upsampling to restore spatial color details
- Output: 3-channel colorized image

**Loss Function**
- `L_total = L1 + 0.1 × L_perceptual(VGG16)`
- Perceptual term ensures semantically consistent colors

<p align="center">
  <img src="https://raw.githubusercontent.com/oualidall/U-Net-Colorization/main/docs/unet_architecture.png" alt="U-Net Diagram" width="650"/>
</p>

---

## 🧩 Training Pipeline / Pipeline d’Entraînement

| Step | Description |
|------|--------------|
| 1️⃣ | **OldPhotoMaker**: applies sepia tone, scratches, vignettes, and film grain to clean RGB images |
| 2️⃣ | Convert degraded RGB → grayscale input |
| 3️⃣ | Train **U-Net** to predict the original RGB colors |
| 4️⃣ | Optimize with **L1 + VGG perceptual loss** |
| 5️⃣ | Evaluate with **SSIM** (structural similarity) and **L1 loss** |

<p align="center">
  <img src="https://raw.githubusercontent.com/oualidall/U-Net-Colorization/main/docs/pipeline_diagram.png" alt="Training Pipeline" width="750"/>
</p>

---

## 📊 Results / Résultats

**Quantitative Metrics (Validation):**
| Epoch | L1 ↓ | VGG ↓ | SSIM ↑ |
|:------|:-----|:------|:------|
| 15 | 0.060 | 0.89 | **0.86** |

---

### 🖼️ Visual Examples (Before → Grayscale → Predicted)

| Example | Link |
|----------|------|
| Dogs | [viz_class_300_test.png](outputs_examples/viz_class_300_test.png) |
| Castles | [viz_class_700_test.png](outputs_examples/viz_class_700_test.png) |
| Class #4 | [viz_class_0004_test.png](outputs_examples/viz_class_0004_test.png) |
| Class #15 | [viz_class_0015_test.png](outputs_examples/viz_class_0015_test.png) |
| Class #620 | [viz_class_0620_test.png](outputs_examples/viz_class_0620_test.png) |

*(Each triplet: Original → Grayscale → U-Net prediction)*

---

## ⚙️ Usage / Utilisation

### 🔧 Installation

```bash
git clone https://github.com/oualidall/U-Net-Colorization.git
cd U-Net-Colorization
pip install -r requirements.txt
