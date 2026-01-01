# 👗 Multimodal Fashion Recommendation System

An end-to-end fashion product recommendation system** that leverages **images, metadata, and text** to generate visually and semantically similar fashion recommendations.  
The system integrates **CLIP-based image embeddings**, **metadata + text encodings**, **hybrid fusion**, and a **Siamese neural network** to learn a joint style space.

---

## 🚀 Project Overview

Online fashion catalogs contain tens of thousands of products, making it difficult for users to discover similar items efficiently—especially in cold-start scenarios where user interaction data is unavailable.

This project addresses that challenge by building a **content-based, multimodal recommender system** that:

- Works without user interaction history  
- Supports image-based, metadata-based, and hybrid queries  
- Learns deep style similarity using contrastive learning  
- Provides an interactive Streamlit demo for real-time exploration  

---

## 🧠 Models Implemented

The system implements **four complementary recommendation models**:

### 1️⃣ Image-Based Recommender (CLIP)
- Uses **CLIP (ViT-B/32)** to extract 512-D image embeddings  
- Computes cosine similarity between product images  
- Captures visual attributes such as color, texture, and silhouette  

### 2️⃣ Metadata + Text-Based Recommender
- Encodes structured features (category, brand, price, season, usage)  
- Uses **TF-IDF + Truncated SVD** on a constructed `style_signature`  
- Effective for semantic consistency and price stability  

### 3️⃣ Hybrid Recommender
- Combines image and metadata similarity:

Hybrid Similarity = α × ImageSim + (1 − α) × MetaSim

yaml
Copy code

- Balances visual appearance with contextual attributes  

### 4️⃣ Siamese Product Encoder (Advanced)
- Learns a **non-linear joint embedding space**  
- Inputs: CLIP image embedding + metadata embedding  
- Trained using **contrastive (InfoNCE-style) loss**  
- Produces a compact **128-D style embedding**  
- Achieves the **highest category precision**  

---

## 📊 Dataset

- Public Kaggle dataset: *Fashion Product Images (Myntra-like)*  
- ~89,000 products  

**Data sources:**
- `styles.csv` (metadata)  
- `images/` (product images)  
- Parsed JSON attribute files  

### Key Features
- **Numerical:** price, rating, discount, inventory  
- **Categorical:** category hierarchy, color, brand, usage, season  
- **Textual:** `style_signature` (concatenated attributes)  

---

## 🔧 Feature Engineering

- Log-transformed prices  
- Discount ratios and binary flags  
- Category strength metrics  
- Price buckets, seasonal grouping, demographic segments  
- TF-IDF + SVD (64-D) text embeddings  
- StandardScaler + OneHotEncoder via `ColumnTransformer`  

---

## 🧪 Evaluation

Since no user interaction data is available, the system is evaluated using **heuristic item–item metrics**:

- **Category Precision**  
- **NDCG@K**  
- **Precision@K**  
- **Average Price Drift (↓)**  

### Key Results

| Model     | Best Strength |
|-----------|---------------|
| Image     | Visual similarity |
| Metadata  | Best NDCG & price stability |
| Hybrid    | Balanced performance |
| Siamese  | Highest category precision |

---

## 🧩 Clustering Analysis

- Applied **K-Means clustering** on CLIP and Siamese embeddings  
- Siamese embeddings formed tighter, more coherent clusters  
- Visualized using **t-SNE**  
- Confirms quality of learned style representations  

---

## 🖥️ Interactive Demo (Streamlit)

The Streamlit app supports:

- Image upload for visual search  
- Metadata-based queries (gender, category, color, price)  
- Hybrid recommendations  
- Siamese-based similarity search  
- Side-by-side model comparison  

---

## 📁 Repository Structure

├── data/
│ ├── styles.csv
│ ├── FeatureEngineered_StyleData.csv
│ └── images/
│
├── embeddings/
│ ├── image_embeddings_clip.npy
│ ├── siamese_embeddings.npy
│
├── models/
│ ├── siamese_encoder.pt
│
├── notebooks/
│ ├── data_preprocessing.ipynb
│ ├── feature_engineering.ipynb
│ ├── recommendationsystem-builder.ipynb
│
├── app.py # Streamlit application
├── requirements.txt
└── README.md

yaml
Copy code

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/your-username/fashion-recommender.git
cd fashion-recommender
pip install -r requirements.txt
Run the app:

bash
Copy code
streamlit run app.py
🔮 Future Work
Fine-tune CLIP or use fashion-specific visual encoders

Hard-negative mining for Siamese training

Outfit-level recommendations (tops + bottoms + shoes)

ANN search using FAISS for large-scale deployment

Incorporate user interaction signals

A/B testing with real users

🎓 Academic Context
This project was developed as a graduate-level final project for
IS 557 – Machine Learning Techniques & Processes.

It demonstrates:

Multimodal ML

Representation learning

Recommender systems

Model evaluation

Deployment-ready engineering

📬 Contact
Author: Jainam Rajput
Program: MS (STEM) – Data / ML
Interests: Recommender Systems, Multimodal ML, Applied AI
