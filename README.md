# 🛡️ ShivaSpear Deepfake Detection

## 🚀 Overview
**ShivaSpear Deepfake Detection** is an AI-powered tool for identifying deepfake images, audio, and fake news. It leverages machine learning and natural language processing (NLP) to combat misinformation.

## 🔥 Key Features
✅ **Deepfake Image & Audio Detection** – High-accuracy identification of manipulated media.  
✅ **Fake News Classification** – NLP-powered fake news detection.  
✅ **Preprocessing Tools** – OpenCV, Librosa, and NLP libraries for data refinement.  
✅ **User-Friendly UI** – Streamlit-powered web app for real-time analysis.  
✅ **Fast & Efficient** – Optimized models for quick inference.  

## 🛠️ Tech Stack
| **Category**        | **Technologies** |
|---------------------|-----------------|
| ML & AI            | TensorFlow, Keras, PyTorch (optional), Scikit-learn |
| Image Processing   | OpenCV, Pillow (PIL) |
| Audio Processing   | Librosa, SciPy |
| NLP & Text Analysis | NLTK, SpaCy, TF-IDF, Word Embeddings |
| Data Handling      | NumPy, Pandas, Matplotlib, Seaborn |
| Deployment         | Streamlit |

## ⚙️ Implementation Workflow

### 🔹 Deepfake Image Detection
📌 **Tech Stack:** TensorFlow, OpenCV, Scikit-learn  
✔ Collect & preprocess images (FaceForensics++, Celeb-DF, DFDC).  
✔ Train CNN models (Xception, EfficientNet, ResNet).  
✔ Evaluate accuracy with precision, recall, and F1-score.  
✔ Deploy model using Streamlit.  

### 🔹 Deepfake Audio Detection
📌 **Tech Stack:** Librosa, TensorFlow, Scikit-learn  
✔ Collect & preprocess datasets (ASVspoof, FakeAVCeleb).  
✔ Extract MFCCs, spectrograms, and chroma features.  
✔ Train CNN-based classification models.  
✔ Deploy using an interactive web app.  

### 🔹 Fake News Detection
📌 **Tech Stack:** NLTK, SpaCy, Scikit-learn  
✔ Collect & clean text datasets (FakeNewsNet, LIAR, Kaggle Fake News).  
✔ Convert text into TF-IDF vectors or embeddings.  
✔ Train Logistic Regression classifier.  
✔ Evaluate with confusion matrix and performance metrics.  
✔ Deploy on Streamlit for real-time analysis.  

## 📊 Performance Metrics
✅ **Image Detection Accuracy:** 90-98%  
✅ **Audio Detection Accuracy:** 85-95%  
✅ **Fake News Classification Accuracy:** 80-95%  
✅ **Inference Speed:** <1 second per input  

## 🚀 Quick Start Guide
1️⃣ **Clone the Repo:**  
```bash
git clone https://github.com/RohanExploit/ShivaSpear_Deepfake.git
cd ShivaSpear_Deepfake
```
2️⃣ **Install Dependencies:**  
```bash
pip install -r requirements.txt
```
3️⃣ **Run the Web App:**  
```bash
streamlit run app.py
```
4️⃣ **Upload an Image, Audio File, or News Text** to analyze deepfakes.

## 📚 References
🔗 FaceForensics++: [GitHub](https://github.com/ondyari/FaceForensics)  
🔗 Celeb-DF Dataset: [GitHub](https://github.com/yuezunli/celeb-deepfake)  
🔗 ASVspoof Dataset: [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336)  
🔗 FakeNewsNet: [GitHub](https://github.com/KaiDMML/FakeNewsNet)  

## 👥 Contributors
- **Rohan Gaikwad** – Project Lead  
- **Team Members** – Devshri Damle , Satyam Kadam
📧 **Contact:** [itzrohan007@gmail.com] | 🔗 [www.linkedin.com/in/rohanvijaygaikwad]  

⭐ *If this project helps you, consider starring the repo!* ⭐

 


