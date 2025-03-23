# 🛡️ Deepfake Image & Audio Detection
#Innoverse
## 🚀 Overview
**ShivaSpear Deepfake Detection** leverages AI to detect manipulated images and audio, combating misinformation with cutting-edge machine learning.

## 🔥 Key Features
✅ **Image & Audio Detection** – Identifies deepfakes with high accuracy.  
✅ **Advanced AI Models** – Uses CNNs (EfficientNet, ResNet, Xception).  
✅ **Preprocessing Tools** – OpenCV & Librosa for data refinement.  
✅ **User-Friendly UI** – Streamlit-powered web app.  
✅ **Real-time Predictions** – Fast inference for instant results.  

## 🛠️ Tech Stack
| **Category** | **Technologies** |
|-------------|-----------------|
| ML & AI | TensorFlow, Keras, PyTorch (optional), Scikit-learn |
| Image Processing | OpenCV, Pillow (PIL) |
| Audio Processing | Librosa, SciPy |
| Data Handling | NumPy, Pandas, Matplotlib, Seaborn |
| Deployment | Streamlit |

## ⚙️ Implementation Workflow

🔹 **Step 1: Data Preprocessing**  
📌 **Tools:** OpenCV, Pillow, Librosa, SciPy  
✔ Collect & clean datasets (FaceForensics++, Celeb-DF, DFDC, ASVspoof).  
✔ Resize images, normalize pixels, extract MFCCs from audio.  
✔ Detect faces & standardize audio sample rates.  

🔹 **Step 2: Model Training**  
📌 **Tools:** TensorFlow/Keras, PyTorch  
✔ Fine-tune pre-trained CNN models.  
✔ Apply data augmentation techniques.  
✔ Optimize classification with Binary Cross-Entropy Loss.  

🔹 **Step 3: Evaluation & Metrics**  
📌 **Tools:** Scikit-learn, Matplotlib  
✔ Track accuracy, precision, recall, and ROC-AUC scores.  
✔ Confusion matrix analysis for insights.  

🔹 **Step 4: Deployment & UI**  
📌 **Tools:** Streamlit  
✔ Load trained models into an interactive web app.  
✔ Enable users to upload files for detection.  
✔ Display predictions with probability scores.  

## 📊 Performance Metrics
✅ **Image Detection Accuracy:** 90-98%  
✅ **Audio Detection Accuracy:** 85-95%  
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
4️⃣ **Upload an Image or Audio File** to detect deepfakes.

## 📚 References
🔗 FaceForensics++: [GitHub](https://github.com/ondyari/FaceForensics)  
🔗 Celeb-DF Dataset: [GitHub](https://github.com/yuezunli/celeb-deepfake)  
🔗 ASVspoof Dataset: [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336)  

## 👥 Contributors
- **Rohan Gaikwad** – Project Lead  
- **Team Members** – Satyam Kamdam , Swapnil koli , Devshri Damle
📧 **Contact:** [itzrohan007@gmail.com] | 🔗 [linkedin/rohangaikwadlink]  

 *If this project helps you, consider starring the repo!* 


