# 🎶 Emotion-Based Music Recommender  

An AI-powered music recommendation system that detects the user’s **emotion in real-time** using facial & hand landmarks, and plays music that matches the mood.  

Built with **Flask**, **Mediapipe**, **TensorFlow/Keras**, and **OpenCV**.  

---

## 🚀 Features  

- 👤 **User Authentication**  
  - Signup & Login system using `users.json`  
  - Saves user preferences like language & genre  

- 🎭 **Emotion Detection**  
  - Uses **Mediapipe** for facial & hand landmarks  
  - Predicts emotions via a **deep learning model** (`model.keras`)  

- 🎵 **Music Recommendation**  
  - Maps detected emotions with user’s chosen **language + genre**  
  - Displays and plays relevant songs  

- 📷 **Real-Time Camera Integration**  
  - Live webcam feed with detected landmarks  
  - Updates predicted emotion continuously  

---

## 🛠️ Tech Stack  

- **Frontend:** HTML, CSS, Jinja Templates (Flask)  
- **Backend:** Python (Flask)  
- **ML/DL:** TensorFlow/Keras, NumPy, scikit-learn  
- **Computer Vision:** Mediapipe, OpenCV  
- **Data Storage:** JSON (for users & preferences), NumPy `.npy` files (for datasets)  

---

## 📖 Run the website

- python app.py

