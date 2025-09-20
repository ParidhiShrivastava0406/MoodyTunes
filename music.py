from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
import json

# Load trained model and labels
model = load_model("model.keras")
label = np.load("labels.npy")

# Initialize Flask app
app = Flask(__name__)

app.secret_key = 'secret'  # For session management

users = {}  # simple in-memory storage

# Initialize MediaPipe Holistic Model
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
holistic = mp_holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Emotion Detection Function
def detect_emotion(frame):
    lst = []
    res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 42)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 42)

        lst = np.array(lst).reshape(1, -1)
        pred = label[np.argmax(model.predict(lst))]
        return pred
    return None

# Video Streaming Generator
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            emotion = detect_emotion(frame)
            if emotion:
                cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                np.save("emotion.npy", np.array([emotion]))  # Save emotion

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route for Home Page
@app.route('/')
def landing():
    return render_template("index.html")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Load users
        with open('users.json', 'r') as f:
            users = json.load(f)

        if username in users:
            user = users[username]
            if user['password'] == password:
                session['user'] = username
                session['language'] = user.get('language', '')
                session['genre'] = user.get('genre', '')
                return redirect(url_for('music'))
            else:
                return "⚠️ Username exists but password is incorrect."

        # Save new user with default language & genre
        users[username] = {
            'password': password,
            'language': '',  # or a default like 'English'
            'genre': ''      # or a default like 'Pop'
        }

        with open('users.json', 'w') as f:
            json.dump(users, f)
        
        
        session['user'] = username
        session['language'] = ''
        session['genre'] = ''

        return redirect(url_for('profile'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Load users
        with open('users.json', 'r') as f:
            users = json.load(f)

        user = users.get(username)
        if user and user['password'] == password:
            session['user'] = username
            session['language'] = user['language']
            session['genre'] = user['genre']
            return redirect(url_for('profile'))
        else:
            return "Invalid credentials"

    return render_template('login.html')

@app.route('/profile', methods=["GET", "POST"])
def profile():
    if 'user' not in session:
        return redirect('/login')

    if request.method == 'POST':
        # Load users
        with open('users.json', 'r') as f:
            users = json.load(f)

        users[session['user']]['language'] = request.form['language']
        users[session['user']]['genre'] = request.form['genre']

        # Save back to file
        with open('users.json', 'w') as f:
            json.dump(users, f)

        # Update session values
        session['language'] = users[session['user']]['language']
        session['genre'] = users[session['user']]['genre']

        return redirect('/music')

    return render_template("profile.html")


@app.route('/music')
def music():
    with open('users.json', 'r') as f:
        users = json.load(f)
    user = users.get(session.get('user'))
    if not user:
        return redirect('/')
    return render_template("music.html", username=session['user'],
                           user_language=user['language'], user_genre=user['genre'])

@app.route('/logout')
def logout():
    session.clear()  # Remove all session data
    return redirect(url_for('landing'))  # Redirect to login page

# Route for Video Feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to Get Detected Emotion
@app.route('/get_emotion', methods=['GET'])
def get_emotion():
    try:
        emotion = np.load("emotion.npy")[0]
        return jsonify({'emotion': emotion})
    except:
        return jsonify({'emotion': 'Unknown'})

# Route to Open YouTube Music Recommendation
@app.route('/recommend_music/<lang>/<singer>', methods=['GET'])
def recommend_music(lang, singer):
    try:
        emotion = np.load("emotion.npy")[0]
        search_url = f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}"
        webbrowser.open(search_url)
        return jsonify({'status': 'success', 'url': search_url})
    except:
        return jsonify({'status': 'error', 'message': 'Emotion not detected'})

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
