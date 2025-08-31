import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import cv2
import numpy as np
import pickle
import mediapipe as mp
from PIL import Image
import tempfile
import os
import json
from pathlib import Path
import time
import pandas as pd
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Deteksi Bahasa Isyarat SIBI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

.main-header {
    font-size: 3.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
    animation: glow 2s ease-in-out infinite alternate;
}

@keyframes glow {
    from { text-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
    to { text-shadow: 0 0 30px rgba(118, 75, 162, 0.8); }
}

.prediction-box {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    margin: 1rem 0;
    text-align: center;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.02); }
    100% { transform: scale(1); }
}

.prediction-letter {
    font-size: 4rem;
    font-weight: bold;
    color: white;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    margin: 1rem 0;
}

.confidence-high {
    color: #00ff88;
    font-weight: bold;
    font-size: 1.2rem;
    text-shadow: 0 0 10px rgba(0,255,136,0.5);
}

.confidence-medium {
    color: #ffaa00;
    font-weight: bold;
    font-size: 1.2rem;
    text-shadow: 0 0 10px rgba(255,170,0,0.5);
}

.confidence-low {
    color: #ff4444;
    font-weight: bold;
    font-size: 1.2rem;
    text-shadow: 0 0 10px rgba(255,68,68,0.5);
}

.sidebar-model-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 15px;
    margin: 1rem 0;
    color: white;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

.metric-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    text-align: center;
    color: white;
    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
}

.letter-button {
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    border: none;
    border-radius: 10px;
    padding: 0.5rem;
    margin: 0.2rem;
    color: white;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
}

.letter-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

.upload-area {
    border: 3px dashed #667eea;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
    transition: all 0.3s ease;
}

.upload-area:hover {
    border-color: #764ba2;
    background: linear-gradient(135deg, rgba(102,126,234,0.2) 0%, rgba(118,75,162,0.2) 100%);
}

.landmark-visualization {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    padding: 1rem;
    border-radius: 15px;
    margin: 1rem 0;
}

.success-animation {
    animation: bounce 1s ease-in-out;
}

@keyframes bounce {
    0%, 20%, 60%, 100% { transform: translateY(0); }
    40% { transform: translateY(-10px); }
    80% { transform: translateY(-5px); }
}

.loading-spinner {
    border: 4px solid #f3f3f3;
    border-top: 4px solid #667eea;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.tab-content {
    padding: 2rem;
    background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(240,242,246,0.9) 100%);
    border-radius: 15px;
    margin-top: 1rem;
}

.footer {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    text-align: center;
    padding: 2rem;
    border-radius: 15px;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

class SignLanguageDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.model = None
        self.scaler = None
        self.labels = [chr(i) for i in range(65, 91)]
        self.prediction_history = []
    
    def load_model(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict):
                self.model = model_data['model']
                self.scaler = model_data.get('scaler')
                self.labels = model_data.get('labels', self.labels)
            else:
                self.model = model_data
                self.scaler = None
            
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def extract_landmarks(self, image):
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            if len(image.shape) == 3:
                if image.shape[2] == 4:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif image.shape[2] == 3:
                    rgb_image = image
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                enhanced_features = self.calculate_enhanced_features(hand_landmarks)
                all_features = landmarks + enhanced_features
                
                return np.array(all_features), results
            
            return None, None
        
        except Exception as e:
            st.error(f"Error extracting landmarks: {e}")
            return None, None
    
    def calculate_enhanced_features(self, hand_landmarks):
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        features = []
        
        key_points = {
            'wrist': 0, 'thumb_tip': 4, 'index_tip': 8, 'middle_tip': 12,
            'ring_tip': 16, 'pinky_tip': 20, 'thumb_mcp': 2, 'index_mcp': 5,
            'middle_mcp': 9, 'ring_mcp': 13, 'pinky_mcp': 17
        }
        
        wrist = landmarks[key_points['wrist']]
        for tip_name, tip_idx in [('thumb_tip', 4), ('index_tip', 8), ('middle_tip', 12), ('ring_tip', 16), ('pinky_tip', 20)]:
            tip = landmarks[tip_idx]
            distance = np.sqrt((tip[0] - wrist[0])**2 + (tip[1] - wrist[1])**2 + (tip[2] - wrist[2])**2)
            features.append(distance)
        
        thumb_vec = np.array(landmarks[4]) - np.array(landmarks[2])
        index_vec = np.array(landmarks[8]) - np.array(landmarks[5])
        cos_angle = np.dot(thumb_vec, index_vec) / (np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec) + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        features.append(angle)
        
        for finger_tips, finger_pips in [(4, 3), (8, 6), (12, 10), (16, 14), (20, 18)]:
            tip = landmarks[finger_tips]
            pip = landmarks[finger_pips]
            mcp_idx = finger_tips - 2 if finger_tips == 4 else finger_tips - 3
            mcp = landmarks[mcp_idx]
            
            tip_to_mcp = np.linalg.norm(np.array(tip) - np.array(mcp))
            pip_to_mcp = np.linalg.norm(np.array(pip) - np.array(mcp))
            extension_ratio = tip_to_mcp / (pip_to_mcp + 1e-8)
            features.append(extension_ratio)
        
        orientation_vec = np.array(landmarks[9]) - np.array(landmarks[0])
        features.extend(orientation_vec)
        
        fingertips = [4, 8, 12, 16, 20]
        for i in range(len(fingertips) - 1):
            tip1 = landmarks[fingertips[i]]
            tip2 = landmarks[fingertips[i + 1]]
            spread = np.linalg.norm(np.array(tip1) - np.array(tip2))
            features.append(spread)
        
        return features
    
    def predict_sign(self, landmarks):
        try:
            if self.model is None:
                return None, 0.0
            
            landmarks_array = landmarks.reshape(1, -1)
            
            if self.scaler:
                landmarks_array = self.scaler.transform(landmarks_array)
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(landmarks_array)[0]
                predicted_class = np.argmax(probabilities)
                confidence = float(probabilities[predicted_class])
                
                second_highest_idx = np.argsort(probabilities)[-2]
                second_confidence = probabilities[second_highest_idx]
                
                if confidence - second_confidence < 0.1:
                    confidence *= 0.8
            else:
                predicted_class = self.model.predict(landmarks_array)[0]
                confidence = 0.7
            
            if predicted_class < len(self.labels):
                predicted_letter = self.labels[predicted_class]
            else:
                predicted_letter = '?'
            
            self.prediction_history.append({
                'letter': predicted_letter,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
            
            return predicted_letter, confidence
        
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            return None, 0.0
    
    def draw_landmarks(self, image, results):
        try:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                    )
            return image
        except Exception as e:
            st.error(f"Error drawing landmarks: {e}")
            return image
    
    def get_confidence_chart(self):
        if not self.prediction_history:
            return None
        
        recent_predictions = self.prediction_history[-10:]
        letters = [p['letter'] for p in recent_predictions]
        confidences = [p['confidence'] for p in recent_predictions]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(letters, confidences, color=['#667eea', '#764ba2', '#4facfe', '#00f2fe', '#f093fb'])
        
        ax.set_title('Confidence Score History', fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Letters', fontsize=12)
        ax.set_ylabel('Confidence', fontsize=12)
        ax.set_ylim(0, 1)
        
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{conf:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig

@st.cache_resource
def get_detector():
    return SignLanguageDetector()

detector = get_detector()

st.sidebar.markdown("""
<div class="sidebar-model-card">
    <h2 style="text-align: center; margin: 0;">⚙️ Model Control</h2>
</div>
""", unsafe_allow_html=True)

model_files = [
    "improved_model_svm_0.999.p",
    "improved_model_svm_0.997.p", 
    "ensemble_model_acc_0.920.p",
    "model_improved.p"
]

selected_model = st.sidebar.selectbox(
    "🎯 Pilih Model:",
    model_files,
    index=0
)

model_path = selected_model

if st.sidebar.button("🚀 Load Model", use_container_width=True) or 'model_loaded' not in st.session_state:
    with st.sidebar:
        with st.spinner("Loading model..."):
            time.sleep(1)
            if os.path.exists(model_path):
                if detector.load_model(model_path):
                    st.success("✅ Model loaded successfully!")
                    st.balloons()
                    st.session_state.model_loaded = True
                else:
                    st.error("❌ Failed to load model")
                    st.session_state.model_loaded = False
            else:
                st.error(f"❌ Model file not found: {selected_model}")
                st.session_state.model_loaded = False

st.sidebar.markdown("---")

if st.session_state.get('model_loaded', False):
    st.sidebar.markdown("""
    <div class="metric-card">
        <h3>📊 Model Information</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.info(f"**📁 File:** {selected_model}")
    st.sidebar.info(f"**📍 Path:** {model_path}")
    
    if detector.model:
        model_type = type(detector.model).__name__
        st.sidebar.info(f"**🤖 Type:** {model_type}")
    
    if detector.scaler:
        scaler_type = type(detector.scaler).__name__
        st.sidebar.info(f"**⚖️ Scaler:** {scaler_type}")
    
    performance_data = {
        "improved_model_svm_0.999.p": {"accuracy": "99.9%", "type": "SVM", "features": "81"},
        "improved_model_svm_0.997.p": {"accuracy": "99.7%", "type": "SVM", "features": "81"},
        "ensemble_model_acc_0.920.p": {"accuracy": "92.0%", "type": "Ensemble", "features": "81"},
        "model_improved.p": {"accuracy": "N/A", "type": "Unknown", "features": "81"}
    }
    
    if selected_model in performance_data:
        perf = performance_data[selected_model]
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            st.metric("🎯 Accuracy", perf["accuracy"])
        with col2:
            st.metric("🔧 Type", perf["type"])
        with col3:
            st.metric("📊 Features", perf["features"])
    
    st.sidebar.markdown("""
    <div class="metric-card">
        <h3>🔤 Supported Letters</h3>
    </div>
    """, unsafe_allow_html=True)
    
    image_folder = "reference_images"
    
    letters_per_row = 3
    for i in range(0, len(detector.labels), letters_per_row):
        cols = st.sidebar.columns(letters_per_row)
        for j, letter in enumerate(detector.labels[i:i+letters_per_row]):
            with cols[j]:
                if st.button(letter, key=f"sidebar_btn_{letter}", use_container_width=True):
                    image_path = os.path.join(image_folder, f"{letter}.jpg")
                    if os.path.exists(image_path):
                        st.sidebar.image(
                            image_path,
                            caption=f"Gesture {letter}",
                            use_column_width=True
                        )
                    else:
                        st.sidebar.warning(f"Image {letter} not found")
else:
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); border-radius: 15px; color: white;">
        <h3>⚠️ Please Load Model First</h3>
        <p>Select and load a model to start detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.subheader("📋 Available Models")
    for model_file in model_files:
        st.sidebar.write(f"• {model_file}")

st.markdown('<h1 class="main-header">🤟 SIBI Sign Language Detection</h1>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📷 Image Upload", "📹 Live Camera"])

with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #667eea;">📸 Upload Your Sign Language Image</h2>
        <p style="font-size: 1.1rem; color: #666;">Upload an image showing SIBI sign language gesture</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image showing SIBI sign language gesture",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 1rem;">
                <h3 style="color: #667eea;">🖼️ Original Image</h3>
            </div>
            """, unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 1rem;">
                <h3 style="color: #667eea;">🔍 Detection Results</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.get('model_loaded', False):
                with st.spinner("Analyzing image..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    landmarks, results = detector.extract_landmarks(image)
                    
                    if landmarks is not None:
                        image_with_landmarks = np.array(image.copy())
                        image_with_landmarks = detector.draw_landmarks(image_with_landmarks, results)
                        
                        st.image(image_with_landmarks, caption="Image with Landmarks", use_column_width=True)
                        
                        prediction, confidence = detector.predict_sign(landmarks)
                        
                        if prediction:
                            if confidence >= 0.8:
                                conf_class = "confidence-high"
                                conf_emoji = "🟢"
                            elif confidence >= 0.6:
                                conf_class = "confidence-medium"
                                conf_emoji = "🟡"
                            else:
                                conf_class = "confidence-low"
                                conf_emoji = "🔴"
                            
                            st.markdown(f"""
                            <div class="prediction-box success-animation">
                                <h3 style="color: white; margin: 0;">🎯 Prediction Result</h3>
                                <div class="prediction-letter">{prediction}</div>
                                <p class="{conf_class}">{conf_emoji} Confidence: {confidence:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.success(f"Detected: {prediction} with {confidence:.1%} confidence")
                            
                            if confidence > 0.9:
                                st.balloons()
                            
                            with st.expander("📊 Detailed Analysis", expanded=False):
                                col_a, col_b = st.columns(2)
                                
                                with col_a:
                                    st.markdown("""
                                    <div class="landmark-visualization">
                                        <h4>🖐️ Hand Landmarks</h4>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    basic_landmarks = landmarks[:63]
                                    enhanced_features = landmarks[63:]
                                    
                                    landmarks_xyz = np.array(basic_landmarks).reshape(-1, 3)
                                    landmarks_df = pd.DataFrame(landmarks_xyz, 
                                                              columns=['X', 'Y', 'Z'],
                                                              index=[f'Point_{i}' for i in range(21)])
                                    st.dataframe(landmarks_df, use_container_width=True)
                                
                                with col_b:
                                    st.markdown("""
                                    <div class="landmark-visualization">
                                        <h4>✨ Enhanced Features</h4>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    enhanced_df = pd.DataFrame({
                                        'Feature': [f'Enhanced_{i}' for i in range(18)],
                                        'Value': enhanced_features
                                    })
                                    st.dataframe(enhanced_df, use_container_width=True)
                                
                                confidence_chart = detector.get_confidence_chart()
                                if confidence_chart:
                                    st.pyplot(confidence_chart)
                        else:
                            st.error("❌ Prediction failed")
                    else:
                        st.markdown("""
                        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); border-radius: 15px; color: white;">
                            <h3>👋 No Hand Detected</h3>
                            <p>Please ensure your hand is clearly visible in the image</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%); border-radius: 15px; color: white;">
                    <h3>⚠️ Model Not Loaded</h3>
                    <p>Please load a model from the sidebar first</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #667eea;">📹 Live Camera Detection</h2>
        <p style="font-size: 1.1rem; color: #666;">Capture images using your webcam for real-time detection</p>
    </div>
    """, unsafe_allow_html=True)

    img_file = st.camera_input("📸 Take a picture with your webcam")
    
    if img_file is not None:
        image = Image.open(img_file)
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 1rem;">
                <h3 style="color: #667eea;">📷 Captured Image</h3>
            </div>
            """, unsafe_allow_html=True)
            st.image(image, caption="Webcam Image", use_column_width=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 1rem;">
                <h3 style="color: #667eea;">🎯 Live Detection</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.get('model_loaded', False):
                with st.spinner("Processing live image..."):
                    landmarks, results = detector.extract_landmarks(image)

                    if landmarks is not None:
                        image_with_landmarks = np.array(image.copy())
                        image_with_landmarks = detector.draw_landmarks(image_with_landmarks, results)
                        st.image(image_with_landmarks, caption="Live Detection", use_column_width=True)

                        prediction, confidence = detector.predict_sign(landmarks)
                        if prediction:
                            if confidence >= 0.8:
                                conf_class = "confidence-high"
                                conf_emoji = "🟢"
                            elif confidence >= 0.6:
                                conf_class = "confidence-medium"
                                conf_emoji = "🟡"
                            else:
                                conf_class = "confidence-low"
                                conf_emoji = "🔴"
                            
                            st.markdown(f"""
                            <div class="prediction-box success-animation">
                                <h3 style="color: white; margin: 0;">🎯 Live Prediction</h3>
                                <div class="prediction-letter">{prediction}</div>
                                <p class="{conf_class}">{conf_emoji} Confidence: {confidence:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if confidence > 0.9:
                                st.success(f"🎉 Excellent detection: {prediction}!")
                            else:
                                st.info(f"Detected: {prediction} ({confidence:.1%})")
                        else:
                            st.error("❌ Prediction failed")
                    else:
                        st.markdown("""
                        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); border-radius: 15px; color: white;">
                            <h3>👋 No Hand Detected</h3>
                            <p>Position your hand clearly in front of the camera</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%); border-radius: 15px; color: white;">
                    <h3>⚠️ Model Not Loaded</h3>
                    <p>Please load a model from the sidebar first</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if detector.prediction_history:
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="color: #667eea;">📈 Prediction History</h2>
    </div>
    """, unsafe_allow_html=True)
    
    recent_predictions = detector.prediction_history[-5:]
    cols = st.columns(len(recent_predictions))
    
    for i, (col, pred) in enumerate(zip(cols, recent_predictions)):
        with col:
            st.markdown(f"""
            <div class="history-card">
                <h3>{pred['letter']}</h3>
                <p>{pred['confidence']:.1%}</p>
                <small>{pred['timestamp'].strftime('%H:%M:%S')}</small>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div class="footer">
    <h2>🤟 SIBI Sign Language Detection System</h2>
    <p>🚀 Powered by Streamlit • 🤖 MediaPipe • 🧠 Machine Learning</p>
</div>
""", unsafe_allow_html=True)
