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

# Configure page
st.set_page_config(
    page_title="Deteksi Bahasa Isyarat SIBI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    margin: 1rem 0;
}
.confidence-high {
    color: #28a745;
    font-weight: bold;
}
.confidence-medium {
    color: #ffc107;
    font-weight: bold;
}
.confidence-low {
    color: #dc3545;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

class SignLanguageDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.3,  # Lebih rendah untuk deteksi yang lebih sensitif
            min_tracking_confidence=0.3    # Lebih rendah
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.model = None
        self.scaler = None
        self.labels = [chr(i) for i in range(65, 91)]  # A-Z
    
    def load_model(self, model_path):
        """Load the trained model"""
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
        """Extract hand landmarks from image with enhanced features"""
        try:
            # Convert PIL to OpenCV format
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Pastikan format RGB untuk MediaPipe
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif image.shape[2] == 3:  # Asumsi sudah RGB dari PIL
                    rgb_image = image
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Process image
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extract basic landmarks (x, y, z) - 63 features
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Calculate enhanced features - 18 features
                enhanced_features = self.calculate_enhanced_features(hand_landmarks)
                
                # Combine all features (63 + 18 = 81)
                all_features = landmarks + enhanced_features
                
                return np.array(all_features), results
            
            return None, None
        
        except Exception as e:
            st.error(f"Error extracting landmarks: {e}")
            return None, None
    
    def calculate_enhanced_features(self, hand_landmarks):
        """Calculate additional geometric and spatial features"""
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        features = []
        
        # Key points mapping
        key_points = {
            'wrist': 0, 'thumb_tip': 4, 'index_tip': 8, 'middle_tip': 12,
            'ring_tip': 16, 'pinky_tip': 20, 'thumb_mcp': 2, 'index_mcp': 5,
            'middle_mcp': 9, 'ring_mcp': 13, 'pinky_mcp': 17
        }
        
        # 1. Distance from wrist to fingertips (5 features)
        wrist = landmarks[key_points['wrist']]
        for tip_name, tip_idx in [('thumb_tip', 4), ('index_tip', 8), ('middle_tip', 12), ('ring_tip', 16), ('pinky_tip', 20)]:
            tip = landmarks[tip_idx]
            distance = np.sqrt((tip[0] - wrist[0])**2 + (tip[1] - wrist[1])**2 + (tip[2] - wrist[2])**2)
            features.append(distance)
        
        # 2. Angle between thumb and index finger (1 feature)
        thumb_vec = np.array(landmarks[4]) - np.array(landmarks[2])
        index_vec = np.array(landmarks[8]) - np.array(landmarks[5])
        cos_angle = np.dot(thumb_vec, index_vec) / (np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec) + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        features.append(angle)
        
        # 3. Extension ratios for each finger (5 features)
        for finger_tips, finger_pips in [(4, 3), (8, 6), (12, 10), (16, 14), (20, 18)]:
            tip = landmarks[finger_tips]
            pip = landmarks[finger_pips]
            mcp_idx = finger_tips - 2 if finger_tips == 4 else finger_tips - 3
            mcp = landmarks[mcp_idx]
            
            tip_to_mcp = np.linalg.norm(np.array(tip) - np.array(mcp))
            pip_to_mcp = np.linalg.norm(np.array(pip) - np.array(mcp))
            extension_ratio = tip_to_mcp / (pip_to_mcp + 1e-8)
            features.append(extension_ratio)
        
        # 4. Hand orientation vector (3 features)
        orientation_vec = np.array(landmarks[9]) - np.array(landmarks[0])
        features.extend(orientation_vec)
        
        # 5. Finger spread distances (4 features)
        fingertips = [4, 8, 12, 16, 20]
        for i in range(len(fingertips) - 1):
            tip1 = landmarks[fingertips[i]]
            tip2 = landmarks[fingertips[i + 1]]
            spread = np.linalg.norm(np.array(tip1) - np.array(tip2))
            features.append(spread)
        
        return features
    
    def predict_sign(self, landmarks):
        """Predict sign from landmarks"""
        try:
            if self.model is None:
                return None, 0.0
            
            # Prepare landmarks
            landmarks_array = landmarks.reshape(1, -1)
            
            # Apply scaling if available
            if self.scaler:
                landmarks_array = self.scaler.transform(landmarks_array)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(landmarks_array)[0]
                predicted_class = np.argmax(probabilities)
                confidence = float(probabilities[predicted_class])
                
                # Apply confidence adjustment for close predictions
                second_highest_idx = np.argsort(probabilities)[-2]
                second_confidence = probabilities[second_highest_idx]
                
                if confidence - second_confidence < 0.1:
                    confidence *= 0.8
            else:
                predicted_class = self.model.predict(landmarks_array)[0]
                confidence = 0.7
            
            # Get predicted letter
            if predicted_class < len(self.labels):
                predicted_letter = self.labels[predicted_class]
            else:
                predicted_letter = '?'
            
            return predicted_letter, confidence
        
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            return None, 0.0
    
    def draw_landmarks(self, image, results):
        """Draw landmarks on image"""
        try:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            return image
        except Exception as e:
            st.error(f"Error drawing landmarks: {e}")
            return image

# Initialize detector
@st.cache_resource
def get_detector():
    return SignLanguageDetector()

detector = get_detector()

# Sidebar for model selection
st.sidebar.title("⚙️ Pengaturan Model")

model_files = [
    "improved_model_svm_0.999.p",
    "improved_model_svm_0.997.p", 
    "ensemble_model_acc_0.920.p",
    "model_improved.p"
]

selected_model = st.sidebar.selectbox(
    "Pilih Model:",
    model_files,
    index=0
)

model_path = selected_model

if st.sidebar.button("🔄 Load Model") or 'model_loaded' not in st.session_state:
    if os.path.exists(model_path):
        if detector.load_model(model_path):
            st.sidebar.success(f"✅ Model {selected_model} berhasil dimuat!")
            st.session_state.model_loaded = True
        else:
            st.sidebar.error("❌ Gagal memuat model")
            st.session_state.model_loaded = False
    else:
        st.sidebar.error(f"❌ File model tidak ditemukan: {selected_model}")
        st.session_state.model_loaded = False

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Informasi Model")

if st.session_state.get('model_loaded', False):
    st.sidebar.info(f"**File:** {selected_model}")
    st.sidebar.info(f"**Path:** {model_path}")
    
    if detector.model:
        model_type = type(detector.model).__name__
        st.sidebar.info(f"**Tipe Model:** {model_type}")
    
    if detector.scaler:
        scaler_type = type(detector.scaler).__name__
        st.sidebar.info(f"**Scaler:** {scaler_type}")
    
    st.sidebar.markdown("**📈 Performa Model**")
    performance_data = {
        "improved_model_svm_0.999.p": {"accuracy": "99.9%", "type": "SVM", "features": "81 landmarks"},
        "improved_model_svm_0.997.p": {"accuracy": "99.7%", "type": "SVM", "features": "81 landmarks"},
        "ensemble_model_acc_0.920.p": {"accuracy": "92.0%", "type": "Ensemble", "features": "81 landmarks"},
        "model_improved.p": {"accuracy": "N/A", "type": "Unknown", "features": "81 landmarks"}
    }
    
    if selected_model in performance_data:
        perf = performance_data[selected_model]
        st.sidebar.metric("Akurasi", perf["accuracy"])
        st.sidebar.metric("Tipe Model", perf["type"])
        st.sidebar.metric("Jumlah Fitur", perf["features"])
    
    st.sidebar.markdown("**🔤 Label yang Didukung**")
    image_folder = "reference_images"
    
    letters_per_row = 3
    for i in range(0, len(detector.labels), letters_per_row):
        cols = st.sidebar.columns(letters_per_row)
        for j, letter in enumerate(detector.labels[i:i+letters_per_row]):
            with cols[j]:
                if st.button(letter, key=f"sidebar_btn_{letter}"):
                    image_path = os.path.join(image_folder, f"{letter}.jpg")
                    if os.path.exists(image_path):
                        st.sidebar.image(
                            image_path,
                            caption=f"Gesture {letter}",
                            use_column_width=True
                        )
                    else:
                        st.sidebar.warning(f"Gambar {letter} tidak ditemukan")
else:
    st.sidebar.warning("⚠️ Silakan load model terlebih dahulu")
    st.sidebar.subheader("📋 Model yang Tersedia")
    for model_file in model_files:
        st.sidebar.write(f"• {model_file}")

st.markdown('<h1 class="main-header">🤟 Deteksi Bahasa Isyarat SIBI</h1>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📷 Upload Gambar", "📹 Webcam Real-time", "📊 Informasi Model"])

# Tab 1: Image Upload
with tab1:
    st.header("📷 Deteksi dari Gambar")
    
    uploaded_file = st.file_uploader(
        "Pilih gambar bahasa isyarat:",
        type=['png', 'jpg', 'jpeg'],
        help="Upload gambar yang menunjukkan gesture bahasa isyarat SIBI"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Gambar Asli")
            st.image(image, caption="Gambar yang diupload", use_column_width=True)
        
        with col2:
            st.subheader("Hasil Deteksi")
            
            if st.session_state.get('model_loaded', False):
                # Extract landmarks
                landmarks, results = detector.extract_landmarks(image)
                
                if landmarks is not None:
                    # Draw landmarks
                    image_with_landmarks = np.array(image.copy())
                    image_with_landmarks = detector.draw_landmarks(image_with_landmarks, results)
                    
                    st.image(image_with_landmarks, caption="Gambar dengan Landmarks", use_column_width=True)
                    
                    # Make prediction
                    prediction, confidence = detector.predict_sign(landmarks)
                    
                    if prediction:
                        # Determine confidence color
                        if confidence >= 0.8:
                            conf_class = "confidence-high"
                        elif confidence >= 0.6:
                            conf_class = "confidence-medium"
                        else:
                            conf_class = "confidence-low"
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>🎯 Hasil Prediksi</h3>
                            <h1 style="text-align: center; color: #1f77b4;">{prediction}</h1>
                            <p class="{conf_class}">Confidence: {confidence:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show landmark coordinates
                        with st.expander("📍 Detail Landmarks"):
                            st.write(f"Jumlah landmarks: {len(landmarks)}")
                            
                            # Pisahkan basic landmarks (63) dan enhanced features (18)
                            basic_landmarks = landmarks[:63]  # 21 landmarks × 3 koordinat (x,y,z)
                            enhanced_features = landmarks[63:]  # 18 enhanced features
                            
                            # Tampilkan basic landmarks dalam format (x, y, z)
                            st.subheader("🖐️ Basic Hand Landmarks (21 points)")
                            landmarks_xyz = np.array(basic_landmarks).reshape(-1, 3)
                            landmarks_df = pd.DataFrame(landmarks_xyz, 
                                                      columns=['X', 'Y', 'Z'],
                                                      index=[f'Point_{i}' for i in range(21)])
                            st.dataframe(landmarks_df, use_container_width=True)
                            
                            # Tampilkan enhanced features
                            st.subheader("✨ Enhanced Features (18 features)")
                            enhanced_df = pd.DataFrame({
                                'Feature': [f'Enhanced_{i}' for i in range(18)],
                                'Value': enhanced_features
                            })
                            st.dataframe(enhanced_df, use_container_width=True)
                    else:
                        st.error("❌ Gagal melakukan prediksi")
                else:
                    st.error("❌ Tidak ada tangan terdeteksi dalam gambar")
            else:
                st.warning("⚠️ Silakan load model terlebih dahulu di sidebar")

# Tab 2: Webcam Real-time
with tab2:
    st.header("📸 Ambil Gambar dari Webcam")

    img_file = st.camera_input("Ambil gambar dengan webcam")
    if img_file is not None:
        image = Image.open(img_file)
        st.image(image, caption="Gambar dari Webcam")

        if st.session_state.get('model_loaded', False):
            # Ekstraksi fitur pakai detector
            landmarks, results = detector.extract_landmarks(image)

            if landmarks is not None:
                # Gambarkan landmarks
                image_with_landmarks = np.array(image.copy())
                image_with_landmarks = detector.draw_landmarks(image_with_landmarks, results)
                st.image(image_with_landmarks, caption="Landmarks", use_column_width=True)

                # Prediksi
                prediction, confidence = detector.predict_sign(landmarks)
                if prediction:
                    st.success(f"Prediksi: {prediction} (Confidence: {confidence:.2f})")
                else:
                    st.error("❌ Gagal melakukan prediksi")
            else:
                st.warning("Tidak ada tangan terdeteksi")
        else:
            st.warning("⚠️ Silakan load model dulu di sidebar")


# Tab 3: Model Information
with tab3:
    st.header("📊 Informasi Model")

    if st.session_state.get('model_loaded', False):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Model yang Dimuat")
            st.info(f"**File:** {selected_model}")
            st.info(f"**Path:** {model_path}")

            if detector.model:
                model_type = type(detector.model).__name__
                st.info(f"**Tipe Model:** {model_type}")

            if detector.scaler:
                scaler_type = type(detector.scaler).__name__
                st.info(f"**Scaler:** {scaler_type}")

        with col2:
            st.subheader("🔤 Label yang Didukung")

            # Folder gambar referensi
            image_folder = "reference_images"

            # Display supported letters as buttons in a grid
            letters_per_row = 6
            for i in range(0, len(detector.labels), letters_per_row):
                cols = st.columns(letters_per_row)
                for j, letter in enumerate(detector.labels[i:i+letters_per_row]):
                    with cols[j]:
                        if st.button(letter, key=f"btn_{letter}"):
                            image_path = os.path.join(image_folder, f"{letter}.jpg")
                            if os.path.exists(image_path):
                                st.image(
                                    image_path,
                                    caption=f"Contoh Gesture Huruf {letter}",
                                    use_column_width=True
                                )
                            else:
                                st.warning(f"Gambar untuk huruf {letter} tidak ditemukan")

        # Model performance info
        st.subheader("📈 Performa Model")
        performance_data = {
            "improved_model_svm_0.999.p": {"accuracy": "99.9%", "type": "SVM", "features": "81 landmarks"},
            "improved_model_svm_0.997.p": {"accuracy": "99.7%", "type": "SVM", "features": "81 landmarks"},
            "ensemble_model_acc_0.920.p": {"accuracy": "92.0%", "type": "Ensemble", "features": "81 landmarks"},
            "model_improved.p": {"accuracy": "N/A", "type": "Unknown", "features": "81 landmarks"}
        }

        if selected_model in performance_data:
            perf = performance_data[selected_model]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Akurasi", perf["accuracy"])
            with col2:
                st.metric("Tipe Model", perf["type"])
            with col3:
                st.metric("Jumlah Fitur", perf["features"])

    else:
        st.warning("⚠️ Silakan load model terlebih dahulu di sidebar")

        st.subheader("📋 Model yang Tersedia")
        for model_file in model_files:
            st.write(f"• {model_file}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤟 Aplikasi Deteksi Bahasa Isyarat SIBI</p>
    <p>Dibuat menggunakan Streamlit dan MediaPipe</p>
</div>
""", unsafe_allow_html=True)
