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
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
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
        """Extract hand landmarks from image"""
        try:
            # Convert PIL to OpenCV format
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Process image
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extract coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y])
                
                return np.array(landmarks), results
            
            return None, None
        
        except Exception as e:
            st.error(f"Error extracting landmarks: {e}")
            return None, None
    
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

# Model selection
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

model_path = f"c:\\Users\\Malik\\sign_language_app\\python\\{selected_model}"

# Load model
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

# Main title
st.markdown('<h1 class="main-header">🤟 Deteksi Bahasa Isyarat SIBI</h1>', unsafe_allow_html=True)

# Tabs for different modes
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
                            landmarks_df = np.array(landmarks).reshape(-1, 2)
                            st.dataframe(landmarks_df, use_container_width=True)
                    else:
                        st.error("❌ Gagal melakukan prediksi")
                else:
                    st.error("❌ Tidak ada tangan terdeteksi dalam gambar")
            else:
                st.warning("⚠️ Silakan load model terlebih dahulu di sidebar")

# Tab 2: Webcam Real-time
with tab2:
    st.header("📹 Deteksi Real-time dengan Webcam")
    
    if st.session_state.get('model_loaded', False):
        st.info("💡 **Instruksi:** Klik tombol di bawah untuk memulai deteksi real-time")
        
        # Webcam controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_webcam = st.button("🎥 Mulai Webcam", type="primary")
        
        with col2:
            stop_webcam = st.button("⏹️ Stop Webcam")
        
        with col3:
            capture_frame = st.button("📸 Capture Frame")
        
        # Webcam placeholder
        webcam_placeholder = st.empty()
        prediction_placeholder = st.empty()
        
        # Initialize session state for webcam
        if 'webcam_running' not in st.session_state:
            st.session_state.webcam_running = False
        
        if start_webcam:
            st.session_state.webcam_running = True
        
        if stop_webcam:
            st.session_state.webcam_running = False
        
        # Webcam loop (simplified for demo)
        if st.session_state.webcam_running:
            st.info("🔴 Webcam aktif - Tunjukkan gesture bahasa isyarat Anda!")
            
            # Note: Real webcam implementation would require additional setup
            # This is a placeholder for the webcam functionality
            webcam_placeholder.info("""
            📹 **Webcam Real-time Mode**
            
            Untuk implementasi webcam real-time, jalankan script berikut di terminal:
            
            ```bash
            cd python
            python webcam.py
            ```
            
            Atau gunakan mode upload gambar di tab sebelumnya untuk testing.
            """)
    else:
        st.warning("⚠️ Silakan load model terlebih dahulu di sidebar")

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
            
            # Display supported letters in a grid
            letters_per_row = 6
            for i in range(0, len(detector.labels), letters_per_row):
                cols = st.columns(letters_per_row)
                for j, letter in enumerate(detector.labels[i:i+letters_per_row]):
                    with cols[j]:
                        st.markdown(f"<div style='text-align: center; font-size: 1.5rem; font-weight: bold; padding: 0.5rem; background-color: #f0f2f6; border-radius: 5px; margin: 0.2rem;'>{letter}</div>", unsafe_allow_html=True)
        
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
    <p>Dibuat dengan ❤️ menggunakan Streamlit dan MediaPipe</p>
</div>
""", unsafe_allow_html=True)