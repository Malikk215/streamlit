import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
from collections import deque, Counter

class ImprovedSignLanguageDetector:
    def __init__(self, model_path='model.p'):
        # Load model
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict):
                self.model = model_data['model']
                self.label_encoder = model_data.get('label_encoder')
                self.scaler = model_data.get('scaler')
                self.labels = model_data.get('labels', list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
            else:
                self.model = model_data
                self.label_encoder = None
                self.scaler = None
                self.labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # MediaPipe hands setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Simplified variables
        self.prediction_history = deque(maxlen=5)
        self.confidence_threshold = 0.3  # Lebih rendah untuk deteksi yang lebih sensitif
        self.stable_prediction = None
        self.stable_confidence = 0.0
        self.stable_count = 0
        self.min_stable_frames = 2
        self.hand_detected = False
        self.hand_bbox = None
        
        print(f"✅ Model loaded successfully: {model_path}")
        print(f"📊 Labels available: {len(self.labels)} classes")
    
    def extract_hand_landmarks(self, image):
        """Extract hand landmarks with proper feature count"""
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extract basic landmarks (63 features: 21 points * 3 coordinates)
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Calculate enhanced features (18 features)
                enhanced_features = self.calculate_enhanced_features(landmarks)
                
                # Combine to get exactly 81 features
                all_features = landmarks + enhanced_features
                
                # Ensure exactly 81 features
                if len(all_features) != 81:
                    if len(all_features) < 81:
                        all_features.extend([0.0] * (81 - len(all_features)))
                    else:
                        all_features = all_features[:81]
                
                # Calculate bounding box
                h, w, _ = image.shape
                x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Add padding
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                self.hand_bbox = (x_min, y_min, x_max, y_max)
                self.hand_detected = True
                
                return all_features, hand_landmarks
            else:
                self.hand_detected = False
                self.hand_bbox = None
                return None, None
                
        except Exception as e:
            print(f"Error extracting landmarks: {e}")
            self.hand_detected = False
            self.hand_bbox = None
            return None, None
    
    def calculate_enhanced_features(self, landmarks):
        """Calculate 18 enhanced features from basic landmarks"""
        try:
            landmarks_array = np.array(landmarks, dtype=np.float32).reshape(21, 3)
            features = []
            
            # 1. Distances from wrist to fingertips (5 features)
            wrist = landmarks_array[0]
            fingertips = [4, 8, 12, 16, 20]
            for tip_idx in fingertips:
                tip = landmarks_array[tip_idx]
                distance = np.sqrt(np.sum((tip - wrist)**2))
                features.append(float(distance))
            
            # 2. Angle between thumb and index finger (1 feature)
            thumb_vec = landmarks_array[4] - landmarks_array[2]
            index_vec = landmarks_array[8] - landmarks_array[5]
            
            cos_angle = np.dot(thumb_vec, index_vec) / (np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            features.append(float(angle))
            
            # 3. Finger extension ratios (5 features)
            finger_data = [(4, 3, 2), (8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]
            for tip_idx, pip_idx, mcp_idx in finger_data:
                tip = landmarks_array[tip_idx]
                pip = landmarks_array[pip_idx]
                mcp = landmarks_array[mcp_idx]
                
                tip_to_mcp = np.linalg.norm(tip - mcp)
                pip_to_mcp = np.linalg.norm(pip - mcp)
                extension_ratio = tip_to_mcp / (pip_to_mcp + 1e-8)
                features.append(float(extension_ratio))
            
            # 4. Hand orientation (3 features)
            orientation_vec = landmarks_array[9] - landmarks_array[0]
            features.extend([float(x) for x in orientation_vec.tolist()])
            
            # 5. Finger spreads (4 features)
            for i in range(len(fingertips) - 1):
                tip1 = landmarks_array[fingertips[i]]
                tip2 = landmarks_array[fingertips[i + 1]]
                spread = np.linalg.norm(tip1 - tip2)
                features.append(float(spread))
            
            # Ensure exactly 18 features
            if len(features) != 18:
                if len(features) < 18:
                    features.extend([0.0] * (18 - len(features)))
                else:
                    features = features[:18]
            
            return features
            
        except Exception as e:
            print(f"Enhanced features error: {e}")
            return [0.0] * 18
    
    def predict_sign(self, landmarks):
        """Main prediction method - simplified and robust"""
        try:
            if landmarks is None:
                return None, 0.0
            
            landmarks_array = np.array([landmarks])
            
            # Apply scaler if available
            if self.scaler is not None:
                landmarks_array = self.scaler.transform(landmarks_array)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                # For sklearn models with probability
                prediction_idx = self.model.predict(landmarks_array)[0]
                probabilities = self.model.predict_proba(landmarks_array)[0]
                confidence = float(np.max(probabilities))
            else:
                # For sklearn models without probability
                prediction_idx = self.model.predict(landmarks_array)[0]
                confidence = 0.7  # Default confidence
            
            # Convert prediction to letter
            if self.label_encoder:
                predicted_class = self.label_encoder.inverse_transform([prediction_idx])[0]
            else:
                predicted_class = self.labels[prediction_idx] if prediction_idx < len(self.labels) else str(prediction_idx)
            
            return predicted_class, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0
    
    def get_smoothed_prediction(self, prediction, confidence):
        """Simplified prediction smoothing"""
        if prediction is None or confidence < self.confidence_threshold:
            return None, 0.0
        
        # Add to history
        self.prediction_history.append((prediction, confidence))
        
        # Update stable prediction
        if prediction == self.stable_prediction:
            self.stable_count += 1
            self.stable_confidence = min(self.stable_confidence * 1.05, 1.0)
        else:
            if confidence > self.confidence_threshold + 0.1:
                self.stable_prediction = prediction
                self.stable_confidence = confidence
                self.stable_count = 1
        
        # Return stable prediction if we have enough frames
        if self.stable_prediction and self.stable_count >= self.min_stable_frames:
            return self.stable_prediction, self.stable_confidence
        
        # Otherwise return most confident recent prediction
        if self.prediction_history:
            recent_predictions = list(self.prediction_history)[-3:]
            best_pred, best_conf = max(recent_predictions, key=lambda x: x[1])
            if best_conf > self.confidence_threshold:
                return best_pred, best_conf
        
        return None, 0.0
    
    def draw_hand_detection(self, frame, hand_landmarks):
        """Draw hand landmarks and bounding box"""
        if hand_landmarks is not None:
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Draw bounding box
        if self.hand_bbox is not None:
            x_min, y_min, x_max, y_max = self.hand_bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    def run_webcam(self):
        """Main webcam loop - simplified and stable"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        prediction_active = True
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        print("🚀 SIBI Sign Language Detector Started!")
        print("Controls:")
        print("  SPACE - Toggle prediction ON/OFF")
        print("  'r' - Reset prediction history")
        print("  'q' - Quit")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks
            landmarks, hand_landmarks = self.extract_hand_landmarks(frame)
            
            # Draw hand detection
            self.draw_hand_detection(frame, hand_landmarks)
            
            # Make prediction if active
            smoothed_prediction = None
            smoothed_confidence = 0.0
            
            if prediction_active and landmarks is not None:
                prediction, confidence = self.predict_sign(landmarks)
                smoothed_prediction, smoothed_confidence = self.get_smoothed_prediction(prediction, confidence)
            
            # Display prediction
            if smoothed_prediction is not None and self.hand_bbox is not None:
                x_min, y_min, x_max, y_max = self.hand_bbox
                
                # Display letter next to hand
                text_x = x_max + 10
                text_y = y_min + 30
                
                # Background for text
                text_size = cv2.getTextSize(smoothed_prediction, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                             (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                
                # Display letter
                color = (0, 255, 0) if smoothed_confidence > 0.7 else (0, 255, 255)
                cv2.putText(frame, smoothed_prediction, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                
                # Display confidence
                conf_text = f"{smoothed_confidence:.0%}"
                cv2.putText(frame, conf_text, (text_x, text_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Status indicators
            status_color = (0, 255, 0) if self.hand_detected else (0, 0, 255)
            status_text = "Hand Detected" if self.hand_detected else "No Hand"
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            prediction_status = "ON" if prediction_active else "OFF"
            prediction_color = (0, 255, 0) if prediction_active else (0, 0, 255)
            cv2.putText(frame, f"Prediction: {prediction_status}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, prediction_color, 2)
            
            # FPS counter
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            cv2.putText(frame, f"FPS: {current_fps}", (500, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Instructions
            instructions = [
                "SPACE - Toggle prediction",
                "R - Reset history",
                "Q - Quit"
            ]
            
            for i, instruction in enumerate(instructions):
                cv2.putText(frame, instruction, (10, 400 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show frame
            cv2.imshow('SIBI Sign Language Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord(' '):
                prediction_active = not prediction_active
                self.prediction_history.clear()
                print(f"🔄 Prediction {'activated' if prediction_active else 'deactivated'}")
            elif key == ord('r') or key == ord('R'):
                self.prediction_history.clear()
                self.stable_prediction = None
                self.stable_confidence = 0.0
                self.stable_count = 0
                print("🔄 Prediction history reset!")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("✅ Webcam closed successfully")

if __name__ == "__main__":
    # Check for model file
    model_files = [
        'improved_model_svm_0.999.p',
        'improved_model_svm_0.997.p', 
        'ensemble_model_acc_0.920.p',
        'model_improved.p', 
        'model.p'
    ]
    model_path = None
    
    for model_file in model_files:
        if os.path.exists(model_file):
            model_path = model_file
            break
    
    if model_path is None:
        print("Error: No model file found. Available files should be:")
        for model_file in model_files:
            print(f"  - {model_file}")
        print("Please train the model first.")
        exit(1)
    
    print(f"Using model: {model_path}")
    
    # Create detector and run webcam
    try:
        detector = ImprovedSignLanguageDetector(model_path)
        detector.run_webcam()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()