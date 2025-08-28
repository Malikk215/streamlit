#!/usr/bin/env python3
import sys
import json
import cv2
import numpy as np
import pickle
import mediapipe as mp
from pathlib import Path

def extract_landmarks(image_path):
    """Extract hand landmarks from image using MediaPipe"""
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            # Get first hand landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y])
                
            return np.array(landmarks)
        
        return None
        
    except Exception as e:
        print(f"Error extracting landmarks: {e}", file=sys.stderr)
        return None

def predict_sign(landmarks, model_path):
    """Predict sign from landmarks using trained model"""
    try:
        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        if isinstance(model_data, dict):
            model = model_data['model']
            scaler = model_data.get('scaler')
            labels = model_data.get('labels', [chr(i) for i in range(65, 91)])  # A-Z
        else:
            model = model_data
            scaler = None
            labels = [chr(i) for i in range(65, 91)]  # A-Z
        
        # Prepare landmarks
        landmarks_array = landmarks.reshape(1, -1)
        
        # Apply scaling if available
        if scaler:
            landmarks_array = scaler.transform(landmarks_array)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(landmarks_array)[0]
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])
            
            # Apply confidence adjustment for close predictions
            second_highest_idx = np.argsort(probabilities)[-2]
            second_confidence = probabilities[second_highest_idx]
            
            if confidence - second_confidence < 0.1:
                confidence *= 0.8
        else:
            predicted_class = model.predict(landmarks_array)[0]
            confidence = 0.7  # Default confidence for non-probabilistic models
        
        # Get predicted letter
        if predicted_class < len(labels):
            predicted_letter = labels[predicted_class]
        else:
            predicted_letter = '?'
            
        return predicted_letter, confidence
        
    except Exception as e:
        print(f"Error in prediction: {e}", file=sys.stderr)
        return None, 0.0

def main():
    """Main function to handle command line arguments and perform prediction"""
    try:
        if len(sys.argv) != 3:  # Keep as 3 - this is correct
            print(json.dumps({
                "success": False,
                "error": "Usage: python predict_sign.py <image_path> <model_path>"
            }))
            sys.exit(1)
        
        image_path = sys.argv[1]
        model_path = sys.argv[2]
        
        # Check if files exist
        if not Path(image_path).exists():
            print(json.dumps({
                "success": False,
                "error": f"Image file not found: {image_path}"
            }))
            sys.exit(1)
            
        if not Path(model_path).exists():
            print(json.dumps({
                "success": False,
                "error": f"Model file not found: {model_path}"
            }))
            sys.exit(1)
        
        # Extract landmarks
        landmarks = extract_landmarks(image_path)
        
        if landmarks is None:
            print(json.dumps({
                "success": False,
                "error": "No hand detected in image"
            }))
            sys.exit(0)
        
        # Make prediction
        prediction, confidence = predict_sign(landmarks, model_path)
        
        if prediction is None:
            print(json.dumps({
                "success": False,
                "error": "Prediction failed"
            }))
            sys.exit(0)
        
        # Return successful result
        result = {
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence, 3),
            "raw_prediction": prediction
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()