from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import face_recognition
import numpy as np
import pickle
from datetime import datetime
import json
from openpyxl import Workbook
import base64
import io
from PIL import Image
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Configuration
UPLOAD_FOLDER = 'uploads'
TRAIN_FOLDER = 'train_images'
OUTPUT_FOLDER = 'output'
MODEL_FILE = 'face_encodings.pkl'

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class AdvancedFaceRecognition:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.attendance = {}
        self.model_file = MODEL_FILE
        self.config = {
            'tolerance': 0.6,
            'model': 'hog',
            'train_path': TRAIN_FOLDER,
            'output_path': OUTPUT_FOLDER
        }

    def load_training_data(self):
        """Load training images and extract face encodings"""
        try:
            # Check if we have saved encodings
            if os.path.exists(self.model_file):
                logger.info("Loading saved face encodings...")
                with open(self.model_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                logger.info(f"Loaded {len(self.known_face_names)} saved face encodings")
                return True
            else:
                logger.info("No saved encodings found")
                return self.process_training_images()
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return False

    def process_training_images(self, new_files_only=False):
    """Process all training images in the train folder"""
    try:
        # Clear existing data unless we're appending
        if not new_files_only:
            self.known_face_encodings = []
            self.known_face_names = []

        train_files = [f for f in os.listdir(TRAIN_FOLDER) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not train_files:
            logger.warning("No training images found")
            return False
        
        # If we're only processing new files, check which ones we haven't processed
        if new_files_only:
            existing_names = set(self.known_face_names)
            train_files = [f for f in train_files 
                          if os.path.splitext(f)[0] not in existing_names]
            
            if not train_files:
                logger.info("No new training images to process")
                return True
        
        for filename in train_files:
            logger.info(f"Processing training image: {filename}")
            image_path = os.path.join(TRAIN_FOLDER, filename)
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {filename}")
                continue
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_image, model=self.config['model'])
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if face_encodings:
                # Use the first face encoding
                name = os.path.splitext(filename)[0].replace('_', ' ').title()
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(name)
                logger.info(f"Successfully processed: {name}")
            else:
                logger.warning(f"No faces found in: {filename}")
        
        if self.known_face_encodings:
            # Save encodings
            with open(self.model_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, f)
            logger.info(f"Saved {len(self.known_face_names)} face encodings")
            return True
        else:
            logger.error("No valid face encodings found")
            return False
            
        
    except Exception as e:
        logger.error(f"Error processing training images: {e}")
        return False

    def recognize_faces(self, image_data):
        """Recognize faces in the provided image data"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_image, model=self.config['model'])
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            results = []
            recognized_names = []
            
            for i, face_encoding in enumerate(face_encodings):
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, 
                    face_encoding, 
                    tolerance=self.config['tolerance']
                )
                
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, 
                    face_encoding
                )
                
                name = "Unknown"
                confidence = 0
                
                if matches and len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = (1 - face_distances[best_match_index]) * 100
                        
                        if name not in recognized_names:
                            recognized_names.append(name)
                
                results.append({
                    'name': name,
                    'confidence': round(confidence, 2),
                    'location': face_locations[i]
                })
            
            return {
                'success': True,
                'faces_found': len(face_locations),
                'recognized_names': recognized_names,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def generate_attendance_report(self, recognized_names):
        """Generate attendance report"""
        try:
            timestamp = datetime.now()
            attendance_data = []
            
            # Check attendance for all known people
            for name in self.known_face_names:
                status = "Present" if name in recognized_names else "Absent"
                attendance_data.append({
                    'name': name,
                    'status': status,
                    'confidence': 0,  # Will be updated with actual confidence
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
                })
            
            # Update confidence for present people
            for entry in attendance_data:
                if entry['status'] == 'Present' and entry['name'] in recognized_names:
                    # This is a simplified approach - in real implementation,
                    # you'd want to store the confidence from recognition
                    entry['confidence'] = 85.0  # Default confidence
            
            return {
                'success': True,
                'attendance_data': attendance_data,
                'total_registered': len(self.known_face_names),
                'present_count': len(recognized_names),
                'absent_count': len(self.known_face_names) - len(recognized_names)
            }
            
        except Exception as e:
            logger.error(f"Error generating attendance report: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# Initialize face recognition system
face_recognizer = AdvancedFaceRecognition()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/upload-training-images', methods=['POST'])
def upload_training_images():
    """Upload training images"""
    try:
        files = request.files.getlist('files')
        
        if not files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400
        
        uploaded_files = []
        
        for file in files:
            if file.filename:
                # Use original filename (assuming it contains person's name)
                filename = file.filename
                filepath = os.path.join(TRAIN_FOLDER, filename)
                file.save(filepath)
                uploaded_files.append(filename)
        
        logger.info(f"Uploaded {len(uploaded_files)} training images")
        
        return jsonify({
            'success': True,
            'uploaded_files': uploaded_files,
            'message': f'Successfully uploaded {len(uploaded_files)} training images'
        })
        
    except Exception as e:
        logger.error(f"Error uploading training images: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the face recognition model on newly uploaded images only"""
    try:
        data = request.get_json()
        files_to_train = data.get('files_to_train', [])
        
        logger.info("Starting training process for new files only...")
        
        # Clear existing data to only keep newly trained faces
        face_recognizer.known_face_encodings = []
        face_recognizer.known_face_names = []
        
        # Only process the specified files
        success = True
        trained_names = []
        
        for filename in files_to_train:
            filepath = os.path.join(TRAIN_FOLDER, filename)
            
            # Load and process the image
            image = cv2.imread(filepath)
            if image is None:
                logger.error(f"Could not load image: {filename}")
                continue
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image, model=face_recognizer.config['model'])
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if face_encodings:
                name = os.path.splitext(filename)[0].replace('_', ' ').title()
                face_recognizer.known_face_encodings.append(face_encodings[0])
                face_recognizer.known_face_names.append(name)
                trained_names.append(name)
                logger.info(f"Successfully trained: {name}")
            else:
                logger.warning(f"No faces found in: {filename}")
                success = False
        
        if trained_names:
            # Save the new encodings
            with open(face_recognizer.model_file, 'wb') as f:
                pickle.dump({
                    'encodings': face_recognizer.known_face_encodings,
                    'names': face_recognizer.known_face_names
                }, f)
                
            return jsonify({
                'success': True,
                'message': f'Training completed successfully. Processed {len(trained_names)} faces.',
                'trained_names': trained_names
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Training failed. No valid faces found in uploaded images.'
            }), 400
            
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/recognize', methods=['POST'])
def recognize_faces():
    """Recognize faces in uploaded images"""
    try:
        data = request.get_json()
        
        if not data or 'images' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        # Load training data if not already loaded
        if not face_recognizer.known_face_encodings:
            if not face_recognizer.load_training_data():
                return jsonify({
                    'success': False, 
                    'error': 'No training data available. Please train the model first.'
                }), 400
        
        all_recognized = set()
        results = []
        
        for i, image_data in enumerate(data['images']):
            logger.info(f"Processing image {i+1}/{len(data['images'])}")
            
            result = face_recognizer.recognize_faces(image_data)
            
            if result['success']:
                all_recognized.update(result['recognized_names'])
                results.append({
                    'image_index': i,
                    'faces_found': result['faces_found'],
                    'recognized_names': result['recognized_names'],
                    'details': result['results']
                })
            else:
                results.append({
                    'image_index': i,
                    'error': result['error']
                })
        
        return jsonify({
            'success': True,
            'total_recognized': list(all_recognized),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error during recognition: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/attendance/generate', methods=['POST'])
def generate_attendance():
    """Generate attendance report"""
    try:
        data = request.get_json()
        recognized_names = data.get('recognized_names', [])
        
        if not face_recognizer.known_face_names:
            if not face_recognizer.load_training_data():
                return jsonify({
                    'success': False,
                    'error': 'No training data available'
                }), 400
        
        report = face_recognizer.generate_attendance_report(recognized_names)
        
        return jsonify(report)
        
    except Exception as e:
        logger.error(f"Error generating attendance: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/attendance/download/<format>', methods=['POST'])
def download_attendance(format):
    """Download attendance report in specified format"""
    try:
        data = request.get_json()
        attendance_data = data.get('attendance_data', [])
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'excel':
            # Create Excel file
            wb = Workbook()
            ws = wb.active
            ws.title = "Attendance Report"
            
            # Header
            ws.append(["Name", "Status", "Confidence", "Timestamp"])
            
            # Data
            for entry in attendance_data:
                ws.append([
                    entry['name'],
                    entry['status'],
                    f"{entry['confidence']}%" if entry['confidence'] > 0 else "-",
                    entry['timestamp']
                ])
            
            filename = f"attendance_{timestamp}.xlsx"
            filepath = os.path.join(OUTPUT_FOLDER, filename)
            wb.save(filepath)
            
            return send_file(filepath, as_attachment=True, download_name=filename)
            
        elif format == 'json':
            filename = f"attendance_{timestamp}.json"
            filepath = os.path.join(OUTPUT_FOLDER, filename)
            
            with open(filepath, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'attendance': attendance_data
                }, f, indent=2)
            
            return send_file(filepath, as_attachment=True, download_name=filename)
            
        else:
            return jsonify({'success': False, 'error': 'Invalid format'}), 400
        
    except Exception as e:
        logger.error(f"Error downloading attendance: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    """Handle configuration get/set"""
    try:
        if request.method == 'GET':
            return jsonify({
                'success': True,
                'config': face_recognizer.config
            })
        
        elif request.method == 'POST':
            data = request.get_json()
            
            # Update configuration
            if 'tolerance' in data:
                face_recognizer.config['tolerance'] = float(data['tolerance'])
            if 'model' in data:
                face_recognizer.config['model'] = data['model']
            if 'train_path' in data:
                face_recognizer.config['train_path'] = data['train_path']
            if 'output_path' in data:
                face_recognizer.config['output_path'] = data['output_path']
            
            return jsonify({
                'success': True,
                'message': 'Configuration updated successfully',
                'config': face_recognizer.config
            })
            
    except Exception as e:
        logger.error(f"Error handling config: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get system logs"""
    try:
        # This is a simple implementation
        # In a real system, you'd want to read from actual log files
        logs = [
            {
                'timestamp': datetime.now().isoformat(),
                'level': 'info',
                'message': 'System is running normally'
            }
        ]
        
        return jsonify({
            'success': True,
            'logs': logs
        })
        
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Load training data on startup
    logger.info("Starting Face Recognition System...")
    face_recognizer.load_training_data()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)