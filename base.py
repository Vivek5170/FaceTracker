import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import time
import json
import os
from collections import defaultdict, deque
import argparse
from pathlib import Path
import glob

class PoseInvariantFaceTracker:
    def __init__(self, model_name='buffalo_l', confidence_threshold=0.5, imgs_dir='imgs'):
        """
        Initialize pose-invariant face tracker
        
        Args:
            model_name: InsightFace model ('buffalo_l', 'buffalo_m', 'buffalo_s')
            confidence_threshold: Minimum confidence for face detection
            imgs_dir: Directory containing person folders with reference images
        """
        print(f"🚀 Initializing Pose-Invariant Face Tracker...")
        print(f"   Model: {model_name}")
        print(f"   Images Directory: {imgs_dir}")
        
        # Initialize InsightFace
        self.app = FaceAnalysis(name=model_name, providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0)
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = 0.4  # Threshold for face matching
        self.imgs_dir = imgs_dir
        
        # Known faces database - now stores multiple embeddings per person
        self.known_faces = {}  # {name: [embedding1, embedding2, ...]}
        
        # Voting system for robust identification
        self.min_votes = 2  # Minimum votes needed for identification
        self.vote_threshold = 0.35  # Lower threshold for individual votes
        
        # Tracking variables
        self.track_history = defaultdict(lambda: deque(maxlen=10))
        self.face_counter = 0
        self.last_recognition_time = defaultdict(float)
        
        # Load face database from images directory
        self.load_faces_from_directory()
        
        print("✅ Pose-Invariant Face Tracker initialized!")
    
    def save_person_embeddings(self, person_name, embeddings, person_dir):
        """Save all embeddings as a single .npy file in the person's folder (e.g., vivek.npy)"""
        if embeddings:
            arr = np.stack(embeddings)
            emb_path = os.path.join(person_dir, f"{person_name}.npy")
            np.save(emb_path, arr)

    def load_faces_from_directory(self):
        """Load and process all face images from the imgs directory structure"""
        if not os.path.exists(self.imgs_dir):
            print(f"⚠️ Images directory '{self.imgs_dir}' not found!")
            return
        
        print(f"📁 Loading faces from directory: {self.imgs_dir}")
        
        # Supported image extensions
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        
        # Process each person's folder
        person_folders = [d for d in os.listdir(self.imgs_dir) 
                         if os.path.isdir(os.path.join(self.imgs_dir, d))]
        
        total_faces_loaded = 0
        
        for person_name in person_folders:
            person_dir = os.path.join(self.imgs_dir, person_name)
            person_embeddings = []
            
            print(f"  Processing {person_name}...")
            
            # Try to load existing single .npy embeddings (e.g., vivek.npy)
            emb_file = os.path.join(person_dir, f"{person_name}.npy")
            if os.path.exists(emb_file):
                try:
                    arr = np.load(emb_file)
                    if arr.ndim == 1:
                        person_embeddings = [arr]
                    else:
                        person_embeddings = [arr[i] for i in range(arr.shape[0])]
                except Exception as e:
                    print(f"    ⚠️ Failed to load embedding {emb_file}: {e}")
            
            if not person_embeddings:
                print(f"    ℹ️ No embeddings found, creating from images...")
                # Get all image files in person's directory
                image_files = []
                for ext in image_extensions:
                    image_files.extend(glob.glob(os.path.join(person_dir, ext)))
                    image_files.extend(glob.glob(os.path.join(person_dir, ext.upper())))
                
                processed_images = 0
                for img_path in image_files:
                    try:
                        # Load and process image
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"    ⚠️ Failed to load: {os.path.basename(img_path)}")
                            continue
                        
                        # Detect faces in image
                        faces = self.app.get(img)
                        
                        if len(faces) == 0:
                            print(f"    ⚠️ No face detected in: {os.path.basename(img_path)}")
                            continue
                        elif len(faces) > 1:
                            print(f"    ⚠️ Multiple faces detected in: {os.path.basename(img_path)}, using largest")
                            # Use the largest face (most likely the main subject)
                            faces = [max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))]
                        
                        # Extract embedding
                        face = faces[0]
                        if face.det_score >= self.confidence_threshold:
                            person_embeddings.append(face.embedding)
                            processed_images += 1
                            print(f"    ✅ Processed: {os.path.basename(img_path)}")
                        else:
                            print(f"    ⚠️ Low confidence in: {os.path.basename(img_path)}")
                        
                    except Exception as e:
                        print(f"    ❌ Error processing {os.path.basename(img_path)}: {e}")
                
                if person_embeddings:
                    self.save_person_embeddings(person_name, person_embeddings, person_dir)
                    print(f"    💾 Saved {len(person_embeddings)} embeddings for {person_name} in {person_name}.npy.")
            
            if person_embeddings:
                self.known_faces[person_name] = person_embeddings
                total_faces_loaded += len(person_embeddings)
                print(f"  ✅ Loaded {len(person_embeddings)} embeddings for {person_name}")
            else:
                print(f"  ❌ No valid embeddings found for {person_name}")
        
        print(f"📊 Total faces loaded: {total_faces_loaded} from {len(self.known_faces)} people")
    
    def cosine_similarity(self, emb1, emb2):
        """Calculate cosine similarity between embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def identify_face_robust(self, embedding):
        """
        Robust face identification using voting system across multiple reference embeddings
        
        Args:
            embedding: Face embedding to identify
            
        Returns:
            tuple: (name, confidence_score, vote_details) or (None, 0, {}) if no match
        """
        if not self.known_faces:
            return None, 0, {}
        
        # Vote counting for each person
        person_votes = defaultdict(list)  # {person_name: [similarity_scores]}
        
        # Compare against all embeddings of all known people
        for person_name, person_embeddings in self.known_faces.items():
            for ref_embedding in person_embeddings:
                similarity = self.cosine_similarity(embedding, ref_embedding)
                if similarity > self.vote_threshold:
                    person_votes[person_name].append(similarity)
        
        # Determine best match based on voting
        best_match = None
        best_confidence = 0
        vote_details = {}
        
        for person_name, similarities in person_votes.items():
            if len(similarities) >= self.min_votes:
                # Calculate confidence as average of top similarities
                top_similarities = sorted(similarities, reverse=True)[:3]  # Top 3 matches
                avg_confidence = np.mean(top_similarities)
                
                vote_details[person_name] = {
                    'votes': len(similarities),
                    'avg_confidence': avg_confidence,
                    'max_confidence': max(similarities),
                    'top_similarities': top_similarities
                }
                
                if avg_confidence > best_confidence and avg_confidence > self.similarity_threshold:
                    best_confidence = avg_confidence
                    best_match = person_name
        
        return best_match, best_confidence, vote_details
    
    def draw_face_info(self, img, face, name=None, confidence=0, vote_info=None):
        """Draw bounding box and information on face"""
        bbox = face.bbox.astype(int)
        
        # Choose color based on identification
        if name:
            color = (0, 255, 0)  # Green for known faces
            label = f"{name} ({confidence:.2f})"
        else:
            color = (0, 0, 255)  # Red for unknown faces
            label = "Unknown"
        
        # Draw bounding box
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img, 
                     (bbox[0], bbox[1] - label_size[1] - 10),
                     (bbox[0] + label_size[0], bbox[1]), 
                     color, -1)
        
        # Draw label text
        cv2.putText(img, label, 
                   (bbox[0], bbox[1] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw additional info
        confidence_text = f"Det: {face.det_score:.2f}"
        cv2.putText(img, confidence_text,
                   (bbox[0], bbox[3] + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw vote information if available
        if vote_info and name and name in vote_info:
            votes = vote_info[name]['votes']
            vote_text = f"Votes: {votes}"
            cv2.putText(img, vote_text,
                       (bbox[0], bbox[3] + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def process_frame(self, frame):
        """
        Process a single frame for face detection and identification
        
        Args:
            frame: Input video frame
            
        Returns:
            processed_frame: Frame with face annotations
        """
        # Detect faces
        faces = self.app.get(frame)
        
        # Process each detected face
        for face in faces:
            if face.det_score < self.confidence_threshold:
                continue
            
            # Get face embedding
            embedding = face.embedding
            
            # Identify face using robust method
            name, confidence, vote_details = self.identify_face_robust(embedding)
            
            # Draw face information
            self.draw_face_info(frame, face, name, confidence, vote_details)
            
            # Update tracking history
            if name:
                self.track_history[name].append(time.time())
        
        return frame
    
    def draw_stats(self, frame):
        """Draw statistics on frame"""
        height, width = frame.shape[:2]
        
        # Background for stats
        cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
        
        # Calculate total embeddings
        total_embeddings = sum(len(embeddings) for embeddings in self.known_faces.values())
        
        # Statistics text
        stats = [
            f"Known People: {len(self.known_faces)}",
            f"Total Embeddings: {total_embeddings}",
            f"Min Votes: {self.min_votes}",
            f"Similarity Threshold: {self.similarity_threshold:.2f}",
            f"Vote Threshold: {self.vote_threshold:.2f}"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (15, 30 + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def reload_faces(self):
        """Reload faces from directory"""
        print("🔄 Reloading faces from directory...")
        self.known_faces.clear()
        self.load_faces_from_directory()
    
    def run_webcam(self, camera_id=0):
        """
        Run real-time face tracking from webcam
        
        Args:
            camera_id: Camera device ID (usually 0 for default camera)
        """
        print(f"📹 Starting webcam feed (Camera {camera_id})...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Reload faces from directory")
        print("  's' - Save current database")
        print("  '+' - Increase similarity threshold")
        print("  '-' - Decrease similarity threshold")
        print("  'v' - Adjust vote threshold")
        print("  'm' - Adjust minimum votes required")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Failed to open camera {camera_id}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera opened successfully!")
        print("🎥 Starting real-time face tracking...")
        
        frame_count = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Draw statistics
            self.draw_stats(processed_frame)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps_end_time = time.time()
                fps = 30 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                
            if frame_count > 30:
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                           (processed_frame.shape[1] - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Pose-Invariant Face Tracking', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reload_faces()
            elif key == ord('s'):
                self.save_face_database()
            elif key == ord('+'):
                self.similarity_threshold = min(1.0, self.similarity_threshold + 0.05)
                print(f"📈 Similarity threshold: {self.similarity_threshold:.2f}")
            elif key == ord('-'):
                self.similarity_threshold = max(0.1, self.similarity_threshold - 0.05)
                print(f"📉 Similarity threshold: {self.similarity_threshold:.2f}")
            elif key == ord('v'):
                self.vote_threshold = max(0.1, min(0.8, self.vote_threshold + 0.05))
                print(f"🗳️ Vote threshold: {self.vote_threshold:.2f}")
            elif key == ord('m'):
                self.min_votes = max(1, min(10, self.min_votes + 1))
                print(f"📊 Minimum votes: {self.min_votes}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Real-time tracking stopped")
    
    def run_video_file(self, video_path):
        """
        Run face tracking on video file
        
        Args:
            video_path: Path to video file
        """
        print(f"🎬 Processing video file: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Failed to open video file: {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📊 Video info: {total_frames} frames, {fps:.1f} FPS")
        
        frame_count = 0
        start_time = time.time()

        
        while True:
            ret, frame = cap.read()


            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Draw progress
            progress = frame_count / total_frames
            cv2.rectangle(processed_frame, (10, processed_frame.shape[0] - 30), 
                         (int(processed_frame.shape[1] * progress), processed_frame.shape[0] - 10),
                         (0, 255, 0), -1)
            
            cv2.putText(processed_frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, processed_frame.shape[0] - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Video Face Tracking', processed_frame)
            
            # Control playback speed
            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break
            
            frame_count += 1

        end_time = time.time()
        elapsed = end_time - start_time
        processing_fps = frame_count / elapsed

        print(f"⏱️ Total processing time: {elapsed:.2f} seconds")
        print(f"🚀 Effective processing FPS: {processing_fps:.2f}")

        
        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Video processing completed: {frame_count} frames processed")


def main():
    parser = argparse.ArgumentParser(description='Pose-Invariant Face Tracking & Identification')
    parser.add_argument('--mode', choices=['webcam', 'video'], default='webcam',
                       help='Run mode: webcam or video file')
    parser.add_argument('--video', help='Path to video file (for video mode)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--model', choices=['buffalo_l', 'buffalo_m', 'buffalo_s'], 
                       default='buffalo_l', help='InsightFace model')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Face detection confidence threshold')
    parser.add_argument('--imgs-dir', default='imgs', help='Directory containing person folders')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = PoseInvariantFaceTracker(
        model_name=args.model,
        confidence_threshold=args.confidence,
        imgs_dir=args.imgs_dir
    )
    
    # Run based on mode
    if args.mode == 'webcam':
        tracker.run_webcam(camera_id=args.camera)
    elif args.mode == 'video':
        if not args.video:
            print("❌ Video file path required for video mode")
            return
        tracker.run_video_file(args.video)


if __name__ == "__main__":
    main()