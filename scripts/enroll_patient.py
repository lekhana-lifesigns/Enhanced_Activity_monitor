# scripts/enroll_patient.py
"""
Patient enrollment script.
Captures reference face image for patient verification.
"""
import sys
import os
import cv2
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.patient.face_recognition import PatientFaceRecognizer
from pipeline.pose.camera import Camera

def main():
    parser = argparse.ArgumentParser(description="Enroll patient for face recognition")
    parser.add_argument("--patient-id", required=True, help="Patient ID (e.g., patient_01)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--config", default="config/system.yaml", help="Config file path")
    args = parser.parse_args()
    
    # Load config
    try:
        cfg = yaml.safe_load(open(args.config))
    except Exception as e:
        print(f"Failed to load config: {e}")
        return 1
    
    # Initialize camera
    camres = tuple(cfg.get("camera_resolution", (1280, 720)))
    fps = cfg.get("camera_fps", 15)
    camera = Camera(index=args.camera, resolution=camres, fps=fps)
    
    # Initialize face recognizer
    face_recognizer = PatientFaceRecognizer(
        reference_faces_dir=cfg.get("patient_faces_dir", "storage/patient_faces"),
        model_name=cfg.get("face_model", "VGG-Face")
    )
    
    if not face_recognizer.enabled:
        print("ERROR: Face recognition not available. Install DeepFace:")
        print("  pip install deepface")
        return 1
    
    print(f"\n{'='*60}")
    print(f"PATIENT ENROLLMENT: {args.patient_id}")
    print(f"{'='*60}")
    print("\nInstructions:")
    print("1. Position patient's face clearly in frame")
    print("2. Press SPACE to capture")
    print("3. Press 'q' to quit")
    print(f"{'='*60}\n")
    
    frame_count = 0
    while True:
        frame = camera.read()
        if frame is None:
            continue
        
        frame_count += 1
        
        # Display frame
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw center guide
        cv2.rectangle(display_frame, 
                     (w//4, h//4), 
                     (3*w//4, 3*h//4), 
                     (0, 255, 0), 2)
        cv2.putText(display_frame, "Position face in green box", 
                   (w//4, h//4 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(display_frame, f"Patient ID: {args.patient_id}", 
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, "SPACE: Capture | Q: Quit", 
                   (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Patient Enrollment", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space to capture
            # Use center region as face bbox
            bbox = [w//4, h//4, w//2, h//2]
            
            if face_recognizer.enroll_patient(frame, bbox, args.patient_id):
                print(f"\n✓ Patient {args.patient_id} enrolled successfully!")
                print(f"  Reference face saved to: {face_recognizer.reference_faces[args.patient_id]}")
                break
            else:
                print(f"\n✗ Enrollment failed. Try again.")
        
        elif key == ord('q'):
            print("\nEnrollment cancelled.")
            break
    
    camera.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    sys.exit(main())

