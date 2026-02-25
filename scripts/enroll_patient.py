# scripts/enroll_patient.py
"""
Guided patient enrollment with face oval guide and distance feedback.
Automatically captures when patient is correctly positioned.

Usage:
    python scripts/enroll_patient.py --patient-id patient_01
    python scripts/enroll_patient.py --patient-id patient_01 --camera 1
"""
import sys
import os
import cv2
import time
import math
import argparse
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.patient.face_recognition import PatientFaceRecognizer
from pipeline.pose.camera import Camera
from pipeline.display.display import ICUMonitorDisplay

# ── Constants ─────────────────────────────────────────────────────────
KNOWN_FACE_WIDTH_CM = 16.0       # Average human face width
HORIZONTAL_FOV_DEG = 60.0        # Typical webcam horizontal FOV
ENROLL_DIST_MIN_CM = 150.0       # Minimum enrollment distance
ENROLL_DIST_MAX_CM = 250.0       # Maximum enrollment distance
STABILITY_REQUIRED_S = 0.5       # Face must be stable this long before countdown
COUNTDOWN_SECONDS = 3            # Countdown duration before auto-capture


def estimate_focal_length(frame_width):
    """Estimate focal length in pixels from frame width and assumed FOV."""
    return frame_width / (2.0 * math.tan(math.radians(HORIZONTAL_FOV_DEG / 2.0)))


def estimate_distance_cm(face_width_px, focal_length_px):
    """Estimate face distance using pinhole camera model."""
    if face_width_px <= 0:
        return 0.0
    return (KNOWN_FACE_WIDTH_CM * focal_length_px) / face_width_px


def is_face_in_oval(face_cx, face_cy, oval_cx, oval_cy, oval_w, oval_h):
    """Check if face center is inside the oval guide region."""
    dx = (face_cx - oval_cx) / oval_w
    dy = (face_cy - oval_cy) / oval_h
    return (dx * dx + dy * dy) <= 1.0


def main():
    parser = argparse.ArgumentParser(description="Guided patient face enrollment")
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

    # Initialize components
    cam_res = tuple(cfg.get("camera_resolution", (1280, 720)))
    fps = cfg.get("camera_fps", 15)
    camera_url = cfg.get("camera_url")
    camera = Camera(
        index=args.camera, resolution=cam_res, fps=fps,
        use_gstreamer=cfg.get("use_gstreamer", False),
        flip_method=cfg.get("camera_flip_method", 0),
        url=camera_url,
    )

    face_recognizer = PatientFaceRecognizer(
        reference_faces_dir=cfg.get("patient_faces_dir", "storage/patient_faces"),
        model_name=cfg.get("face_model", "VGG-Face")
    )
    if not face_recognizer.enabled:
        print("ERROR: Face recognition not available. Install DeepFace:")
        print("  pip install deepface")
        return 1

    display = ICUMonitorDisplay(title="Patient Enrollment")

    # OpenCV face detector (built-in, no extra deps)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Focal length estimate (computed once from frame width)
    focal_length = estimate_focal_length(cam_res[0])

    # ── State machine ─────────────────────────────────────────────────
    state = "POSITIONING"
    stable_since = 0.0           # Timestamp when face became stable in oval
    countdown_start = 0.0        # Timestamp when countdown started
    capture_frame = None         # Frame to use for enrollment
    capture_bbox = None          # Face bbox at capture time
    result_time = 0.0            # Timestamp of result display

    print(f"\n{'=' * 60}")
    print(f"  PATIENT ENROLLMENT: {args.patient_id}")
    print(f"  Position face in the oval guide at 150-250cm distance")
    print(f"  Auto-captures after {COUNTDOWN_SECONDS}s countdown")
    print(f"{'=' * 60}\n")

    while True:
        frame = camera.read()
        if frame is None:
            continue

        h, w = frame.shape[:2]
        now = time.time()

        # Oval guide dimensions (must match display.py)
        oval_cx, oval_cy = w // 2, h // 2
        oval_w = int(w * 0.13)
        oval_h = int(h * 0.22)

        distance_cm = 0.0
        face_in_oval = False
        countdown = 0

        if state in ("POSITIONING", "FACE_ALIGNED", "COUNTDOWN"):
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            if len(faces) > 0:
                # Pick the largest face (closest / most prominent)
                largest = max(faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = largest
                face_cx = fx + fw // 2
                face_cy = fy + fh // 2

                # Estimate distance
                distance_cm = estimate_distance_cm(fw, focal_length)

                # Check if face is in the oval
                face_in_oval = is_face_in_oval(face_cx, face_cy, oval_cx, oval_cy,
                                                oval_w, oval_h)

                # Check distance is in range
                dist_ok = ENROLL_DIST_MIN_CM <= distance_cm <= ENROLL_DIST_MAX_CM

                if face_in_oval and dist_ok:
                    if state == "POSITIONING":
                        state = "FACE_ALIGNED"
                        stable_since = now

                    elif state == "FACE_ALIGNED":
                        # Check stability duration
                        if (now - stable_since) >= STABILITY_REQUIRED_S:
                            state = "COUNTDOWN"
                            countdown_start = now

                    elif state == "COUNTDOWN":
                        elapsed = now - countdown_start
                        countdown = max(0, COUNTDOWN_SECONDS - int(elapsed))
                        if elapsed >= COUNTDOWN_SECONDS:
                            # Auto-capture
                            state = "CAPTURING"
                            capture_frame = frame.copy()
                            capture_bbox = [fx, fy, fx + fw, fy + fh]
                else:
                    # Face not aligned or distance wrong — reset
                    if state in ("FACE_ALIGNED", "COUNTDOWN"):
                        state = "POSITIONING"
                        stable_since = 0.0

                # Draw face detection rectangle (subtle)
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (200, 200, 200), 1)

            else:
                # No face detected — reset
                if state in ("FACE_ALIGNED", "COUNTDOWN"):
                    state = "POSITIONING"
                    stable_since = 0.0

        elif state == "CAPTURING":
            # Perform enrollment
            success = face_recognizer.enroll_patient(capture_frame, capture_bbox,
                                                      args.patient_id)
            if success:
                state = "DONE"
                print(f"\n  Patient {args.patient_id} enrolled successfully!")
                print(f"  Reference face saved.")
            else:
                state = "FAILED"
                print(f"\n  Enrollment failed. Press R to retry.")
            result_time = now

        elif state == "DONE":
            if (now - result_time) >= 2.0:
                break

        elif state == "FAILED":
            pass  # Wait for key input (R or Q)

        # Draw registration overlay
        display.draw_registration_overlay(frame, state, distance_cm,
                                           face_in_oval, countdown,
                                           args.patient_id)

        # Show frame (waitKey handled manually for key detection)
        cv2.imshow("Patient Enrollment", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nEnrollment cancelled.")
            break
        elif key == ord(' ') and state in ("POSITIONING", "FACE_ALIGNED"):
            # Manual capture override
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            if len(faces) > 0:
                largest = max(faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = largest
                state = "CAPTURING"
                capture_frame = frame.copy()
                capture_bbox = [fx, fy, fx + fw, fy + fh]
            else:
                print("  No face detected for manual capture.")
        elif key == ord('r') and state == "FAILED":
            state = "POSITIONING"
            stable_since = 0.0
            print("  Retrying enrollment...")

    camera.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
