import cv2
import numpy as np
import logging

log = logging.getLogger("display")

class ICUMonitorDisplay:
    def __init__(self, title="ICU Live Monitor"):
        self.title = title
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)

    def draw_metrics(self, frame, metrics: dict):
        y = 30
        for k, v in metrics.items():
            text = f"{k}: {v}"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25
        return frame
    
    def draw_posture_with_timestamp(self, frame, posture_state: str, timestamp: float, 
                                   posture_confidence: float = None):
        """
        Draw posture with timestamp prominently on frame.
        
        Args:
            frame: Input frame (BGR)
            posture_state: Current posture (supine, sitting, etc.)
            timestamp: Unix timestamp
            posture_confidence: Optional confidence score
        
        Returns:
            Frame with posture and timestamp overlay
        """
        try:
            from datetime import datetime
            h, w = frame.shape[:2]
            
            # Format timestamp
            dt = datetime.fromtimestamp(timestamp)
            time_str = dt.strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
            date_str = dt.strftime("%Y-%m-%d")
            
            # Posture text
            posture_text = f"Posture: {posture_state.upper()}"
            if posture_confidence is not None:
                posture_text += f" ({posture_confidence:.2f})"
            
            # Timestamp text
            timestamp_text = f"Time: {date_str} {time_str}"
            
            # Draw background box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            (posture_w, posture_h), _ = cv2.getTextSize(posture_text, font, font_scale, thickness)
            (time_w, time_h), _ = cv2.getTextSize(timestamp_text, font, 0.6, 1)
            
            box_width = max(posture_w, time_w) + 20
            box_height = posture_h + time_h + 30
            box_x = w - box_width - 10
            box_y = 10
            
            # Draw semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw border
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                         (0, 255, 0), 2)
            
            # Draw posture text (larger, prominent)
            cv2.putText(frame, posture_text, (box_x + 10, box_y + 30), 
                       font, font_scale, (0, 255, 0), thickness)
            
            # Draw timestamp text
            cv2.putText(frame, timestamp_text, (box_x + 10, box_y + 55), 
                       font, 0.6, (255, 255, 255), 1)
            
        except Exception as e:
            log.debug("Failed to draw posture with timestamp: %s", e)
        
        return frame

    def draw_skeleton(self, frame, keypoints):
        if not keypoints:
            return frame
        for kp in keypoints:
            if kp and len(kp) >= 2:
                x, y = int(kp[0]), int(kp[1])
                # Use confidence if available (3rd element)
                conf = kp[2] if len(kp) > 2 else 1.0
                # Color intensity based on confidence
                color_intensity = int(255 * conf)
                cv2.circle(frame, (x, y), 4, (0, 0, color_intensity), -1)
        return frame
    
    def draw_bbox(self, frame, bbox, label="", track_id=None, reid_enabled=False, verified=None):
        """
        Draw proper bounding box on frame.
        Supports both [x1, y1, x2, y2] (preferred) and [x, y, w, h] formats.
        """
        if not bbox or len(bbox) < 4:
            return frame
        
        # Handle different bbox formats - prefer (x1, y1, x2, y2)
        a, b, c, d = bbox[:4]
        h, w = frame.shape[:2]
        
        # Check if it's (x1,y1,x2,y2) format (proper format)
        # Improved detection: check if values are reasonable for absolute coordinates
        if (c > a and d > b and 
            (c - a) > 10 and (d - b) > 10 and  # Minimum size
            c <= w and d <= h and  # Within frame bounds
            a >= 0 and b >= 0):  # Non-negative
            # Proper format (x1, y1, x2, y2)
            x1, y1, x2, y2 = int(a), int(b), int(c), int(d)
        else:
            # Likely (x, y, w, h) - convert to (x1, y1, x2, y2)
            x1, y1 = int(a), int(b)
            x2, y2 = int(a + c), int(b + d)
        
        # Clamp to frame bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Draw rectangle with proper format
        color = (0, 255, 0) if reid_enabled else (255, 0, 0)
        thickness = 3  # Thicker for better visibility
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label with track ID and verification status
        label_text = label if label else "person"
        if track_id is not None:
            label_text = f"{label_text} ID:{track_id}"
        if reid_enabled:
            label_text += " [ReID]"
        if verified is not None:
            label_text += f" {'✓' if verified else '✗'}"
        
        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                     (x1 + text_width + 10, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def draw_segmentation(self, frame, mask, alpha=0.3, color=(0, 255, 0)):
        """
        Draw instant segmentation mask overlay on frame.
        
        Args:
            frame: Input frame (BGR)
            mask: Binary mask (H, W) or None
            alpha: Transparency (0-1)
            color: Mask color (B, G, R)
        
        Returns:
            Frame with segmentation overlay
        """
        if mask is None:
            return frame
        
        try:
            # Ensure mask is same size as frame
            h, w = frame.shape[:2]
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Create colored mask
            mask_colored = np.zeros_like(frame)
            mask_colored[mask > 0] = color
            
            # Blend with frame
            frame = cv2.addWeighted(frame, 1.0 - alpha, mask_colored, alpha, 0)
            
            # Draw mask contour for better visibility
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, color, 2)
            
        except Exception as e:
            log.debug("Failed to draw segmentation: %s", e)
        
        return frame

    def draw_distance_feedback(self, frame, distance_feedback: dict):
        """
        Draw distance feedback message on frame.
        
        Args:
            frame: Input frame (BGR)
            distance_feedback: Distance feedback dict with message, status, etc.
        
        Returns:
            Frame with distance feedback overlay
        """
        if not distance_feedback:
            return frame
        
        try:
            h, w = frame.shape[:2]
            message = distance_feedback.get("message", "")
            status = distance_feedback.get("status", "unknown")
            distance_cm = distance_feedback.get("distance_cm", 0)
            target_cm = distance_feedback.get("target_cm", 200)
            
            # Choose color based on status
            if status == "too_close" or status == "too_far":
                color = (0, 0, 255)  # Red - urgent
                bg_color = (0, 0, 200)
            elif status == "close_to_optimal" or status == "far_from_optimal":
                color = (0, 165, 255)  # Orange - warning
                bg_color = (0, 100, 200)
            else:
                color = (0, 255, 0)  # Green - optimal
                bg_color = (0, 200, 0)
            
            # Draw background banner at top
            banner_height = 80
            cv2.rectangle(frame, (0, 0), (w, banner_height), bg_color, -1)
            
            # Draw main message
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Split message into lines if too long
            words = message.split()
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + " " + word if current_line else word
                (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
                if text_width > w - 20:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)
            
            # Draw lines
            y_offset = 30
            for i, line in enumerate(lines[:3]):  # Max 3 lines
                cv2.putText(frame, line, (10, y_offset + i * 25), 
                           font, font_scale, color, thickness)
            
            # Draw distance info
            distance_text = f"Current: {distance_cm}cm | Target: {target_cm}cm"
            cv2.putText(frame, distance_text, (10, banner_height - 10), 
                       font, 0.5, (255, 255, 255), 1)
            
        except Exception as e:
            log.debug("Failed to draw distance feedback: %s", e)
        
        return frame

    def show(self, frame):
        cv2.imshow(self.title, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True

    def close(self):
        cv2.destroyAllWindows()