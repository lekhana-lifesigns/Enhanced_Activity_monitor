import cv2


class ICUMonitorDisplay:
    """
    ICU Display Handler:
    - Live camera
    - Bounding box
    - Skeleton overlay
    - Color coded alerts
    """

    COLOR_MAP = {
        "normal":     (0, 255, 0),     # green
        "agitation":  (0, 165, 255),   # orange
        "fall":       (0, 0, 255),     # red
        "seizure":    (128, 0, 128),   # purple
        "faint":      (255, 0, 0),     # blue
        "unknown":    (200, 200, 200)  # gray
    }

    def __init__(self, title="ICU Live Monitor"):
        self.title = title
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)

    def draw_bbox(self, frame, bbox, label="unknown"):
        if bbox is None:
            return frame

        x, y, w, h = bbox
        color = self.COLOR_MAP.get(label, (255, 255, 255))

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            label.upper(),
            (x, max(25, y-8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        return frame

    def draw_skeleton(self, frame, keypoints):
        if keypoints is None:
            return frame

        for kp in keypoints:
            if kp is None:
                continue
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)  # red joints
        return frame

    def draw_metrics(self, frame, metrics: dict):
        y = 30
        for k, v in metrics.items():
            cv2.putText(
                frame,
                f"{k}: {v}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )
            y += 22
        return frame

    def show(self, frame):
        cv2.imshow(self.title, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True

    def close(self):
        cv2.destroyAllWindows()
