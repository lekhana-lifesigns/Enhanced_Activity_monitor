def draw_overlay(self, frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Background panels
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, h - 100), (w, h), (0, 0, 0), -1)

    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    current_time = self.get_current_time()
    cv2.putText(
        frame,
        f"Time: {current_time:.2f}s / {self.duration:.2f}s",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    status = "PLAYING" if self.playing else "PAUSED"
    color = (0, 255, 0) if self.playing else (0, 0, 255)
    cv2.putText(frame, status, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if self.segment_start is not None:
        seg_text = f"Segment: {self.segment_start:.2f}s → "
        seg_text += f"{self.segment_end:.2f}s" if self.segment_end else "..."
        cv2.putText(frame, seg_text, (300, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if self.current_label:
        cv2.putText(frame, f"Label: {self.current_label}",
                    (300, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Timeline
    bar_y, bar_h = h - 80, 20
    cv2.rectangle(frame, (20, bar_y), (w - 20, bar_y + bar_h),
                  (100, 100, 100), -1)

    for ann in self.annotations:
        x1 = int(20 + (ann['start_time'] / self.duration) * (w - 40))
        x2 = int(20 + (ann['end_time'] / self.duration) * (w - 40))
        cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h),
                      (0, 200, 0), -1)

    pos_x = int(20 + (current_time / self.duration) * (w - 40))
    cv2.line(frame, (pos_x, bar_y - 5),
             (pos_x, bar_y + bar_h + 5), (255, 0, 0), 2)

    cv2.putText(frame,
                "SPACE:Play | S:Start | E:End | 1-9:Label | Q:Quit",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1)

    return frame
