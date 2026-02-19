import cv2

# 0 is usually the default built-in webcam
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera Test Started. Controls:")
print(" - Press 's' to save a snapshot")
print(" - Press 'q' to quit")

img_counter = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break

    # Display the live feed
    cv2.imshow('Camera Test', frame)

    # Listen for key presses
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        # Quit the program
        break
    elif key == ord('s'):
        # Save the current frame as an image
        img_name = f"snapshot_{img_counter}.png"
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        img_counter += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
