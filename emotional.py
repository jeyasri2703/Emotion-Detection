from deepface import DeepFace
import cv2

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("✅ Webcam opened")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read camera")
        break

    try:
        # Analyze emotion
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        # Get dominant emotion
        emotion = result[0]['dominant_emotion']

        # Display emotion
        cv2.putText(frame,
                    f"Emotion: {emotion}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

    except Exception as e:
        print("Error:", e)

    # Show webcam
    cv2.imshow("Emotion Detection", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
result = DeepFace.analyze(
    frame,
    actions=['emotion'],
    enforce_detection=False
)