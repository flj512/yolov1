import cv2

from inference import YOLOPredictor


def run_webcam_demo(checkpoint_path, camera_id=0):
    # Initialize predictor
    predictor = YOLOPredictor(
        checkpoint_path=checkpoint_path, conf_threshold=0.5, nms_threshold=0.4
    )

    # Initialize webcam
    cap = cv2.VideoCapture(camera_id)

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get predictions
        result_frame, boxes = predictor.predict_image(frame)

        # Display frame
        cv2.imshow("YOLOv1 Webcam Demo", result_frame)

        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    checkpoint_path = "path/to/your/checkpoint.pth"
    run_webcam_demo(checkpoint_path)
