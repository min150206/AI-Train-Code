def predict_video():
    """Predict on video file with bounding boxes."""
    from ultralytics import YOLO
    import cv2

    WEIGHTS  = "./System/train_yolo11_final/weights/best.pt"
    VIDEO_IN = "./Demo/datatest/demo_video.mp4"
    CONF     = 0.5
    DELAY    = 30    # ← tăng số này để chậm hơn (ms), 1=nhanh nhất, 60=khá chậm

    model = YOLO(WEIGHTS)
    cap   = cv2.VideoCapture(VIDEO_IN)

    # Resize xuống 720p để chạy mượt hơn
    DISPLAY_W = 1920
    DISPLAY_H = 1080

    print("Playing... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize trước khi predict → nhanh hơn nhiều
        frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

        results   = model.predict(frame, conf=CONF, verbose=False)[0]
        annotated = results.plot()

        cv2.imshow("GTSRB Detection", annotated)
        if cv2.waitKey(DELAY) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


predict_video()