import cv2
import mediapipe as mp  # for face detection

# Method to calculate face dimensions
def calculate_face_dimensions(detection):
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = image.shape
    ymin, xmin, h, w = int(bboxC.ymin * ih), int(bboxC.xmin * iw), \
                       int(bboxC.height * ih), int(bboxC.width * iw)
    ymax, xmax = ymin + h, xmin + w
    return (xmin, ymin), (xmax, ymax), (w, h)

# Method to classify face type based on dimensions
def classify_face_type(dimensions):
    width, height = dimensions
    if width >= 130 and width <= 150 and height >= 70 and height <= 120:
        return "Oval Face"
    elif width >= 110 and width <= 135 and height >= 75 and height <= 130:
        return "Round Face"
    elif width >= 115 and width <= 140 and height >= 65 and height <= 120:
        return "Square Face"
    elif width >= 120 and width <= 145 and height >= 80 and height <= 140:
        return "Heart-Shaped Face"
    elif width >= 115 and width <= 145 and height >= 70 and height <= 140:
        return "Diamond Face"
    else:
        return "Undefined Face Type"

# Method to recommend glasses based on face type
def recommend_glasses(face_type):
    glasses_recommendations = {
        "Oval Face": "Rectangular Glasses",
        "Round Face": "Square Glasses",
        "Square Face": "Round Glasses",
        "Heart-Shaped Face": "Aviator Glasses",
        "Diamond Face": "Cat-eye Glasses",
        "Undefined Face Type": "No Recommendation"
    }
    return glasses_recommendations.get(face_type, "No Recommendation")

# Specify the desired distance to start measurement (in pixels)
desired_distance = 0.5  # Adjust this value based on the desired distance

#reading the face from cam or video
mp_face_detection = mp.solutions.face_detection
# draw a rectangle for the face detected
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.4) as face_detection:  # Initialize face detection model
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        # recognize the face
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # change
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
                # Calculate face dimensions
                pt1, pt2, dimensions = calculate_face_dimensions(detection)
                # Classify face type
                face_type = classify_face_type(dimensions)
                # Recommend glasses
                glasses_recommendation = recommend_glasses(face_type)
                # Display dimensions, face type, and glasses recommendation using cv2.putText()
                cv2.putText(image, f'{face_type}\nWidth: {dimensions[0]} Height: {dimensions[1]}\nGlasses: {glasses_recommendation}',
                            (pt1[0], pt1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Check if face is at desired distance to start measurement
                if dimensions[1] == desired_distance:
                    # Start measuring the face dimensions and calculating the distance
                    initial_face_height = dimensions[1]  # Set initial face height
                    observed_distance = desired_distance  # Set observed distance
                    cv2.putText(image, f'Distance: {observed_distance:.2f} (pixels)', (pt1[0], pt1[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('MediaPipe Face Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
