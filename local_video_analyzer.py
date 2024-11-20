import argparse
import cv2 as cv

def display_video(video):
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Process each frame to detect and highlight the slide
        processed_frame = preprocess_frame(frame)

        # Display the processed frame with bounding box
        cv.imshow("Video", processed_frame)
        
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture and close windows
    video.release()
    cv.destroyAllWindows()


def preprocess_frame(frame):
    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv.Canny(gray, 50, 150)

    # Find contours from the edges
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Loop over the contours to find rectangular shapes
    for contour in contours:
        # Approximate the contour to a polygon to simplify its shape
        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        
        # Check if the approximated contour has 4 points, which suggests a rectangle or square
        if len(approx) == 4:
            # Calculate the bounding box
            x, y, w, h = cv.boundingRect(approx)
            
            # Define conditions to identify the slide (adjust based on slide size in frame)
            aspect_ratio = float(w) / h
            if 0.9 < aspect_ratio < 1.1 and w > 100 and h > 100:  # Adjust values as needed
                # Draw a large square around the detected rectangle
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Green square
                break  # Draw the first detected slide to avoid multiple boxes

    return frame


def main(video_name):
    # Load the video
    video = cv.VideoCapture(video_name)
    if not video.isOpened():
        print("Error: Could not open video")
        return
    
    display_video(video)
    print("we are in main")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detects and highlights slides in a video"
    )
    parser.add_argument("video_name", type=str, help="The name of the video file")
    args = parser.parse_args()
    main(args.video_name)
