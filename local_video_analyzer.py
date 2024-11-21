import cv2
import numpy as np
import argparse
import os
import statistics

class PresentationAreaDetector:
    def __init__(self, debug=False):
        """
        Initialize the Presentation Area Detector
        
        Args:
            debug (bool): Enable debug mode for additional visualizations
        """
        self.debug = debug
        self.detection_threshold = {
            'min_area': 10000,  # Minimum contour area
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.5
        }
        # Store contour information across frames
        self.contour_history = {
            'areas': [],
            'bounding_boxes': []
        }

    def detect_presentation_area(self, frame):
        """
        Detect the presentation area in a single frame.
        
        Args:
            frame (numpy.ndarray): Input image frame
        
        Returns:
            tuple: Contour of the presentation area, processed frame with contour, extracted presentation
        """
        # Create a copy of the original frame for drawing
        original = frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding to handle different lighting conditions
        thresh = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours based on area to find the presentation
        presentation_contour = None
        potential_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter contours based on area and aspect ratio
            if area > self.detection_threshold['min_area']:
                # Compute the bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio to filter out unlikely shapes
                aspect_ratio = w / float(h)
                
                if (self.detection_threshold['min_aspect_ratio'] < aspect_ratio < 
                    self.detection_threshold['max_aspect_ratio']):
                    potential_contours.append((contour, area))
        
        # Sort potential contours by area in descending order
        potential_contours.sort(key=lambda x: x[1], reverse=True)
        
        # Select the largest potential contour
        if potential_contours:
            presentation_contour = potential_contours[0][0]
            
            # Update contour history
            x, y, w, h = cv2.boundingRect(presentation_contour)
            self.contour_history['areas'].append(cv2.contourArea(presentation_contour))
            self.contour_history['bounding_boxes'].append((x, y, w, h))
        
        # Draw the contour on the original image
        result_frame = original.copy()
        extracted_presentation = None
        
        if presentation_contour is not None:
            # Draw contour
            cv2.drawContours(
                result_frame, 
                [presentation_contour], 
                -1, 
                (0, 255, 0),  # Green color
                3  # Thickness
            )
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(presentation_contour)
            
            # Extract presentation area
            extracted_presentation = original[y:y+h, x:x+w]
        
        return presentation_contour, result_frame, extracted_presentation

    def process_video(self, video_path):
        """
        Process a video to detect presentation area and create output videos.
        
        Args:
            video_path (str): Path to the input video
        
        Returns:
            dict: Video processing statistics
        """
        # Reset contour history
        self.contour_history = {
            'areas': [],
            'bounding_boxes': []
        }

        # Ensure the video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Open the input video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Prepare output video writers
        output_path_with_contour = 'presentation_area_with_contour.mp4'
        output_path_presentation_only = 'presentation_area_extracted.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Create output directories
        output_dir = 'presentation_analysis'
        os.makedirs(output_dir, exist_ok=True)
        
        # Variables to track video writers
        out_contour = None
        out_presentation = None
        
        # Video processing variables
        frame_count = 0
        detected_frames = 0
        
        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            
            # Break the loop if no more frames
            if not ret:
                break
            
            frame_count += 1
            
            try:
                # Detect presentation area
                contour, processed_frame, extracted_presentation = self.detect_presentation_area(frame)
                
                if contour is not None:
                    detected_frames += 1
                    
                    # Initialize video writers on first detected frame
                    if out_contour is None and extracted_presentation is not None:
                        # Compute the most common or median bounding box
                        stable_boxes = self._get_stable_presentation_box()
                        
                        if stable_boxes:
                            x, y, w, h = stable_boxes
                            # Create video writers with stable dimensions
                            out_contour = cv2.VideoWriter(
                                output_path_with_contour, 
                                fourcc, 
                                fps, 
                                (width, height)
                            )
                            out_presentation = cv2.VideoWriter(
                                output_path_presentation_only, 
                                fourcc, 
                                fps, 
                                (w, h)
                            )
                    
                    # Write frames to output videos if writers exist
                    if out_contour:
                        out_contour.write(processed_frame)
                    
                    if out_presentation:
                        # Extract stable presentation area
                        if stable_boxes:
                            x, y, w, h = stable_boxes
                            extracted_presentation = frame[y:y+h, x:x+w]
                            out_presentation.write(extracted_presentation)
                    
                    # If in debug mode, save individual frames
                    if self.debug:
                        debug_path_contour = os.path.join(
                            output_dir, 
                            f'frame_{frame_count:04d}_with_contour.jpg'
                        )
                        debug_path_extracted = os.path.join(
                            output_dir, 
                            f'frame_{frame_count:04d}_extracted.jpg'
                        )
                        cv2.imwrite(debug_path_contour, processed_frame)
                        if extracted_presentation is not None:
                            cv2.imwrite(debug_path_extracted, extracted_presentation)
                
                # Optional: Display the frame (useful for real-time debugging)
                if self.debug:
                    cv2.imshow('Presentation Area Detection', processed_frame)
                    
                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
        
        # Release resources
        cap.release()
        if out_contour:
            out_contour.release()
        if out_presentation:
            out_presentation.release()
        
        if self.debug:
            cv2.destroyAllWindows()
        
        # Analyze and print contour statistics
        self._print_contour_statistics()
        
        # Print analysis summary
        print(f"\nVideo Analysis Summary:")
        print(f"Input Video: {video_path}")
        print(f"Output Video (with Contour): {output_path_with_contour}")
        print(f"Output Video (Presentation Only): {output_path_presentation_only}")
        print(f"Total Frames: {frame_count}")
        print(f"Frames with Detected Presentation Area: {detected_frames}")
        print(f"Detection Rate: {detected_frames/frame_count*100:.2f}%")
        
        return {
            'input_video': video_path,
            'output_video_with_contour': output_path_with_contour,
            'output_video_presentation_only': output_path_presentation_only,
            'total_frames': frame_count,
            'detected_frames': detected_frames,
            'detection_rate': detected_frames/frame_count*100
        }

    def _get_stable_presentation_box(self, threshold_percentage=0.25):
        """
        Compute a stable presentation area bounding box.
        
        Args:
            threshold_percentage (float): Percentage of frames to consider for stable box
        
        Returns:
            tuple: (x, y, width, height) of the most stable presentation box
        """
        # Ensure we have detected some contours
        if not self.contour_history['bounding_boxes']:
            return None
        
        # Compute the number of frames to consider
        num_frames = len(self.contour_history['bounding_boxes'])
        threshold_frames = max(1, int(num_frames * threshold_percentage))
        
        # Sort bounding boxes by frequency
        from collections import Counter
        box_counter = Counter(self.contour_history['bounding_boxes'])
        
        # Find the most common box that appears at least threshold_frames times
        stable_boxes = [box for box, count in box_counter.items() if count >= threshold_frames]
        
        if stable_boxes:
            # If multiple stable boxes, choose the one that appears most frequently
            return max(stable_boxes, key=box_counter.get)
        
        return None

    def _print_contour_statistics(self):
        """
        Print detailed statistics about detected contours.
        """
        if not self.contour_history['areas']:
            print("\nNo contours detected.")
            return
        
        print("\nContour Statistics:")
        print(f"Total contours detected: {len(self.contour_history['areas'])}")
        
        # Compute basic statistics for areas
        print(f"Contour Areas:")
        print(f"  Minimum Area: {min(self.contour_history['areas']):.2f}")
        print(f"  Maximum Area: {max(self.contour_history['areas']):.2f}")
        print(f"  Mean Area: {statistics.mean(self.contour_history['areas']):.2f}")
        print(f"  Median Area: {statistics.median(self.contour_history['areas']):.2f}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Analyze video to detect and extract presentation area'
    )
    parser.add_argument(
        'video_path', 
        type=str, 
        help='Path to the input video file'
    )
    parser.add_argument(
        '-d', '--debug', 
        action='store_true', 
        help='Enable debug mode with additional visualizations'
    )
    parser.add_argument(
        '-t', '--threshold', 
        type=float, 
        default=10000,
        help='Minimum contour area threshold (default: 10000)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Create detector instance
        detector = PresentationAreaDetector(debug=args.debug)
        
        # Set custom threshold if provided
        detector.detection_threshold['min_area'] = args.threshold
        
        # Process the video
        result = detector.process_video(args.video_path)
    
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()