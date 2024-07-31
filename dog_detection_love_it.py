import cv2
import numpy as np
import os
import sys

class DogDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.net = None
        self.classes = []
        self.output_layers = []
        self.cap = None
        self.out = None
        self.frame_count = 0

    def check_file_exists(self, file_path):
        if not os.path.exists(file_path):
            print(f"Error: File does not exist: {file_path}")
            return False
        return True

    def load_yolo(self):
        if not self.check_file_exists("yolov3.weights") or not self.check_file_exists("yolov3.cfg") or not self.check_file_exists("coco.names"):
            return False

        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        layer_names = self.net.getLayerNames()
        unconnected_layers = self.net.getUnconnectedOutLayers()
        
        if isinstance(unconnected_layers, np.ndarray):
            self.output_layers = [layer_names[i - 1] for i in unconnected_layers.flatten()]
        else:
            self.output_layers = [layer_names[i[0] - 1] for i in unconnected_layers]
        
        print(f"Output layers: {self.output_layers}")
        return True

    def open_video(self):
        if not self.check_file_exists(self.video_path):
            return False

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return False

        return True

    def setup_output_video(self):
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        print(f"Video properties: {frame_width}x{frame_height} @ {fps}fps")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))
        print(f"VideoWriter object created: {type(self.out)}")

    def process_video(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print(f"End of video or error reading frame at frame {self.frame_count}")
                break

            self.frame_count += 1

            if frame is None or frame.size == 0:
                print(f"Empty frame encountered at frame {self.frame_count}")
                continue

            height, width, channels = frame.shape
            print(f"Frame {self.frame_count} shape: {width}x{height}")

            try:
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                self.net.setInput(blob)
                outs = self.net.forward(self.output_layers)
            except cv2.error as e:
                print(f"OpenCV error processing frame {self.frame_count}: {str(e)}")
                continue

            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 and self.classes[class_id] == "dog":
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(self.classes[class_ids[i]])
                    confidence = confidences[i]
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

            print(f"Before writing frame {self.frame_count}, out is of type: {type(self.out)}")
            try:
                cv2.imshow("Dog Detection", frame)
                self.out.write(frame)
            except Exception as e:
                print(f"Error displaying or writing frame {self.frame_count}: {str(e)}")
                print(f"out is of type: {type(self.out)}")
                print(f"frame is of type: {type(frame)}")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def run(self):
        if not self.load_yolo():
            return
        if not self.open_video():
            return
        self.setup_output_video()
        self.process_video()
        self.cleanup()

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()
        print(f"Processed {self.frame_count} frames")

# Main execution
if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    
    detector = DogDetector("find_dog.mp4")
    detector.run()
