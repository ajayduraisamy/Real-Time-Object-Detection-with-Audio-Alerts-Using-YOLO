import cv2
import pyttsx3
import time
import threading


net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

layer_names = net.getLayerNames()
output_layer_indices = net.getUnconnectedOutLayers().flatten()
output_layers = [layer_names[i - 1] for i in output_layer_indices]


engine = pyttsx3.init()

def speak_async(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

def get_position(center_x, frame_width):
    """Determine Left, Center, Right based on x position."""
    third_width = frame_width / 3
    if center_x < third_width:
        return "Left"
    elif center_x > 2 * third_width:
        return "Right"
    else:
        return "Center"

def detect_objects():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Running object detection... Press ESC to quit")

    last_message = ""
    last_speak_time = 0
    speak_interval = 3  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids, confidences, boxes = [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = int(scores.argmax())
                confidence = scores[class_id]
                if confidence > 0.4:
                    center_x, center_y, w, h = (detection[0:4] * 
                                                [width, height, width, height]).astype('int')
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        current_message = ""

        if len(indexes) > 0:
            i = indexes.flatten()[0]
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            center_x = x + w // 2
            position = get_position(center_x, width)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({position})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            current_message = f"{label} on the {position}"

        
        now = time.time()
        if current_message:
            if (current_message != last_message) or (now - last_speak_time > speak_interval):
                print(f"Speaking: {current_message}")
                speak_async(current_message)  
                last_message = current_message
                last_speak_time = now
        else:
            last_message = "" 

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_objects()
