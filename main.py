import cv2 as cv  # OpenCV library for image processing
import numpy as np  # Library for numerical computations and arrays
import sys  # For handling program exits and errors


# Close the window when any key is pressed


# Set your paths
photo_address = r'pizza_cat2.jpg'  # Image path
weight_address = r'yolov3.weights'  # Path to YOLOv3 weights file
config_address = r'yolov3.cfg'  # Path to YOLOv3 config file
list_address = r'coco.txt'  # Path to label list


# Set confidence and NMS (Non-Maximum Suppression) thresholds
confidence_threshold = 0.7  # Confidence threshold for object detection
nms_threshold = 0.3  # Threshold for removing overlapping boxes in NMS


# Function to validate the image
def image_validation(img):
   if img is None:  # If the image is not loaded
       print('Invalid Input!')  # Display error message
       sys.exit()  # Exit the program
   else:
       print('Valid Input!')  # If the image is valid, display success message


# Function to resize frames
def rescale_frame(frame, percent=100):
   width = int(frame.shape[1] * percent / 100)  # New width
   height = int(frame.shape[0] * percent / 100)  # New height
   dim = (width, height)  # Set new dimensions
   return cv.resize(frame, dim, interpolation=cv.INTER_AREA)  # Resize the image


# Pre-process the image
def load_data_and_pre_process(img):
   h, w = img.shape[:2]  # Get image dimensions (height and width)
   pre_process_img = cv.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
   # Convert the image into a format suitable for neural network input
   return img, h, w, pre_process_img  # Return image, height, width, and pre-processed image


# Read models and labels
def read_models_and_labels(label_address, weight_address, config_address):
   labels = open(label_address).read().strip().split('\n')  # Read and parse the labels
   net = cv.dnn.readNet(weight_address, config_address)  # Load neural network with weights and config file
   return labels, net  # Return labels and neural network


# Perform inference using the neural network
def inference(pre_process_img, net):
   net.setInput(pre_process_img)  # Set pre-processed image as input to the network
   output_layer = ["yolo_82", "yolo_94", "yolo_106"]  # Select YOLOv3 output layers
   predictions = net.forward(output_layer)  # Get predictions from the network
   return predictions  # Return predictions


# Post-process inference results
def post_processing(predictions, w, h):
   classIDs = []  # List of detected class IDs
   confidences = []  # List of confidence scores
   boxes = []  # List of bounding boxes


   for layer in predictions:  # Process each layer of predictions
       for detected_object in layer:  # Process each detected object
           scores = detected_object[5:]  # Class scores
           classID = np.argmax(scores)  # Choose the class with the highest score
           confidence = scores[classID]  # Get the confidence for that class


           if confidence > confidence_threshold:  # If confidence is above the threshold
               box = detected_object[0:4] * np.array([w, h, w, h])  # Calculate bounding box
               (cx, cy, width, height) = box.astype("int")  # Convert coordinates to integers
               x = int(cx - width / 2)  # Calculate x position
               y = int(cy - height / 2)  # Calculate y position
               classIDs.append(classID)  # Add class ID to list
               confidences.append(float(confidence))  # Add confidence to list
               boxes.append([x, y, int(width), int(height)])  # Add bounding box to list
   return classIDs, confidences, boxes  # Return class IDs, confidences, and bounding boxes


# Display the final result on the image
def show_result(img, classIDs, confidences, boxes, labels):
   idxs = cv.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
   # Apply NMS to remove overlapping boxes
   for i in idxs.flatten():  # Process remaining boxes
       x = boxes[i][0]  # x coordinate
       y = boxes[i][1]  # y coordinate
       w = boxes[i][2]  # Box width
       h = boxes[i][3]  # Box height
       colors = np.random.uniform(0, 255, size=(80, 3))  # Generate random colors for boxes
       cv.rectangle(img, (x, y), (x + w, y + h), colors[i], 1)  # Draw bounding box
       cv.rectangle(img, (x, y - 45), (x + 250, y), colors[i], -1)  # Draw background for text
       text = '{}:{:.2f}'.format(labels[classIDs[i]], confidences[i])  # Display class and confidence
       cv.putText(img, text, (x, y - 10), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)  # Write text on image


# Main part of the program
img = cv.imread(photo_address)  # Read the image from the specified path
image_validation(img)  # Validate the image
img = rescale_frame(img)  # Resize the image
img, h, w, pre_process_img = load_data_and_pre_process(img)  # Pre-process the image
labels, net = read_models_and_labels(list_address, weight_address, config_address)  # Load labels and model
predictions = inference(pre_process_img, net)  # Perform inference
classIDs, confidences, boxes = post_processing(predictions, w, h)  # Post-process results
show_result(img, classIDs, confidences, boxes, labels)  # Display the result on the image
cv.imshow('image', img)  # Show the image
cv.waitKey(0)  # Wait for a key press
cv.destroyAllWindows()  # Close all windows



