import cv2
import numpy as np 
import os 
import imutils 

import rospkg

pkg = rospkg.RosPack()
path = pkg.get_path('point_models')
# path_to_video = path + '/videos/bangalore.mp4'
path_to_video = path + '/videos/carla_world.mp4'


# Make sure the video file is in the same directory as your code
filename = path_to_video
file_size = (1920,1080)
 
# We want to save the output to a video file
output_filename = 'semantic_seg_4_orig_lane_detection_1.mp4'
output_frames_per_second = 20.0
 
ENET_DIMENSIONS = (1024, 512) # Dimensions that ENet was trained on
RESIZED_WIDTH = 1200
IMG_NORM_RATIO = 1 / 255.0 # In grayscale a pixel can range between 0 and 255
 
# Load the names of the classes
class_names = (
  open('./enet-cityscapes/enet-classes.txt').read().strip().split("\n"))
     
# Load a list of colors. Each class will have a particular color. 
if os.path.isfile('./enet-cityscapes/enet-colors.txt'):
  IMG_COLOR_LIST = (
    open('./enet-cityscapes/enet-colors.txt').read().strip().split("\n"))
  IMG_COLOR_LIST = [np.array(color.split(",")).astype(
    "int") for color in IMG_COLOR_LIST]
  IMG_COLOR_LIST = np.array(IMG_COLOR_LIST, dtype="uint8")
     
# If the list of colors file does not exist, we generate a 
# random list of colors
else:
  np.random.seed(1)
  IMG_COLOR_LIST = np.random.randint(0, 255, size=(len(class_names) - 1, 3),
    dtype="uint8")
  IMG_COLOR_LIST = np.vstack([[0, 0, 0], IMG_COLOR_LIST]).astype("uint8")
 
def main():
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        # Capture one frame at a time
        success, frame = cap.read() 
            
        # Do we have a video frame? If true, proceed.
        if success:
        # Resize the frame while maintaining the aspect ratio
            frame = imutils.resize(frame, width=RESIZED_WIDTH)
        
            # Create a blob. A blob is a group of connected pixels in a binary 
            # frame that share some common property (e.g. grayscale value)
            # Preprocess the frame to prepare it for deep learning classification
            frame_blob = cv2.dnn.blobFromImage(frame, IMG_NORM_RATIO, ENET_DIMENSIONS, 0, swapRB=True, crop=False)
            
            # Load the neural network (i.e. deep learning model)
            enet_neural_network = cv2.dnn.readNet('./enet-cityscapes/enet-model.net')
        
            # Set the input for the neural network
            enet_neural_network.setInput(frame_blob)
        
            # Get the predicted probabilities for each of 
            # the classes (e.g. car, sidewalk)
            # These are the values in the output layer of the neural network
            enet_neural_network_output = enet_neural_network.forward()
        
            # Extract the key information about the ENet output
            # (number_of_classes, height, width) = (enet_neural_network_output.shape[1:4]) 

            # Find the class label that has the greatest 
            # probability for each frame pixel
            class_map = np.argmax(enet_neural_network_output[0], axis=0)

            # Tie each class ID to its color
            # This mask contains the color for each pixel. 
            class_map_mask = IMG_COLOR_LIST[class_map]
        
            # We now need to resize the class map mask so its dimensions
            # is equivalent to the dimensions of the original frame
            class_map_mask = cv2.resize(class_map_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        
            # Overlay the class map mask on top of the original frame. We want 
            # the mask to be transparent. We can do this by computing a weighted 
            # average of the original frame and the class map mask.
            # enet_neural_network_output = ((0.90 * class_map_mask) + (0.10 * frame)).astype("uint8")
            enet_neural_network_output = ((0.5 * class_map_mask) + (0.5 * frame)).astype("uint8")
            cv2.imshow('Image' , enet_neural_network_output)
            print('reached detection')
            # cv2.waitKey(0)
            # cv2.waitKeyS
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
 
main()