""" Importing the libraries
- BaseTransform: Do the required transformations for the images will be 
compatible with the neural network 
- VOC_CLASSES: Do the Encoding
- build_ssd: Constructor of the SSD neural network
- imageio: Process the images of the video
- labelmap: dictionary that maps the names of the classes with numbers
"""
import torch
from   torch.autograd import Variable
import cv2
from   data           import BaseTransform, VOC_CLASSES as labelmap
from   ssd            import build_ssd
import imageio

def detect(frame, net, transform):
    # Defining a function doing the detection
    
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0] # Get the first element of the function -> Numpy array
    
    """
    - Convert to Torch tensor
    - The neural network SSD was trained under the convention, green, red, blue
    Go from red, blue, green to green, red, blue
    """
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    
    """ 
    - Add fake dimension corresponding to the batch
    - The neural network cannot actually accept single inputs, 
        it only accepts them into some batches of inputs 
    - Use the unsqueeze function to create fake dimension of the batch
    - The Batch should always be the first dimension (index 0) 
    -------------------------------------------------------------------
    Convert this batch of Torch tensor input into the Torch Variable 
    - A torch variable is a highly advanced variable containing both a 
    tensor and a gradient, will become an element of the dynamic graph,
    compute very efficiently the gradient of any composition functions 
    during backward propagation 
    """
    x = Variable(x.unsqueeze(0))
    
    # Feed X to the neural network -> Return output y
    y = net(x)
    
    """
    - Extract the important information that we need
    - Torch variable is composed of two elements a torch tensor (data) and a gradient
    - [batch, #classes detected, #occurrence of the class, 
    (score, x0, y0, x1, y1)]
    - For each occurrence of each class in the batch, we will get a score ( low -> high) 
    for this occurrence. The coordinates of the upper left corner of the rectangle detecting 
    the occurence and the lower right corner of the detected object
    - Score < 0.6: the occurrence of the class won't be found in the image
    """
    detections = y.data
    
    """ 
    - The position of the detected objects inside the image has to be normalized between 
    0 & 1. To do this normalization, need this scale tensor with four dimensions.
    - The first two width height corresponding to the scale of values of the upper left 
    corner of the rectangle detector. Second is the lower right corner
    - Doing this to normalize the scale of values of the position of the detected objects 
    between 0 & 1
    """
    scale = torch.Tensor([width, height, width, height])
    
    """ Iterate through all the classes, through all the occurences of the classes
    - detections.size(1): the number of classes """
    for i in range(detections.size(1)):
        j = 0 # The occurence
        
        """
        - The score of the occurrence j of the class i >= 0.6
        - if this matching score is high enough, we keep it
        - Keep that occurrence by keeping the point (1:)
        - Apply the normalization give us the coordinates of these points 
        at the scale of the image
        - To use OpenCV, put that back into numpy array
        """
        while (detections[0, i,  j, 0] >= 0.6):
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(img=frame, pt1=(int(pt[0]), int(pt[1])), 
                        pt2=(int(pt[2]), int(pt[3])), color=(255, 0, 0), thickness=2)
            cv2.putText(img=frame, text=labelmap[i - 1], org=(int(pt[0]), int(pt[1])), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255),
                        thickness=2, lineType=cv2.LINE_AA)
            j += 1
    return frame

# Creating the SSD neural network
net = build_ssd(phase='test')
""" Load the weights of an already pre trained SSD neural network
- torch.load: open a sensor that contain these weights
- load_state_dict: attribute these weights to our SSD neural network """
net.load_state_dict(torch.load(r'C:\Users\Doan Trong Hieu\Downloads\IMPORTANT\SPECIALIZATION\Artificial_Intelligence\COMPUTER VISION\CODING_COMPUTER_VISION\UDM_Ligency_I_Team_De_Le_an_Co_Vi_A_Z\Module 2 - Object Detection\Code for Windows\ssd300_mAP_77.43_v2.pth', 
                            map_location= lambda storage, loc: storage))

""" Create the transformation
- Right scale: the scale under which the neural network was trained under 
some certain convention"""
transform = BaseTransform(size=net.size, mean=(104/256.0, 117/256.0, 123/256.0))

# Doing some Object Detection on a video
reader = imageio.get_reader(r'C:\Users\Doan Trong Hieu\Downloads\IMPORTANT\SPECIALIZATION\Artificial_Intelligence\COMPUTER VISION\CODING_COMPUTER_VISION\UDM_Ligency_I_Team_De_Le_an_Co_Vi_A_Z\Module 2 - Object Detection\Code for Windows\funny_dog.mp4')
# get the frequency of the frames (frames per second)
fps = reader.get_meta_data()['fps']
# create an output video, object contain a video
writer = imageio.get_writer('output.mp4', fps=fps)

# number of the image that is processed
# iterate through all the frames of the reader video
for i, frame in enumerate(reader):
    """ apply the detect function to our frame with our neural network net 
    and with our transformation """
    frame = detect(frame, net.eval(), transform)
    # append this frame to writer output video
    writer.append_data(frame)
    print(i, end=' - ')
    # close the process manages the creation of this video
writer.close()