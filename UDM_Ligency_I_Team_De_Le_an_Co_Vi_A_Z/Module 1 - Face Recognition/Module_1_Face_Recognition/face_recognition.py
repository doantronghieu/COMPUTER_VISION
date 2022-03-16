import cv2
from cv2 import VideoCapture

# Loading the Cascades
# Cascades: Series of filters that will apply one after the other to dectec
#  the face
face_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
eye_cascade  = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Defining a function that will do the detections
def detect(gray, frame):
    """
    - Applied on each image one by one
    faces: x & y Coordinates of the upper left corner of 
    the rectangle that will detect the face; W and H 
    - Get the coordinates of the rectangle that will detect the face
    - The size of the image will be reduced 1.3 times
    - The minimum number of neighbors: 5
    - In order for a zone of pixels to be accepted, 
    at least five neighbor zones must also be accepted
    """
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    """ Each of these faces, draw a rectangle and will detect some eyes """
    for (x, y, w, h) in faces:
        # draw the rectangle
        cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0),
                    thickness=2)
        """ Get two regions of interest 
            - Black & White image 
            - original color image """
        roi_gray  = gray [y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        """ Detecting the eyes in the referential of the face 
        that is the zone inside the rectangle detecting the face """
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        # Draw these rectangles
        for (e_x, e_y, e_w, e_h) in eyes:
        # draw the rectangle
            cv2.rectangle(img=roi_gray, pt1=(e_x, e_y), pt2=(e_x+e_w, e_y+e_h),
                        color=(0, 255, 0), thickness=2)
            
    return frame

# Doing some Face Recognition with the webcam
# Get the last frame from the webcam. Use the internal webcam (0)
video_capture = cv2.VideoCapture(0)
while True:
    # Get the last frame coming from the webcam
    _, frame = video_capture.read()
    # Get the grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    canvas = detect(gray=gray, frame=frame)

    # Display all the successive processed images in a window
    cv2.imshow('Video', canvas)
    
    # Stop the webcam and face detection process
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

# Exactly turn up the webcam
video_capture.release()
cv2.destroyAllWindows()