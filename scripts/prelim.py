#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from std_msgs.msg import String
from face_detection.msg import faceArr, facedata
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:


  
  def __init__(self):
    self.image_pub = rospy.Publisher("cropped_face",faceArr, queue_size=10)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("image_raw",Image,self.callback)
    self.counter = 0

  def rect_to_bb(self,rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect[1].left()
    y = rect[1].top()
    w = rect[1].right() - x
    h = rect[1].bottom() - y
 
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


  def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
 
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
      coords[i] = (shape.part(i).x, shape.part(i).y)
 
    # return the list of (x, y)-coordinates
    return coords
  
  def callback(self,data):


    print(self.counter)
    flag = False
    self.counter = self.counter+1
    if self.counter%10==0:
      try:
	pad = 30
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
	
        detector = dlib.get_frontal_face_detector()
        cv_image = imutils.resize(cv_image, height= 300 , width = 400)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = gray

        rects = detector(gray,1)
        print(len(rects))
	numFaces = []
        for rect in enumerate(rects):
	  flag=True
	  outputval = facedata()
          (x,y,w,h) = self.rect_to_bb(rect)
	  outputval.left = x
	  outputval.top = y
	  outputval.ht = h
	  outputval.wd = w
	  print("WW",w)
          cv2.rectangle(cv_image,(x-pad,y-pad),(x+w+pad,y+h+pad),(0,255,0),2)
	  cropped_face = cv_image[y-pad:y+h+pad,x-pad:x+w+pad]
	  resized_face = imutils.resize(cropped_face, height = 140 , width = 140)

	  print(resized_face.shape)
	  print(outputval)
	  outputval.face = self.bridge.cv2_to_imgmsg(resized_face, "8UC1")

	  numFaces.append(outputval)
	  #self.image_pub.publish(outputval)
	  #self.image_pub.publish(self.bridge.cv2_to_imgmsg(resized_face, "8UC1"))
	self.image_pub.publish(numFaces)
	#print(numFaces)
	cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

      except CvBridgeError as e:
        print(e)






def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
