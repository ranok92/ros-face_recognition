#!/usr/bin/env python
from __future__ import print_function
import os
import roslib
import sys
import rospy
import cv2
import dlib
import csv
import imutils
import numpy as np
from imutils import face_utils
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats



class face_recognizer:

    def __init__(self):
      print("Initializing")

      self.class_pub = rospy.Publisher("final_result",String, queue_size = 10)
      print("Here1")
      self.bridge = CvBridge()
      self.feat_sub = rospy.Subscriber("extracted_features",numpy_msg(Floats),self.callback)
      print("Here2")
      self.counter = 0

    def checkClass(centroid_list, entry):
    
      axis = 1
      classes = 40
      list_shape = centroid_list.shape
      print(list_shape)
      collector = np.zeros((classes,1), dtype= float)
      for i in range(classes):
        print(i)
        collector[i-1] = np.linalg.norm(centroid_list[i-1,0:256,:]-entry)
    
    
      return np.argmin(collector) , collector
    
    
    def callback(self,data):

      centroid_info_dir = '/home/abhisek/Study/Robotics/face_data/centroid_info'
      file_name = 'centroid_info'
      print(type(data))
      data = np.asarray(data).astype(float)

      try:
        centroid_info = np.load(os.path.join(centroid_info_dir,file_name))
      except:
        print("Could not load the centroid_info array.")


    #sanity check
      print (centroid_info.shape)
      print (data.shape)
      cl2 , collector = self.checkClass(centroid_info,data)
      final_Class = cl2+1
      try:
        self.class_pub.publish(final_Class)
      except rospy.ROSInterruptException:
        pass





def main(args):
  print("STRT")
  rg = face_recognizer()
  rospy.init_node('recognizer', anonymous=True)
  print("And also here")
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
