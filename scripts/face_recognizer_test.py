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



class face_recognizer_test:

    def __init__(self):
      print("Initializing")

      self.class_pub = rospy.Publisher("final_result",String, queue_size = 10)
      print("Here1")
      self.bridge = CvBridge()
      #self.feat_sub = rospy.Subscriber("extracted_features",Floats,self.callback)
      print("Here2")
      self.counter = 0

    def checkClass(self,centroid_list, entry):
    
      list_shape = centroid_list.shape
      classes = list_shape[0]
      print('the list shape',centroid_list.shape)
      collector = np.zeros((classes,1), dtype= float)
      entry = np.array(entry, dtype=np.float).reshape(256,1)

      for i in range(classes):
        #print(i)
        collector[i] = np.linalg.norm(centroid_list[i,0:256,:]-entry)

    	t = centroid_list[i,0:256,:] - entry
	
	#print('collecotr',collector)
	#print('entryshape',entry.shape)

      return np.argmin(collector) , collector
    
    
    def talker(self):
      print("inside talker")
      centroid_info_dir = '/home/abhisek/Study/Robotics/face_data/centroid_info'
      file_name = 'centroid_info_ros2.npy'
      #data = np.asarray(data.data).astype(float)
      data = np.genfromtxt('/home/abhisek/Study/Robotics/face_data/face_feat_ros/11/3.csv')
      print(data)
      try:
        centroid_info = np.load(os.path.join(centroid_info_dir,file_name))
      except:
        print("Could not load the centroid_info array.")

      print ("this is the centroid info")
      print (centroid_info[:,:,:])
      #print (type(data.data))
      #print (data.data.shape)
      cl2 , collector = self.checkClass(centroid_info,data)
      final_class = cl2+1
      #print("This is the final class")
      print(collector)
      try:
        self.class_pub.publish(str(final_class))
      except rospy.ROSInterruptException:
        pass





def main(args):

    print("STRT")
    rg = face_recognizer_test()
    rospy.init_node('face_recognizer_test', anonymous=True)
    rg.talker()
    print("And also here")
    try:
      print("TRying to spin")
      rospy.spin()
    except KeyboardInterrupt:
      print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
