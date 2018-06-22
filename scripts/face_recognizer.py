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
from rospy.numpy_msg import numpy_msg
from face_detection.msg import featArr,classArr,classdata, featdata

class face_recognizer:

    def __init__(self):
      print("Initializing")

      self.class_pub = rospy.Publisher("final_result",classArr,queue_size = 0)
      print("Here1")
      self.bridge = CvBridge()
      self.feat_sub = rospy.Subscriber("extracted_features",featArr,self.callback)
      print("Here2")
      self.counter = 0

    def checkClass(self,centroid_list, entry):
    
      list_shape = centroid_list.shape
      classes = list_shape[0]
      #print('the list shape',entry.shape)
      collector = np.zeros((classes,1), dtype= float)
      entryval = np.array(entry, dtype=np.float).reshape(256,1)

      for i in range(classes):
        #print(i)
        collector[i] = np.linalg.norm(centroid_list[i,0:256,:]-entryval)

    	t = centroid_list[i,0:256,:] - entryval
	
	#print('collecotr',collector)
	#print('entryshape',entry.shape)

      return np.argmin(collector) , collector
    
    
    def callback(self,data):
      print("inside callback")
      final_class = ''
      centroid_info_dir = '/home/abhisek/Study/Robotics/face_data/centroid_info'
      file_name = 'centroid_info_8py.npy'
      print("THIS IS THE dATA",data.data)
      print(type(data))
      #print(data.data.shape)
      #data = np.asarray(data.data).astype(float)
      #print (data.data"
      try:
        centroid_info = np.load(os.path.join(centroid_info_dir,file_name))
      except:
        print("Could not load the centroid_info array.")

      print ("this is the centroid info")
      #print (centroid_info[:,:,:])
      #print (type(data.data))
      #print (data.data.shape)
      #no_of_feats_read = data.data.shape[0]/256
      #print(no_of_feats_read)
      classInfoArray = classArr()
      classInfoArray.header.stamp = rospy.Time.now()
      for i in range(len(data.data)):
	classInfo = classdata()
      	cl2 , collector = self.checkClass(centroid_info,data.data[i].feature.data)
	classInfo.classval = str(cl2+1)
	classInfo.left = data.data[i].left
	classInfo.top = data.data[i].top
	classInfo.ht = data.data[i].ht
	classInfo.wd = data.data[i].wd
      	final_class = final_class+str(cl2+1)+','
	print(final_class)
	classInfoArray.data.append(classInfo)
      #print("This is the final class")
      print(collector)
      try:
        self.class_pub.publish(classInfoArray)
      except rospy.ROSInterruptException:
        pass





def main(args):

    print("STRT")
    rg = face_recognizer()
    rospy.init_node('face_recognizer', anonymous=True)
    print("And also here")
    try:
      print("TRying to spin")
      rospy.spin()
    except KeyboardInterrupt:
      print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
