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
import shutil
import traceback
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from imutils import face_utils
from std_msgs.msg import String
from sensor_msgs.msg import Image
from face_detection.msg import classArr, command


class command_test:


    def __init__(self):

        self.feature_pub = rospy.Publisher("command", command, queue_size=10)
	rospy.init_node('command_test',anonymous = True)
        
        
    def publisher(self):
        
	testcomm = command()
	testcomm.order = 'find 41'
	testcomm.header.stamp = rospy.Time.now()
	self.feature_pub.publish(testcomm)
	



def main(args):
    comm = command_test()
    comm.publisher()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    
if __name__=='__main__':
    main(sys.argv)
