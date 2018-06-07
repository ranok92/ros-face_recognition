#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
import dlib
import csv
import imutils
import yaml
import math
import numpy as np
from imutils import face_utils
from std_msgs.msg import String
from sensor_msgs.msg import Image
import message_filters
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from ptu_driver.msg import PanTiltStateStamped, PanTilt
from face_detection.msg import featArr,classArr,command
import traceback


class ptu_controls:

	def __init__(self):

		print("Initializing")

		self.class_pub = rospy.Publisher("/ptu_driver/cmd_rel",PanTilt,queue_size = 10)
		self.classifier_info = message_filters.Subscriber("final_result",classArr)
		self.command_info = message_filters.Subscriber("command",command)
		self.ptu_state = message_filters.Subscriber("/ptu_driver/state",PanTiltStateStamped)
		print("Here2")
		self.control = message_filters.ApproximateTimeSynchronizer([self.classifier_info , self.command_info ,self.ptu_state],10,slop=2)
		self.control.registerCallback(self.primaryCallback)
		print("there")	
		self.target = ''
		self.explore_mode = ''
		self.hitCounter = 0
		with open('/home/abhisek/.ros/camera_info/camera.yaml','r') as cam_file:
			calib_data = yaml.load(cam_file)
		self.projection_matrix = np.asarray(calib_data['projection_matrix']['data']).reshape((3,4))
		self.camera_matrix = np.asarray(calib_data['camera_matrix']['data']).reshape((3,3))
		self.focal_length = self.projection_matrix[0][0]
		self.image_size_x = 400
		self.image_size_y = 300
		print(self.projection_matrix)
		print(self.camera_matrix)

	def primaryCallback(self,final_result, command, state):

		print(final_result)
		print(command)
		print(state)
		received_order = str(command.order).strip().split(' ')
		verb = received_order[0]
		self.target = received_order[1]
		if (verb=='find'):

			res = self.finder()
			print ("This is the result obtained",res)

		if verb=='explore':
			self.explorer()

	def finder(self):

		finder_call = message_filters.ApproximateTimeSynchronizer([self.classifier_info , self.ptu_state],10,slop=1)
		result = finder_call.registerCallback(self.findCallback)
		#finder_call.unregister()
 		return result


 	def tracker(self,final_result,state):

 		print("inside tracker")
 		move = PanTilt()
 		c = 203*3.7795275591
 		print(final_result.data)
 		w = c/final_result.data[0].wd*self.focal_length
 		w = w/100
 		print('W : ',w)
 		init_point = np.asarray([final_result.data[0].left*w , final_result.data[0].top*w , w])
 		print("Init_point :" ,init_point)
 		inv_mat = np.linalg.inv(self.camera_matrix)
 		init_point_wc = np.dot(inv_mat,init_point)
 		print("Init_point_wc :",init_point_wc)
 		final_point = np.asarray([(self.image_size_x/2-final_result.data[0].wd/2)*w , init_point[1] , w])
 		final_point_tilt = np.asarray([init_point[0],(self.image_size_y/2-final_result.data[0].ht/2)*w , w])
 		final_point_wc = np.dot(inv_mat,final_point)
 		final_point_wc_tilt = np.dot(inv_mat,final_point_tilt)
 		print("Final_point_wc :",final_point_wc)
 		cos_theta = np.dot(init_point_wc,final_point_wc)/((np.linalg.norm(init_point_wc)*np.linalg.norm(final_point_wc)))
 		cos_theta_tilt = np.dot(init_point_wc,final_point_wc_tilt)/((np.linalg.norm(init_point_wc)*np.linalg.norm(final_point_wc_tilt)))
		pan_value = math.acos(cos_theta)*90/math.pi
		tilt_value = math.acos(cos_theta_tilt)*90/math.pi
 		print(pan_value)
 		try:
 			if init_point[0] > final_point[0]:
 				move.pan = -pan_value
 			else: 
				move.pan = pan_value
			if init_point[1] > final_point_tilt[1]:
 				move.tilt = -tilt_value
 			else:
 				move.tilt = tilt_value
 			self.class_pub.publish(move)
 		except Exception as e:

 			print(traceback.format_exc())
 			print(sys.exc_info()[0])



 	def moveLeft(self):
 		try:
 			move = PanTilt()
 			move.pan = -5
 			move.tilt = 0
 			self.class_pub.publish(move)
 		except Exception as e:

 			print(traceback.format_exc())
 			print(sys.exc_info()[0])


 	def moveRight(self):
 		try:
 			move = PanTilt()
 			move.pan = 5
 			move.tilt = 0
 			self.class_pub.publish(move)
 		except Exception as e:

 			print(traceback.format_exc())
 			print(sys.exc_info()[0])



 	def findCallback(self,final_result, state):
 		self.hitCounter = 0
 		move = PanTilt()
 		foundEm = False
 		left = False
 		for i in final_result.data:

			if (self.target == i.classval):
				print(i.classval)
				foundEm = True
				break

		if foundEm:
			#finetune
			print("found them")
			#return 1
			print(foundEm)
			self.tracker(final_result,state)

		else:
			print("cant see so i should turn")
			if self.hitCounter<2:
				if (self.explore_mode==''):
					if abs(state.position.pan+160) < abs(state.position.pan-160):
						print("and here")
						self.explore_mode = 'Left'
					else:
						self.explore_mode = 'Right'

				if self.explore_mode=='Left':
					if abs(state.position.pan+160)>10:
						self.moveLeft()
					else:
						self.explore_mode = 'Right'
						self.hitCounter = self.hitCounter+1
				if self.explore_mode=='Right':
					if abs(state.position.pan-160)>10:
						self.moveRight()
					else:
						self.explore_mode = 'Left'
						self.hitCounter = self.hitCounter+1
			else:

				print("Did not find", self.target)
				return 0


def main(args):

	print("STRT")
	rg = ptu_controls()
	rospy.init_node('ptu_controls', anonymous=True)
	print("And also here")
	try:
		print("TRying to spin")
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")


if __name__ == '__main__':
	main(sys.argv)
