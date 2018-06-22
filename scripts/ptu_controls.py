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

		self.class_pub = rospy.Publisher("/ptu_driver/cmd_rel",PanTilt,queue_size = 0)
		self.classifier_info = message_filters.Subscriber("final_result",classArr)
		self.command_info = message_filters.Subscriber("command",command)
		self.ptu_state = message_filters.Subscriber("/ptu_driver/state",PanTiltStateStamped)
		self.class_vel_pub = rospy.Publisher("/ptu_driver/cmd_vel",PanTilt,queue_size=0)
		print("Here2")
		self.control = message_filters.ApproximateTimeSynchronizer([self.classifier_info , self.command_info ,self.ptu_state],3,slop=.2)
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
		self.is_tracking = False
		self.thresh = 1.4
		self.track_final_pos = PanTilt()
		self.integral_term_pan = 0
		self.integral_term_tilt = 0
		self.prev_val_pan = 0
		self.prev_val_tilt = 0
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

		finder_call = message_filters.ApproximateTimeSynchronizer([self.classifier_info , self.ptu_state],1,slop=.1)
		result = finder_call.registerCallback(self.findCallback)
		#finder_call.unregister()
 		return result


 	def tracker(self,final_resultT,state):

 		print("inside tracker")
 		move = PanTilt()
 		c = 180*3.7795275591
 		print(final_resultT)
 		w = c/final_resultT.wd*self.focal_length
 		w = w/1
 		print('W : ',w)
 		init_point = np.asarray([final_resultT.left*w , final_resultT.top*w , w])
 		print("Init_point :" ,init_point)
 		inv_mat = np.linalg.inv(self.camera_matrix)
 		init_point_wc = np.dot(inv_mat,init_point)
 		print("Init_point_wc :",init_point_wc)
 		final_point = np.asarray([(self.image_size_x/2-final_resultT.wd/2)*w , init_point[1] , w])
 		final_point_tilt = np.asarray([init_point[0],(self.image_size_y/2-final_resultT.ht/2)*w , w])
 		final_point_wc = np.dot(inv_mat,final_point)
 		final_point_wc_tilt = np.dot(inv_mat,final_point_tilt)
 		print("Final_point_wc :",final_point_wc)
 		cos_theta = np.dot(init_point_wc,final_point_wc)/((np.linalg.norm(init_point_wc)*np.linalg.norm(final_point_wc)))
 		cos_theta_tilt = np.dot(init_point_wc,final_point_wc_tilt)/((np.linalg.norm(init_point_wc)*np.linalg.norm(final_point_wc_tilt)))
 		print(cos_theta)
 		print(cos_theta_tilt)
 		try:
 			pan_value = math.acos(cos_theta)*90/math.pi
 			#pan_value = -(final_point[0] - init_point[0])/(8*w)
 			print("pan_value :", pan_value)
 			tilt_value = math.acos(cos_theta_tilt)*90/math.pi
 			#tilt_value = (final_point[1] - init_point[1])/(8*w)
 			if init_point[0] > final_point[0]:
 				move.pan = -pan_value
 			else: 
				move.pan = pan_value
			if init_point[1] > final_point_tilt[1]:
 				move.tilt = -tilt_value
 			else:
 				move.tilt = tilt_value
 			print("state : ",state.position)
 			print("move :",move)
 			print("final_position :",self.track_final_pos)
 			if (self.is_tracking):
 				if (abs(state.position.pan - self.track_final_pos.pan) < self.thresh and abs(state.position.tilt - self.track_final_pos.tilt) < self.thresh):
 					if (abs(move.pan) > self.thresh or abs(move.tilt) > self.thresh):
 						print("track1")
 						self.class_pub.publish(move)
 						self.track_final_pos.pan = state.position.pan + move.pan
 						self.track_final_pos.tilt = state.position.tilt + move.tilt
 				else:
 					print("skipping")
 			else:
 				if (abs(move.pan) > self.thresh or abs(move.tilt) > self.thresh):
 					print("track2")
 					self.class_pub.publish(move)
					self.is_tracking = True
					self.track_final_pos.pan = state.position.pan + move.pan
 					self.track_final_pos.tilt = state.position.tilt + move.tilt
 				else:
 					print("track3")
 		except Exception as e:

 			print(traceback.format_exc())
 			print(sys.exc_info()[0])


	def trackerPID(self,final_resultT,state):

 		print("inside tracker")
 		move = PanTilt()
 		c = 180*3.7795275591
 		print(final_resultT)
 		w = c/final_resultT.wd*self.focal_length
 		w = w/1
 		print('W : ',w)
 		init_point = np.asarray([final_resultT.left*w , final_resultT.top*w , w])
 		print("Init_point :" ,init_point)
 		inv_mat = np.linalg.inv(self.camera_matrix)
 		init_point_wc = np.dot(inv_mat,init_point)
 		print("Init_point_wc :",init_point_wc)
 		final_point = np.asarray([(self.image_size_x/2-final_resultT.wd/2)*w , init_point[1] , w])
 		final_point_tilt = np.asarray([init_point[0],(self.image_size_y/2-final_resultT.ht/2)*w , w])
 		final_point_wc = np.dot(inv_mat,final_point)
 		final_point_wc_tilt = np.dot(inv_mat,final_point_tilt)
 		print("Final_point_wc :",final_point_wc)
 		cos_theta = np.dot(init_point_wc,final_point_wc)/((np.linalg.norm(init_point_wc)*np.linalg.norm(final_point_wc)))
 		cos_theta_tilt = np.dot(init_point_wc,final_point_wc_tilt)/((np.linalg.norm(init_point_wc)*np.linalg.norm(final_point_wc_tilt)))
 		print(cos_theta)
 		print(cos_theta_tilt)
 		try:
 			pan_value = math.acos(cos_theta)*90/math.pi
 			#pan_value = -(final_point[0] - init_point[0])/(8*w)
 			print("pan_value :", pan_value)
 			tilt_value = math.acos(cos_theta_tilt)*90/math.pi
 			#tilt_value = (final_point[1] - init_point[1])/(8*w)
 			if init_point[0] > final_point[0]:
 				move.pan = self.PID_value(-pan_value)
 			else: 
				move.pan = self.PID_value(pan_value)
			if init_point[1] > final_point_tilt[1]:
 				move.tilt = self.PID_value(-tilt_value)
 			else:
 				move.tilt = self.PID_value(tilt_value)
 			self.class_pub.publish(move)
 		except Exception as e:

 			print(traceback.format_exc())
 			print(sys.exc_info()[0])




 	def trackerpixel(self,final_resultT,state):

 		print("inside tracker")
 		move = PanTilt()
 		c = 180*3.7795275591
 		print(final_resultT)
 		w = c/final_resultT.wd*self.focal_length
 		w = w/1
 		print('W : ',w)
 		init_point = np.asarray([final_resultT.left*w , final_resultT.top*w , w])
 		print("Init_point :" ,init_point)
 		inv_mat = np.linalg.inv(self.camera_matrix)
 		init_point_wc = np.dot(inv_mat,init_point)
 		print("Init_point_wc :",init_point_wc)
 		final_point = np.asarray([(self.image_size_x/2-final_resultT.wd/2)*w , init_point[1] , w])
 		final_point_tilt = np.asarray([init_point[0],(self.image_size_y/2-final_resultT.ht/2)*w , w])
 		final_point_wc = np.dot(inv_mat,final_point)
 		final_point_wc_tilt = np.dot(inv_mat,final_point_tilt)
 		print("Final_point_wc :",final_point_wc)
 		cos_theta = np.dot(init_point_wc,final_point_wc)/((np.linalg.norm(init_point_wc)*np.linalg.norm(final_point_wc)))
 		cos_theta_tilt = np.dot(init_point_wc,final_point_wc_tilt)/((np.linalg.norm(init_point_wc)*np.linalg.norm(final_point_wc_tilt)))
 		print(cos_theta)
 		print(cos_theta_tilt)
 		try:
 			#pan_value = math.acos(cos_theta)*90/math.pi
 			pan_value = (final_point[0] - init_point[0])/(8*w)
 			print("pan_value :", pan_value)
 			#tilt_value = math.acos(cos_theta_tilt)*90/math.pi
 			tilt_value = (final_point[1] - init_point[1])/(8*w)
 			move.pan = pan_value
 			move.tilt = tilt_value
 			self.class_pub.publish(move)

 		except Exception as e:

 			print(traceback.format_exc())
 			print(sys.exc_info()[0])

 	def tracker_vel(self,final_resultT,state):
		
		print("inside tracker")
 		move = PanTilt()
 		print(final_resultT)
 		init_point = np.asarray([final_resultT.left , final_resultT.top])
		#init_point = np.asarray([10,10])
		#final_resultT.left = 10
		#final_resultT.top = 10
		#final_resultT.wd = 50
		#final_resultT.ht = 50
 		print("Init_point :" ,init_point)
 		final_point = np.asarray([(self.image_size_x/2-final_resultT.wd/2) , init_point[1] ])
 		final_point_tilt = np.asarray([init_point[0],(self.image_size_y/2-final_resultT.ht/2)])
 		pan_dist = (final_resultT.left+final_resultT.wd/2)-200
 		tilt_dist = (final_resultT.top+final_resultT.ht/2)-150
 		print("pan_dist :",pan_dist)
 		print("tilt_dist :",tilt_dist)
 		pan_value = self.sigmoid(pan_dist)
 		tilt_value = self.sigmoid(tilt_dist)
 		print("pan_value :",pan_value)
 		print("tilt_value :",tilt_value)
 		move.pan = self.PID_value_pan(-pan_dist)
 		move.tilt = self.PID_value_tilt(-tilt_dist)
		print("move :",move)
 		try:

			self.class_vel_pub.publish(move) 
			#rospy.sleep(10)
 		except Exception as e:

 			print(traceback.format_exc())
 			print(sys.exc_info()[0])

	def PID_value_pan(self, error_term):

		int_mult = 0
		int_error = 10
		int_der = .1
		self.integral_term_pan = self.integral_term_pan+error_term
		integral_term = self.other_func(self.integral_term_pan)*int_mult
		diff_term = self.other_func(error_term - self.prev_val_pan)*int_der
		self.prev_val_pan = error_term
		error_term = self.other_func(error_term)*int_error
		print("INtegral :",integral_term)
		print("error:",error_term)
		print("diff :",diff_term)
		return integral_term+diff_term+error_term


	def PID_value_tilt(self, error_term):

		int_mult = 0
		int_error = 4
		int_der = .1
		self.integral_term_tilt = self.integral_term_tilt+error_term
		integral_term = self.other_func(self.integral_term_tilt)*int_mult
		diff_term = self.other_func(error_term - self.prev_val_tilt)*int_der
		self.prev_val_tilt = error_term
		error_term = self.other_func(error_term)*int_error
		print("INtegral :",integral_term)
		print("error:",error_term)
		print("diff :",diff_term)
		return integral_term+diff_term+error_term


		
	def other_func(self, value):

		if abs(value) > 70:
			if value > 0:
				return 10
			else:
				return -10
		if abs(value) < 20:
			return 0
		else:
			return value/10

 	def sigmoid(self, dist):

 		num = np.exp(dist)
 		den = num+1
 		multilplier = 1
 		return ((num/den)-.5)*multilplier


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
			self.tracker_vel(i,state)

		else:
			print("cant see so i should turn")
			self.is_tracking = False
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
