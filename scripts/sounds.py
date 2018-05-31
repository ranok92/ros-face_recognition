#!/usr/bin/env python
import sys

import rospy
from sound_play.msg import SoundRequest
from sound_play.libsoundplay import SoundClient
from std_msgs.msg import String

class sounds:
	
	def __init__(self):

		self.img_sub = rospy.Subscriber("final_result",String, self.callback)      
		self.prevCallback = 'None'
		self.prefix = 'Hi'
		self.suffix = 'How are you?'

	def callback(self,data):
		
		
		if (self.prevCallback!=data.data and data.data!=''):
			soundHandle = SoundClient()
			#rospy.sleep(5)
			voice = 'voice_kal_diphone'
			volume = 7.0
			print(data.data)
			soundHandle.say(str(self.prefix+' '+data.data+self.suffix), voice,volume)
			rospy.sleep(5)
			self.prevCallback = data.data
		else:
			pass


def main(args):

	sd =  sounds()
	rospy.init_node('sounds',anonymous = True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("shutting down")
	
    
if __name__=='__main__':
    main(sys.argv) 
		
