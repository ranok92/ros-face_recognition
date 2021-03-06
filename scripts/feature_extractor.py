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
import time
import traceback

from rospy_tutorials.msg import Floats
sys.path.append(os.path.abspath("/home/abhisek/Study/Robotics/face_data/LightCNN"))
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from light_cnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
from load_imglist import ImageList

from face_detection.msg import faceArr,facedata	
from face_detection.msg import featdata,featArr
from imutils import face_utils
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rospy.numpy_msg import numpy_msg

class feature_extractor:


    def __init__(self):
        self.bridge = CvBridge()
        self.feature_pub = rospy.Publisher("extracted_features", featArr, queue_size = 0)
        self.img_sub = rospy.Subscriber("cropped_face",faceArr, self.callback)      
        self.path_to_saved_model = "/home/abhisek/Study/Robotics/face_data/LightCNN_29Layers_V2_checkpoint.pth.tar"
        self.model = LightCNN_29Layers_v2(num_classes = 80013 )
        self.cuda = True
        if self.cuda:
            self.model = torch.nn.DataParallel(self.model).cuda()
        if self.path_to_saved_model:
            print("There")
            if os.path.isfile(self.path_to_saved_model):
                self.checkpoint = torch.load(self.path_to_saved_model)
                self.model.load_state_dict(self.checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

        self.transform = transforms.Compose([transforms.Resize([128,128]),
                        transforms.CenterCrop(128),
                        transforms.ToTensor()])
        self.input = torch.zeros(1,1,128,128)



    def callback(self, data):
        
        ##
	#featnpArr = np.zeros([len(data.data)*256,1], dtype=np.float32)
        #path_to_saved_model = "/home/abhisek/Study/Robotics/face_data/LightCNN_29Layers_V2_checkpoint.pth.tar"
        #path_to_imgList = 'This is where cropped face data come in.'
        #model = LightCNN_29Layers_v2(num_classes = 80013 )
        #path_to_save = "no need for this publish it into another topic"
        #cuda = True

        #model.eval()
        #if cuda:
        #    model = torch.nn.DataParallel(model).cuda()
    
        ##if path_to_saved_model:
	    #print("There")
	    #print(path_to_saved_model)
        #    if os.path.isfile(path_to_saved_model):
        #        print("here")
        #        print("=> loading checkpoint '{}'".format(path_to_saved_model))
        #        checkpoint = torch.load(path_to_saved_model)
        #        model.load_state_dict(checkpoint['state_dict'])
        #else:
        #    print("=> no checkpoint found at '{}'".format(args.resume))
    	
    #==============================================================================
    #     img_list  = read_list(args.img_list)
    #     transform = transforms.Compose([transforms.Resize([128,156]),
    # 				    transforms.CenterCrop(128),
    # 				    transforms.ToTensor()])
    #     count     = 0
    #     input     = torch.zeros(1, 1, 128, 128)
    #     for img_name in img_list:
    #         count = count + 1
    #         img   = cv2.imread(os.path.join(args.root_path, img_name), cv2.IMREAD_GRAYSCALE)
    #         print(os.path.join(args.root_path, img_name))
    # 	print(img.shape)
    #==============================================================================
	
	
	print("HE LIENGKHDSG",type(data.data))
	feat_array = []
	
	for i in range(len(data.data)):
	    cur = data.data[i].face
	    sing_feat = featdata()
	    sing_feat.top = data.data[i].top
	    sing_feat.left = data.data[i].left
	    sing_feat.ht = data.data[i].ht
	    sing_feat.wd = data.data[i].wd
	    imgdata = self.bridge.imgmsg_to_cv2(cur, "8UC1")
	    imgdata = np.expand_dims(imgdata, axis=2)
	    imgdatapil = transforms.functional.to_pil_image(imgdata)
	    imgdatatrans = self.transform(imgdatapil)
	    print(type(imgdatatrans))
	
	    print("THE shape now")
	    print(imgdatatrans.shape)

	    print("PROCESSED DATA!!!")
	    print(imgdatatrans)
	    #time.sleep(10)
            try:
                self.input[0,:,:,:] = imgdatatrans
    	        print("the INPUT")
	    
                start = time.time()
                if self.cuda:
                    self.input = self.input.cuda()
                input_var   = torch.autograd.Variable(self.input, volatile=True)
	        print(input)
                _, features = self.model(input_var)
                end = time.time() - start
	        feature = features.cpu().detach().numpy()
	        print("FEAT")
	        print(feature) 
	        features = features.data.cpu().numpy()[0]
	        print(type(feature))
	        a = np.array(features, dtype=np.float32)
		sing_feat.feature.data = a
		feat_array.append(sing_feat)
		#featnpArr[i*256:i*256+256,0]= a
                #self.feature_pub.publish(a)
            except Exception as e:
	        print(traceback.format_exc())
	        print(sys.exc_info()[0])

	try:
	    self.feature_pub.publish(feat_array)
	    print("Published Features!!!")
	    print(type(feat_array))
	    print(feat_array)
	    #time.sleep(30)
	except Exception as e:
	    print(traceback.format_exc())
	    print(sys.exc_info()[0])

def main(args):
    feat_extract = feature_extractor()
    rospy.init_node('feature_extractor',anonymous = True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    
if __name__=='__main__':
    main(sys.argv)
