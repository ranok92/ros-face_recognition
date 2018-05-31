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
from rospy.numpy_msg import numpy_msg
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


from imutils import face_utils
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class lightCNN_test:


    def __init__(self):
        self.bridge = CvBridge()
        self.feature_pub = rospy.Publisher("extracted_features",String, queue_size=10)  
        
    def read_list(self,list_path):
        img_list = []
        with open(list_path, 'r') as f:
            for line in f.readlines()[0:]:
                img_path = line.strip().split()
                img_list.append(img_path[0])
        print('There are {} images..'.format(len(img_list)))
        return img_list
    

    def save_feature(self,save_path, img_name, features):
    	img_path = os.path.join(save_path, img_name)
    	img_dir  = os.path.dirname(img_path) + '/';
    	if not os.path.exists(img_dir):
        	os.makedirs(img_dir)
    	fname = os.path.splitext(img_path)[0]
    	fname = fname + '.csv'

    	if '/2/2.pgm' in img_name:
		print('The type of features is :')
    		print(type(features))
		print(img_path)
    		print(features)
    	#fid.write(features)
    	#fid.close()
    	np.savetxt(fname,features,delimiter=',')

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
        
        ##
	print("in callback")
        path_to_saved_model = "/home/abhisek/Study/Robotics/face_data/LightCNN_29Layers_V2_checkpoint.pth.tar"
        path_to_img = '/home/abhisek/Study/Robotics/face_data/output.txt'
	save_path = '/home/abhisek/Study/Robotics/face_data/face_feat_upaaa'
	imglist = self.read_list(path_to_img)
        model = LightCNN_29Layers_v2(num_classes = 80013 )
        path_to_save = "no need for this publish it into another topic"
        cuda = True


	centroid_info_dir = '/home/abhisek/Study/Robotics/face_data/centroid_info'
        file_name = 'centroid_info_8py.npy'
        try:
            centroid_info = np.load(os.path.join(centroid_info_dir,file_name))
        except:
            print("Could not load the centroid_info array.")

        model.eval()
        if cuda:
            model = torch.nn.DataParallel(model).cuda()
    
        if path_to_saved_model:
	    print("There")
	    print(path_to_saved_model)
            if os.path.isfile(path_to_saved_model):
		print("here")
                print("=> loading checkpoint '{}'".format(path_to_saved_model))
                checkpoint = torch.load(path_to_saved_model)
                model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(path_to_saved_model))
    	
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
	transform = transforms.Compose([transforms.Resize([128,156]),
    				    transforms.CenterCrop(128),
     				    transforms.ToTensor()])
        input = torch.zeros(1,1,128,128)
	for img_name in imglist:

	    imgdata = cv2.imread(os.path.join(img_name), cv2.IMREAD_GRAYSCALE)

	    imgdata = np.expand_dims(imgdata, axis=2)
	    imgdatapil = transforms.functional.to_pil_image(imgdata)
	    imgdatatrans = transform(imgdatapil)
	    print(type(imgdatatrans))
	
	    print("THE shape now")
      	    print(imgdatatrans.shape)
 
	    print("PROCESSED DATA!!!")
	    #print(imgdatatrans)
	#time.sleep(10)
            try:
                input[0,:,:,:] = imgdatatrans
    	        print("the INPUT")
	    
                start = time.time()
                if cuda:
                    input = input.cuda()
                input_var   = torch.autograd.Variable(input, volatile=True)
	        #print(input)
                _, features = model(input_var)
                end = time.time() - start
	        feature = features.cpu().detach().numpy()
	        #print("FEAT")
	        #print(feature) 
	        features = features.data.cpu().numpy()[0]
	        print(type(feature))
	        a = np.array(features, dtype=np.float32)
                #self.feature_pub.publish(a)
	        print("Published Features!!!")
		print(img_name)
	        print(features)
		#self.save_feature(save_path,img_name,features)
		cl2 , collector = self.checkClass(centroid_info,features)
      		final_class = cl2+1
      		#print("This is the final class")
		self.feature_pub.publish(str(final_class))
            except Exception as e:
	        print(traceback.format_exc())
	        print(sys.exc_info()[0])



def main(args):
    print("start")
    light = lightCNN_test()
    rospy.init_node('lightCNN_test',anonymous = True)
    light.talker()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    
if __name__=='__main__':
    main(sys.argv)
