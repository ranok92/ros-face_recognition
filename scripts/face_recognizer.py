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
import copy


class face_recognizer:

    def __init__(self):
        print("Initializing")

        self.class_pub = rospy.Publisher("final_result",classArr,queue_size = 0)
        print("Here1")
        self.bridge = CvBridge()
        self.feat_sub = rospy.Subscriber("extracted_features",featArr,self.callback)
        self.image_sub = rospy.Subscriber("image_raw",Image,self.imageStorageCall)
        print("Here2")
        self.counter = 0
        self.centroid_info_dir = '/home/abhisek/Study/Robotics/face_data/centroid_info'
        self.file_name = 'centroid_info_8py.npy'
        self.currentFrame = Image()
        #print(data.data.shape)
        #data = np.asarray(data.data).astype(float)
        #print (data.data"
        self.bridge = CvBridge()
        self.threshold = 200
        self.new_feat_dict = {}
        self.new_centroid_dict = {}
        self.prev_frame_info = featArr()
        try:
            self.centroid_info = np.load(os.path.join(self.centroid_info_dir,self.file_name))
        except:
            print("Could not load the centroid_info array.")
        self.num_classes = self.centroid_info.shape[0]
        print ("this is the entroid info", self.centroid_info.shape)

    
    def imageStorageCall(self,data):

        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.currentFrame = imutils.resize(cv_image, height= 300 , width = 400)

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
    
    def generate_Cluster(self,rFeatData):

        #rFeatData is of type - featdata
        #checks if it overlaps with any of the previous classes
        #return -1 if no match found
        key_1 = self.checkSpatialOverlap(rFeatData)
        if key_1 == -1:
            #no overlap means: no spatial relationship with the existing new faces seen so far
            '''can be possible if 
                - if the system is seeing the face for the very first time.
                    -In that case create a new entry in the new_centroid_dict
                - if the system has seen it before but not recently( ie the face has gone out of frame for sometime)
                    -To check if this is the case, try for a k-means within the centroids of the new classes
                        -If this is true
                            - add it to that existing dictionary entry
                        -else
                            -create a new entry
            '''
            key_2 = self.checkFeatureSimilarity(rFeatData)
            if key_2 == -1:
                #so no match in feature space too. Create a new entry in the dictionary
                print("I see a new face")
                new_key_no = self.num_classes+1+len(self.new_feat_dict.keys())
                self.new_feat_dict[str(new_key_no)] = []
                self.new_feat_dict[str(new_key_no)].append(rFeatData) #add featData
            else:
                #no spatial match but there is match in feature. Add to the key_2 with which it finds a match
                self.new_feat_dict[key_2].append(rFeatData)
        else:
            #finds a spatial match add to the dictionary right away
            self.new_feat_dict[key_1].append(rFeatData)

        self.updateInfo() #updates the centroid values of the new forming classes with the addition of each entry
        #also checks for keys that have over 20 values. If it finds one, the function clears that key and creates a 
        #permanent entry of that key into the actual centroid_info array
        #self.generate_Centroid()

    def checkSpatialOverlap(self,rFeatData):

        #checks spatial overlap with the face info in the latest frames of the new_feat_dict
        keyval = -1
        lft = rFeatData.left
        top = rFeatData.top
        for key in self.new_feat_dict:

            length = len(self.new_feat_dict[key])
            if length >0:
                #last_val = len(self.new_feat_dict[key])
                #print(self.new_feat_dict[key][len-1].left)
                left_marker = self.new_feat_dict[key][length-1].left
                top_marker = self.new_feat_dict[key][length-1].top

                if (pow(abs(lft-left_marker),2)+pow(abs(top-top_marker),2)) <= pow(rFeatData.wd/2,2):
                #there is an overlap
                    return key

        return keyval


    def checkFeatureSimilarity(self,rFeatData):

        key_val = -1
        for key in self.new_centroid_dict:
            similarity = np.linalg.norm(self.new_centroid_dict[key] - rFeatData.feature.data)
            if similarity < self.threshold:

                #there is a match in similarity return the key of the match
                return key

        return key_val

    def updateInfo(self):

        #update the centroid information 
        samples_req = 20
        temp_feat = np.zeros([1,256,1])
        temp_feat_arr = np.zeros([1,256,1])
        for key in self.new_feat_dict:
            for i in range(len(self.new_feat_dict[key])):

                temp_feat_arr[0,0:256,0] = np.asarray(self.new_feat_dict[key][i].feature.data)
                print(temp_feat_arr.shape)
                temp_feat = temp_feat+temp_feat_arr

            self.new_centroid_dict[key] = temp_feat/len(self.new_feat_dict[key])
        #check for the 20 criteria.. if any key has more than 20 entries, its officially a new class

        temp_dict = copy.deepcopy(self.new_feat_dict)
        for key in temp_dict:

            if len(self.new_feat_dict[key]) > samples_req:

                new_class = self.centroid_info.shape[0]
                #concat it to the end of the array
                adder = np.zeros([1,257,1])
                adder[0,0:256,:] = self.new_centroid_dict[key]
                adder[0,256,0]  = key 
                del(self.new_centroid_dict[key])
                del(self.new_feat_dict[key])
                self.centroid_info = np.concatenate((self.centroid_info,adder),axis=0)
                self.num_classes = self.num_classes+1

        


    def callback(self,data):
        print("inside callback")
        final_class = ''
        #print("THIS IS THE dATA",data.data)
        #print(type(data))
        #print (centroid_info[:,:,:])
        #print (type(data.data))
        #print (data.data.shape)
        #no_of_feats_read = data.data.shape[0]/256
        #print(no_of_feats_read)
        classInfoArray = classArr()
        classInfoArray.header.stamp = rospy.Time.now()
        for i in range(len(data.data)):
            classInfo = classdata()
            cl2 , collector = self.checkClass(self.centroid_info,data.data[i].feature.data)
            if collector[cl2] < self.threshold:
                classInfo.classval = str(cl2+1)
                classInfo.left = data.data[i].left
                classInfo.top = data.data[i].top
                classInfo.ht = data.data[i].ht
                classInfo.wd = data.data[i].wd
                cv2.rectangle(self.currentFrame,(int(data.data[i].left),int(data.data[i].top)),(int(data.data[i].left+data.data[i].wd),int(data.data[i].top+data.data[i].wd)),(0,255,0),2)
                final_class = final_class+str(cl2+1)+','
		font = cv2.FONT_HERSHEY_COMPLEX_SMALL
		cv2.putText(self.currentFrame,str(cl2+1),(int(data.data[i].left-2),int(data.data[i].top-2)),font,1,(0,255,0))
                print(final_class)
                #print(collector)
                classInfoArray.data.append(classInfo)
            else:
                #create a new class because it doesnt match the existing ones
		cv2.rectangle(self.currentFrame,(int(data.data[i].left),int(data.data[i].top)),(int(data.data[i].left+data.data[i].wd),int(data.data[i].top+data.data[i].wd)),(0,0,255),2)
                final_class = final_class+str(cl2+1)+','
		font = cv2.FONT_HERSHEY_COMPLEX_SMALL
		cv2.putText(self.currentFrame,'???',(int(data.data[i].left-2),int(data.data[i].top-2)),font,1,(0,0,255))
                self.generate_Cluster(data.data[i])

            #print("This is the final class")
            #SSprint(collector)
        cv2.imshow("Image window", self.currentFrame)
        cv2.waitKey(1)

        try:
            self.class_pub.publish(classInfoArray)
            self.prev_frame_info = data
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
