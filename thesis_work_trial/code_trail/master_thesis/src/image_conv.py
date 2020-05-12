#!/usr/bin/env python
from __future__ import print_function

#import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

backSub = cv2.createBackgroundSubtractorMOG2()

class image_converter:

  def __init__(self):
	  self.image_pub = rospy.Publisher("/depthimage",Image, queue_size=10)
	  self.bridge = CvBridge()
	  self.image_sub = rospy.Subscriber("/sim_ros_interface/vrep/kinect/depth",Image,self.callback)

  def callback(self,data):
	  global backSub
	  try:
		  cv_image = self.bridge.imgmsg_to_cv2(data) #, "32FC1")#, "32FC1") #"rgb8")#"bgr8")
	  except CvBridgeError as e:
		  print(e)
	  #(rows,cols) = cv_image.shape
	  print(cv_image.shape)
	  backtorgb = cv_image.copy()
	  #backtorgb = cv2.cvtColor(backtorgb,cv2.COLOR_GRAY2RGB)
	  print(backtorgb.shape)
	  print("got an image in BGS")
	  fgMask = backSub.apply(backtorgb, learningRate=1.0/100)
	  #fgMask = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)
	  #ret, mask = cv2.threshold(fgMask, 15, 255, cv2.THRESH_BINARY)
	  # find coutour
	  contour_image, contours, hierarchy = cv2.findContours(fgMask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	  # Fit bounding box
	  try: hierarchy = hierarchy[0]
	  except: 
		  hierarchy = []
		  print("empty")
	  
	  height, width = fgMask.shape
	  min_x, min_y = width, height
	  max_x = max_y = 0
	  # Convert input frame to visualize output
	  output_img = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)
	  # Display all found contours via rectangles
	  for contour, hier in zip(contours, hierarchy):
		  (x,y,w,h) = cv2.boundingRect(contour)
		  min_x, max_x = min(x, min_x), max(x+w, max_x)
		  min_y, max_y = min(y, min_y), max(y+h, max_y)
		  #if w > 100 and h > 100:
			#  cv2.rectangle(output_img, (x,y), (x+w,y+h), (255, 0, 0), 2)
		  if max_x - min_x > 75 and max_y - min_y > 75:
			  cv2.rectangle(output_img, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
	  #cv2.imshow('mask_thresh', mask)
	  cv2.imshow('final_image', output_img)
	  cv2.imshow('FG Mask', fgMask)
	  cv2.imshow("Image window", backtorgb)
	  cv2.waitKey(3)
	  try:
		  self.image_pub.publish(self.bridge.cv2_to_imgmsg(backtorgb))
	  except CvBridgeError as e:
		  print(e)
      
  
def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
