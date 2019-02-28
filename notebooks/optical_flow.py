import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import argparse
import sys

def optical_flow(video):
    
    '''
        This is for computing the optical flow of a given video.
        Args:
            video: the input video
        Return:
            final: final aray matrix 256,256'''
                           
     
    # storing the first frame of the video
    initial_frame=video[0]
    
    #converting it into proper format
    hsv = np.zeros_like(initial_frame)
    hsv = np.expand_dims(hsv, axis=2)
    rgb_frame=cv2.cvtColor(initial_frame,cv2.COLOR_GRAY2RGB)
    hsv=cv2.cvtColor(rgb_frame,cv2.COLOR_RGB2HSV)
    hsv[...,1] = 255
    final=0 
    #optical flow code reference:https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html
    for next_frame in video[1:]:
        previous_frame=initial_frame
        flow = cv2.calcOpticalFlowFarneback(previous_frame,next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) 
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR) 
        gaussian_smoothing = cv2.blur(bgr,(5,5))
        thresh= cv2.threshold(gaussian_smoothing, 66, 255, cv2.THRESH_BINARY)[1]
        gray = cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',bgr)
        #k = cv2.waitKey(30) & 0xff
    
        # storing the value convert value in final aray
        final += np.asarray(gray)
        previous_frame=next_frame
    
    return final

def video(image_files):
    '''
        This is for computing the optical flow of a given video.
        Args:
            image_files: list containing the path of the frame
        Return:
            video: the input video to used for optical flow'''
    
    video = []
    #create a video
    for image in image_files:
        img=cv2.imread(image,0)
        res = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        video.append(res)
    
    return video

def path(dir_path):
    '''
        This is for computing the optical flow of a given video.
        Args:
            dir_path: path of the folder containing frame
        Return:
            imagefile: list conatining the path of the frame'''
    
    # loading images in the list
    image_files = sorted([os.path.join(dir_path, file)
                      for file in os.listdir(dir_path) 
                      if file.endswith('.png')])

    return image_files

if __name__=='__main__' :
    
    print("give the path of the directory")
    dir_path="/home/anant/data_science_practicum/p2/cilia Segementation/documents and data/dataset/data/new/"
    image_files=path(dir_path)
    video=video(image_files)
    optical_flow_array=optical_flow(video)



