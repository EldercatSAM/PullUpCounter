#!/usr/bin/env python

import numpy as np
import cv2
from openni import openni2
from openni import _openni2 as c_api

use_rgb = True
use_depth = False

openni2.initialize() 
if (openni2.is_initialized()):
    print ("openNI2 initialized")
else:
    print ("openNI2 not initialized")

dev = openni2.Device.open_any()
if use_rgb:
    rgb_stream = dev.create_color_stream()
    rgb_stream.start()
if use_depth:
    depth_stream = dev.create_depth_stream()

    print ('Get b4 video mode', depth_stream.get_video_mode()) # Checks depth video configuration
    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=640, resolutionY=480, fps=30))

    print ('Mirroring info1', depth_stream.get_mirroring_enabled())
    depth_stream.set_mirroring_enabled(False)

    depth_stream.start()

def get_rgb(Camera_height=480, Camera_width = 640):
    bgr   = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(Camera_height,Camera_width,3)
    rgb   = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    return rgb    

def get_depth(Camera_height=480, Camera_width = 640):
    """
    Returns numpy ndarrays representing the raw and ranged depth images.
    Outputs:
        dmap:= distancemap in mm, 1L ndarray, dtype=uint16, min=0, max=2**12-1
        d4d := depth for dislay, 3L ndarray, dtype=uint8, min=0, max=255    
    Note1: 
        fromstring is faster than asarray or frombuffer
    Note2:     
        .reshape(120,160) #smaller image for faster response 
                OMAP/ARM default video configuration
        .reshape(240,320) # Used to MATCH RGB Image (OMAP/ARM)
                Requires .set_video_mode
    """
    dmap = np.fromstring(depth_stream.read_frame().get_buffer_as_uint16(),dtype=np.uint16).reshape(Camera_height,Camera_width)  # Works & It's FAST
    d4d = np.uint8(dmap.astype(float) *255/ 2**12-1) # Correct the range. Depth images are 12bits
    d4d = 255 - cv2.cvtColor(d4d,cv2.COLOR_GRAY2RGB)
    return dmap, d4d

def Camera_stop():
    global dev,rgb_stream,depth_stream
    rgb_stream.stop()
    depth_stream.stop()
    openni2.unload()
    print ("Terminated")

    """
    rgb = get_rgb()
    
    #DEPTH
    _,d4d = get_depth()
    
    # Canvas
    rgbd = np.hstack((rgb,d4d))


    ## Display the stream syde-by-side
    cv2.imshow('depth || rgb', rgbd)
    """