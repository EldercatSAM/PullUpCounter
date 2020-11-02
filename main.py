import os
os.system('source OpenNIDevEnviorment')
from getcamera import Camera

import cv2
import threading
import time

Use_coral = True
if Use_coral:
    import argparse
    import platform
    import subprocess
    from edgetpu.detection.engine import DetectionEngine
    from edgetpu.utils import dataset_utils
    from PIL import Image
    from PIL import ImageDraw
    import numpy as np
else:
    
    from centerface import CenterFace
    import scipy.io as sio

model = 'mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'


def Detect_face():
    frame = camera.get_rgb()
    frame = cv2.resize(frame,(240,180),1,1,cv2.INTER_AREA)
    dets, lms = centerface(frame, 180, 240, threshold=0.35)
    for det in dets:
        boxes, score = det[:4], det[4]
        cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
    for lm in lms:
        for i in range(0, 5):
            cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
    cv2.imshow('out', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True
    # Press Q on keyboard to stop recording

def Detect_face_coral(img):
    ans = engine.DetectWithImage(
        img,
        threshold=0.05,
        keep_aspect_ratio=True,
        relative_coord=False,
        top_k=10)

  # Display result.
    if ans:
        for obj in ans:
            print('-----------------------------------------')
            print('score = ', obj.score)
            box = obj.bounding_box.flatten().tolist()
            print('box = ', box)
      # Draw a rectangle.
            draw.rectangle(box, outline='red')
            #img.save(output_name)
            #subprocess.Popen(['feh', output_name])
    else:
        print('No object detected!')
    return img,ans

Color = None
Depth = None
Depth_map = None
def getIMG():
    global Color,Depth,Depth_map
    while True:
        Color = camera.get_rgb()
        Depth_map,Depth = camera.get_depth()

if __name__ == '__main__':
    camera = Camera()
    th1 = threading.Thread(target=getIMG)
    th1.setDaemon(True)     # 设置为后台线程，这里默认是False，设置为True之后则主线程不用等待子线程
    th1.start() 
    if Use_coral:
        engine = DetectionEngine(model)
    else:
        centerface = CenterFace()
    #getcamera.Camera_init()
    while True:
        if Use_coral:
            if Color is not None and Depth is not None:
                #frame = camera.get_rgb()
                #dmap,d4d = camera.get_depth()
                #frame = cv2.resize(frame,(640,480),1,1,cv2.INTER_AREA)
                img = Image.fromarray(cv2.cvtColor(Color,cv2.COLOR_BGR2RGB))
                d4d = Depth
                draw = ImageDraw.Draw(img)
                image,ans = Detect_face_coral(img)
                img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
                #print 'Center pixel is {}mm away'.format(dmap[119,159])

                ## Display the stream syde-by-side
                rgbd = np.hstack((img,Depth))


                ## Display the stream syde-by-side
                cv2.imshow('depth || rgb', rgbd)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)
        else:
            ret = Detect_fase()
            if ret is False:
                break
    getcamera.Camera_stop()
        