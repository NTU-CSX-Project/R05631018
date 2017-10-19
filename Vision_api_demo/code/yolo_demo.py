import json
import os
import cv2
import matplotlib.pyplot as plt
from darkflow.net.build import TFNet
import numpy as np

options = {"model": "cfg/yolo.cfg",
           "load": "bin/yolo.weights",
           "threshold": 0.3}
tfnet = TFNet(options)

def img_recog(img, tfnet):
    preds = tfnet.return_predict(img)
    font = cv2.FONT_HERSHEY_DUPLEX
    thickness = 2
    for pred in preds:
        x_tl = pred['topleft']['x']
        y_tl = pred['topleft']['y']
        x_br = pred['bottomright']['x']
        y_br = pred['bottomright']['y']
        label = pred['label']
        w_text, h_text = cv2.getTextSize(label, font, 0.5, thickness=thickness)[0]

        cv2.rectangle(img, (x_tl, y_tl), (x_br, y_br), (0, 255, 0), 2)
        cv2.rectangle(img, (x_tl, int(y_tl-(h_text/2+1))), (x_tl+w_text+1, y_tl), (0, 255, 0), 7)
        cv2.putText(img, label, (x_tl, y_tl), font, 0.5, (255, 0, 0), thickness=thickness)
    return preds, img

file_name = 'field_robot_2001.npy'
test_video = np.load('./imgs/'+file_name)
test_video_new = list()
#plt.ion()
for i in range(len(test_video)):
    img = test_video[i]
    preds, img = img_recog(img, tfnet)
    test_video_new.append(img)
    if (i+1)%20 == 0:
        print('('+str((i+1))+'/'+str(len(test_video))+') Processing...')
    #plt.imshow(img)
    #plt.pause(0.01)
print('Done.')
np.save('./imgs/'+file_name+'_new', test_video_new)