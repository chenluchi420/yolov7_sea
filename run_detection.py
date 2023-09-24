import sys
import torch
import cv2
import time
import requests
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple
from tqdm import tqdm
import argparse



def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


cuda = True
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']

#names = ['pedestrian', 'vehicle','scooter', 'bicycle'] #IVS
names = ['bicycle', 'pedestrian', 'scooter', 'vehicle']  #roboflow
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

w = "last_best.onnx"
session = ort.InferenceSession(w, providers=providers)

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("image_list", type=str)
    parser.add_argument("output_path", type=str)  # file/folder, 0 for webcam
    opt = parser.parse_args()
    
    #D:\Yun\IVS_Aidea_data\ivslab_test_public\small\image_list.txt
    image_list = opt.image_list
    #image_list = 'D:/Yun/IVS_Aidea/ivslab_test_public/small/image_list.txt'
    list_txt = open(image_list,'r')
    files = []
    for line in list_txt.readlines():
        line = line.rstrip('\n')
        files.append(line)

    output_path = str(opt.output_path)
    #output_path = 'D:/Yun/IVS_Aidea/ivslab_test_public/small/'
    output_name = 'submission.csv'
    new_file_path = output_path+ output_name
    output_csv = open(new_file_path,'w')
    headline = 'image_filename,label_id,x,y,w,h,confidence'+'\n'
    output_csv.write(headline)
    output_csv.close()
    
    for i in tqdm(files):
        #print(i)
        img = cv2.imread(i)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        file_name = i
        file_name = file_name.replace("\\",'?')
        file_name = file_name.replace("/",'?')
        name_list = file_name.split('?')
        #file_name = file_name.replace(output_path,'')
        file_name = name_list[-1]
        print(file_name)
        image = img.copy()
        image, ratio, dwdh = letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255
        im.shape

        outname = [i.name for i in session.get_outputs()]
        outname

        inname = [i.name for i in session.get_inputs()]
        inname

        inp = {inname[0]:im}
        
        outputs = session.run(outname, inp)[0]
        outputs
        ori_images = [img.copy()]

        output_csv = open(new_file_path,'a')
        
        for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
            
            image = ori_images[int(batch_id)]
            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            #print(box)
            cls_id = int(cls_id)
            score = round(float(score),3)
            name = names[cls_id]
            color = colors[name]
            name += ' '+str(score)
            cv2.rectangle(image,box[:2],box[2:],color,2)
            cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  
            box_x_width = box[2]- box[0]
            box_y_width = box[3]- box[1]
            
            obj_id = str(cls_id)
            if obj_id =='0':
                trans_label_id ='4'
            elif obj_id =='1':
                trans_label_id = '2'
            elif obj_id =='2':
                trans_label_id = '3'
            elif obj_id =='3':
                trans_label_id = '1'
            
            write_str = str(file_name)+ ','+ str(trans_label_id)+ ','+ str(box[0])+ ','+ str(box[1])\
                                + ',' + str(box_x_width)+ ','+ str(box_y_width)\
                                + ',' + str(score)+ '\n'
            output_csv.write(write_str)
        
        # cv2.imshow('img',image)
        # # # cv2.imwrite(output_img, image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        