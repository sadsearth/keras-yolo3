#-*- coding : utf-8 -*-
# coding: utf-8
import codecs
import os
import re
from PIL import Image


image_path = "/Users/wangzijian/Desktop/Detection/pedestrian_datasets/INRIAPerson/Train/pos"
# 图片存放路径，路径固定
annotations_path = "/Users/wangzijian/Desktop/Detection/pedestrian_datasets/INRIAPerson/Train/annotations/"
#文件夹目录, INRIA标签存放路径
annotations= os.listdir(annotations_path) #得到文件夹下的所有annotations文件名称

# 获取文件夹下所有图片的图片名
def get_name(file_dir):
   list_file=[]
   for root, dirs, files in os.walk(file_dir):
      for file in files:
         # splitext()将路径拆分为文件名+扩展名，例如os.path.splitext(“E:/lena.jpg”)将得到”E:/lena“+".jpg"
         if os.path.splitext(file)[1] == '.jpg':
            list_file.append(os.path.join(root, file))
   return list_file

# 将所有待处理的图片路径放入2012_train.txt文件夹中
image_names = get_name(image_path)   # 所有图片名字（有路径）
list_file = open('INRIA_train.txt', 'w')  # 写入2012_train.txt文件中
for image_name in image_names:
    image_name1=image_name.split("pos/")[1]  #获取图片名字
    image_name2=image_name1.replace('.jpg', '.txt') #获取图片相应的annotation
    str_XY = "(Xmax, Ymax)"
    bnd1=""
    for line in open(annotations_path+"/"+image_name2, encoding='utf-8', errors='replace'):
        if str_XY in line:
           strlist = line.split(str_XY)
           strlist1 = "".join(strlist[1:])  # 把list转为str
           strlist1 = strlist1.replace(':', '')
           strlist1 = strlist1.replace('-', '')
           strlist1 = strlist1.replace('(', '')
           strlist1 = strlist1.replace(')', '')
           strlist1 = strlist1.replace(',', '')
           b = strlist1.split()
           bnd = b[0] + ',' + b[1] + ',' + b[2] + ',' + b[3] + ',0'

          #   text_create(str_name, bnd)
        else:
           continue
        bnd1 = bnd1+' '+bnd
    list_file.write('%s'%(image_name)+bnd1)
    list_file.write('\n')
list_file.close()

# for file in annotations: #遍历annotations文件夹
#   # str_name = file.replace('.txt', '')
#    if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
#       #with open(annotations_path+"/"+file) as f : #打开文件
#       with codecs.open(annotations_path+"/"+file, encoding='utf-8', errors='replace') as f:
#          iter_f = iter(f) #创建迭代器
#          for line in iter_f: #遍历文件，一行行遍历，读取文本
#             str_XY = "(Xmax, Ymax)"
#             if str_XY in line:
#                strlist = line.split(str_XY)
#                strlist1 = "".join(strlist[1:])    # 把list转为str
#                strlist1 = strlist1.replace(':', '')
#                strlist1 = strlist1.replace('-', '')
#                strlist1 = strlist1.replace('(', '')
#                strlist1 = strlist1.replace(')', '')
#                strlist1 = strlist1.replace(',', '')
#                b = strlist1.split()
#                bnd = (b[0]+ ',' +b[1] + ',' +b[2] +','+ b[3]+',0')
#             bnd1=
#             #   text_create(str_name, bnd)
#             else:
#                continue
#
# # 在labels目录下创建每个图片的标签txt文档
# def text_create(name,bnd):
#    full_path = "/Users/wangzijian/Desktop/Detection/pedestrian datasets/VOCdevkit/INRIA2VOC/labels/%s.txt"%(name)
#    size = get_size(name + '.jpg')
#    convert_size = convert(size, bnd)
#    file = open(full_path, 'a')
#    file.write(str(convert_size[0]) + ' ' + str(convert_size[1]) + ' ' + str(convert_size[2]) + ' ' + str(convert_size[3]) + '0 ')
#    file.write('\n')
#
# # # 将Tagphoto的x,y,w,h格式转换成yolo的X,Y,W,H
# # def convert(size, box):
# #    dw = 1./size[0]
# #    dh = 1./size[1]
# #    x = (box[0] + box[2])/2.0
# #    y = (box[1] + box[3])/2.0
# #    w = box[2] - box[0]
# #    h = box[3] - box[1]
# #    x = x*dw
# #    w = w*dw
# #    y = y*dh
# #    h = h*dh
# #    return (x,y,w,h)
#
# # # 获取要查询的图片的w,h
# # def get_size(image_id):
# #    im = Image.open('/Users/wangzijian/Desktop/Detection/pedestrian datasets/INRIAPerson/Train/pos/%s'%(image_id))
# #    # 源图片存放路径
# #    size = im.size
# #    w = size[0]
# #    h = size[1]
# #    return (w,h)
#
# import xml.etree.ElementTree as ET
# from os import getcwd
#
# #voc_annoataion.py 把xml文件转换为txt
#
# sets=[('2012', 'train'), ('2012', 'val')]
#
# #classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# classes = ["person"]
#
# def convert_annotation(year, image_id, list_file):
#     in_file = open('/Users/wangzijian/Desktop/Detection/pedestrian datasets/VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
#     tree=ET.parse(in_file)
#     root = tree.getroot()
#
#     for obj in root.iter('object'):
#         difficult = obj.find('difficult').text
#         cls = obj.find('name').text
#         if cls not in classes or int(difficult)==1:
#             continue
#         cls_id = classes.index(cls)
#         xmlbox = obj.find('bndbox')
#         b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
#
#         list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
#
#
# wd = getcwd()
#
# for year, image_set in sets:  # year: 2007  image_set: train/val/test
#     image_ids = open('/Users/wangzijian/Desktop/Detection/pedestrian datasets/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
#     list_file = open('%s_%s.txt'%(year, image_set), 'w')
#
#     for image_id in image_ids:
#
#         FLAGS=0
#         in_file = open('/Users/wangzijian/Desktop/Detection/pedestrian datasets/VOCdevkit/VOC%s/Annotations/%s.xml' % (
#         year, image_id))
#         tree = ET.parse(in_file)
#         root = tree.getroot()
#
#         for obj in root.iter('object'):
#             difficult = obj.find('difficult').text
#             cls = obj.find('name').text
#             if cls not in classes or int(difficult) == 1:
#                 continue
#             cls_id = classes.index(cls)
#             xmlbox = obj.find('bndbox')
#             if xmlbox:
#                 FLAGS=1
#         if FLAGS:
#             list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (wd, year, image_id))
#
#             convert_annotation(year, image_id, list_file)
#             list_file.write('\n')
#
#     list_file.close()