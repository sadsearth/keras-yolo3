import xml.etree.ElementTree as ET
from os import getcwd

#voc_annoataion.py 把xml文件转换为txt

sets=[('2012', 'train'), ('2012', 'val')]

#classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["person"]

def convert_annotation(year, image_id, list_file):
    in_file = open('/Users/wangzijian/Desktop/Detection/pedestrian datasets/VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))

        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


wd = getcwd()

for year, image_set in sets:  # year: 2007  image_set: train/val/test
    image_ids = open('/Users/wangzijian/Desktop/Detection/pedestrian datasets/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')

    for image_id in image_ids:

        FLAGS=0
        in_file = open('/Users/wangzijian/Desktop/Detection/pedestrian datasets/VOCdevkit/VOC%s/Annotations/%s.xml' % (
        year, image_id))
        tree = ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            if xmlbox:
                FLAGS=1
        if FLAGS:
            list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (wd, year, image_id))

            convert_annotation(year, image_id, list_file)
            list_file.write('\n')

    list_file.close()
