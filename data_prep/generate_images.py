import cv2
import shutil
import os
from glob import glob
from metrics.xml_to_gt import rotate_rect, pts_to_box
import xml.etree.ElementTree as ET


xmls = glob("/data/text_block_detection/manual_labels/*.xml")
xmls += glob("/data/text_block_detection/validation_labels/*.xml")
print(len(xmls))

for xml in xmls:
    filename = xml.split("/")[-1]
    filename = filename.split(".xml")[0]
    img = os.path.join("/sdata/raw/backimage/doc_type_11", filename + ".jpg")
    i = cv2.imread(img)

    if i is None:
        # No image found so skip
        continue

    tree = ET.parse(xml)
    root = tree.getroot()
    lines = []
    text = '###'
    try:
        for member in root.findall('object'):
            member = member.find('bndbox')
            xmin = int(member.find('xmin').text)
            ymin = int(member.find('ymin').text)
            xmax = int(member.find('xmax').text)
            ymax = int(member.find('ymax').text)
            xmin, ymin, xmax, y_max = rotate_rect(xmin, ymin, xmax, ymax, theta=270)
            pts = pts_to_box(xmin, ymin, xmax, ymax)
            line = ""
            for pt in pts:
                line += str(pt) + ","
            line += text
            lines.append(line)

        for i in range(len(lines) - 1):
            lines[i] += "\n"

        with open(os.path.join('/sdata/tb/tb_detector/backimages', "gt_" + filename + '.txt'), 'w') as outfile:
            outfile.writelines(lines)
    except Exception as e:
        print(e)
        continue

    print("aaa")
    i_rot = cv2.rotate(i, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(os.path.join("/sdata/tb/tb_detector/backimages", filename + ".jpg"), i_rot)
