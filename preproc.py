import os
import argparse
import cv2 as cv
import xml.etree.cElementTree as ET
import glob
from xml.dom import minidom

parser = argparse.ArgumentParser(description='preproc')
# parser.add_argument('-i', '--input', help='Input path')
parser.add_argument('-r', '--resize',
                    help='Do resize. Input: folder path')
parser.add_argument('-p', '--parse', nargs=2,
                    help="Parse xml files annotations from LabelImg software." \
                         " Inputs: xmls folder path, xml output path/name")


def do_resize(path):
    out_dir = "tmp_small"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    imgs = os.listdir(path)
    for im in imgs:
        imgorig = cv.imread('{}/{}'.format(path, im))
        imgres = cv.resize(imgorig, (200, 150), interpolation=cv.INTER_AREA)
        print("Shape: ", imgorig.shape)
        print("Shape: ", imgres.shape)
        name = im.split('.')[0]
        cv.imwrite('{}/{}.jpg'.format(out_dir, name), imgres)


def do_parse_annotations(xmls_path_in, xml_path_out):
    """
    if annotations were made with LabelImg this function is to
    elaborate a xml which contains all annotations in one single file
    :param xmls_path_in: folder path with all xml files
    :param xml_path_out: name for the xml generated
    :return: None
    """
    label = "puzzle piece"
    new_xml = "<?xml version='1.0' encoding='ISO-8859-1'?>"
    new_xml += "<dataset>\n<name>Puzzle dataset</name>\n<comment>Created by imglab tool.</comment>\n"
    new_xml += "<images>\n"
    for i, annot in enumerate(glob.glob(xmls_path_in + "/*.xml")):
        tree = ET.parse(annot)
        root = tree.getroot()
        left = root[6][4][0].text
        top = root[6][4][1].text
        width = 70#str(int(root[6][4][2].text) - int(left))
        height = 70#str(int(root[6][4][3].text) - int(top))
        new_xml += "{}<image file='{}jpg'>\n".format(1 * ' ', annot[:-3])
        new_xml += "{}<box top='{}' left='{}' width='{}' height='{}'>\n".format(2 * ' ', top, left, width, height)
        new_xml += "{}<label>{}</label>\n{}</box>\n'{}</image>\n".format(3 * ' ', label, 2 * ' ', 1 * ' ')

    new_xml += "</images>\n</dataset>"
    out_file = open(xml_path_out, "w")
    out_file.write(new_xml)
    out_file.close()


def _prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def main():
    args = parser.parse_args()
    imgs_path = args.resize
    xmls_path = args.parse

    if imgs_path is not None:
        do_resize(imgs_path)
    elif xmls_path is not None:
        do_parse_annotations(xmls_path[0], xmls_path[1])


if __name__ == '__main__':
    main()
