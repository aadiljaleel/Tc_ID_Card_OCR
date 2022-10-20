import sys
import PySide2.QtCore as QtC
import PySide2.QtWidgets as QtW
import PySide2.QtGui as QtG

import cv2
import torch
import numpy as np
import os.path
from datetime import datetime

import cv2
from matplotlib import pyplot as plt
import numpy as np
import utlis
import torch
from find_nearest_box import NearestBox
from pytorch_unet.unet_predict import UnetModel
from pytorch_unet.unet_predict import Res34BackBone
from extract_words import OcrFactory
import extract_words
import os
import time
import argparse
import detect_face


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 500

DEFAULT_SPACING = 4
DEFAULT_PADDING = 8


def get_scaled_image(img: QtG.QImage) -> QtG.QImage:
    width, height = img.width(), img.height()

    if width > WINDOW_WIDTH * 0.8 or height > WINDOW_HEIGHT * 0.75:
        if width > height:
            new_width = WINDOW_WIDTH * 0.8
            new_height = (height / width) * new_width

            return img.scaledToWidth(new_width, QtC.Qt.TransformationMode.SmoothTransformation)
        else:
            new_height = WINDOW_HEIGHT * 0.75
            new_width = (width / height) * new_height
            width, height = new_width, new_height

            return img.scaledToHeight(new_height, QtC.Qt.TransformationMode.SmoothTransformation)
    else:
        return img


def getCenterRatios(img, centers):
    """
    Calculates the position of the centers of all boxes
    in the ID card image and Unet Mask relative to the width and height of the image
    and returns these ratios as a numpy array.
    """
    if(len(img.shape) == 2):
        img_h, img_w = img.shape
        ratios = np.zeros_like(centers, dtype=np.float32)
        for i, center in enumerate(centers):
            ratios[i] = (center[0]/img_w, center[1]/img_h)
        return ratios
    else :
        img_h, img_w,_ = img.shape
        ratios = np.zeros_like(centers, dtype=np.float32)
        for i, center in enumerate(centers):
            ratios[i] = (center[0]/img_w, center[1]/img_h)
        return ratios


def matchCenters(ratios1, ratios2):
    """
    It takes the ratio of the centers of the regions
    included in the mask and CRAFT result on the image
    and maps them according to the absolute distance.
    Returns the index of the centers with the lowest absolute difference accordingly
    """

    bbb0 = np.zeros_like(ratios2)
    bbb1 = np.zeros_like(ratios2)
    bbb2 = np.zeros_like(ratios2)
    bbb3 = np.zeros_like(ratios2)

    for i , r2 in enumerate(ratios2):
        bbb0[i] = abs(ratios1[0] - r2)
        bbb1[i] = abs(ratios1[1] - r2)
        bbb2[i] = abs(ratios1[2] - r2)
        bbb3[i] = abs(ratios1[3] - r2)

    sum_b0 = np.sum(bbb0, axis = 1)
    sum_b0 = np.reshape(sum_b0, (-1, 1))
    arg_min_b0 = np.argmin(sum_b0, axis=0)

    sum_b1 = np.sum(bbb1, axis = 1)
    sum_b1 = np.reshape(sum_b1, (-1, 1))
    arg_min_b1 = np.argmin(sum_b1, axis=0)

    sum_b2 = np.sum(bbb2, axis = 1)
    sum_b2 = np.reshape(sum_b2, (-1, 1))
    arg_min_b2 = np.argmin(sum_b2, axis=0)

    sum_b3 = np.sum(bbb3, axis = 1)
    sum_b3 = np.reshape(sum_b3, (-1, 1))
    arg_min_b3 = np.argmin(sum_b3, axis=0)

    return np.squeeze(arg_min_b0), np.squeeze(arg_min_b1), np.squeeze(arg_min_b2),np.squeeze(arg_min_b3)



def getCenterOfMasks(thresh):
    """
    Find centers of 4 boxes in mask from top to bottom with unet model output and return them
    """

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by size from smallest to largest
    contours = sorted(contours, key = cv2.contourArea, reverse=False)

    contours = contours[-4:] # get the 4 largest contours

    #print("size of cnt", [cv2.contourArea(cnt) for cnt in contours])
    boundingBoxes = [cv2.boundingRect(c) for c in contours]

    # Sort the 4 largest regions from top to bottom so that we filter the relevant regions
    (cnts, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),key=lambda b:b[1][1], reverse=False))

    detected_centers = []

    for contour in cnts:
        (x,y,w,h) = cv2.boundingRect(contour)
        #cv2.rectangle(thresh, (x,y), (x+w,y+h), (255, 0, 0), 2)
        cX = round(int(x) + w/2.0)
        cY = round(int(y) + h/2.0)
        detected_centers.append((cX, cY))
        #cv2.circle(thresh, (cX, cY), 7, (255, 0, 0), -1)

    return np.array(detected_centers)


def getBoxRegions(regions):
    """
    The coordinates of the texts on the id card are converted
    to x, w, y, h type and the centers and coordinates of these boxes are returned.
    """
    boxes = []
    centers = []
    for box_region in regions:

        x1,y1, x2, y2, x3, y3, x4, y4 = np.int0(box_region.reshape(-1))
        x = min(x1, x3)
        y = min(y1, y2)
        w = abs(min(x1,x3) - max(x2, x4))
        h = abs(min(y1,y2) - max(y3, y4))

        cX = round(int(x) + w/2.0)
        cY = round(int(y) + h/2.0)
        centers.append((cX, cY))
        bbox = (int(x), w, int(y), h)
        boxes.append(bbox)

    #print("number of detected boxes", len(boxes))
    return np.array(boxes), np.array(centers)


def id_card_ocr(cv2_img, neighbor_box_distance: float, rotation_interval: int, face_recognition: str, ocr_method: str):
    ORI_THRESH = 3 # Orientation angle threshold for skew correction
    use_cuda = "cuda" if torch.cuda.is_available() else "cpu"
    model = UnetModel(Res34BackBone(), use_cuda)
    nearestBox = NearestBox(distance_thresh = neighbor_box_distance, draw_line=False)
    face_detector = detect_face.face_factory(face_model = face_recognition)
    findFaceID = face_detector.get_face_detector()
    Image2Text =  OcrFactory().select_ocr_method(ocr_method = ocr_method, border_thresh=3, denoise = False)

    img1 = cv2.cvtColor(cv2_img , cv2.COLOR_BGR2RGB)
    final_img = findFaceID.changeOrientationUntilFaceFound(img1, rotation_interval)

    if(final_img is None):
        return None, None

    final_img = utlis.correctPerspective(final_img)
    txt_heat_map, regions = utlis.createHeatMapAndBoxCoordinates(final_img)
    txt_heat_map = cv2.cvtColor(txt_heat_map, cv2.COLOR_BGR2RGB)
    predicted_mask = model.predict(txt_heat_map)
    orientation_angle = utlis.findOrientationofLines(predicted_mask.copy())

    if ( abs(orientation_angle) > ORI_THRESH ):
        final_img = utlis.rotateImage(orientation_angle, final_img)

        txt_heat_map, regions = utlis.createHeatMapAndBoxCoordinates(final_img)
        txt_heat_map = cv2.cvtColor(txt_heat_map, cv2.COLOR_BGR2RGB)
        predicted_mask = model.predict(txt_heat_map)


    bbox_coordinates , box_centers = getBoxRegions(regions)

    mask_centers = getCenterOfMasks(predicted_mask)

    # centers ratio for 4 boxes
    centers_ratio_mask = getCenterRatios(predicted_mask, mask_centers)

    # centers ratio for all boxes
    centers_ratio_all = getCenterRatios(final_img, box_centers)
    matched_box_indexes = matchCenters(centers_ratio_mask , centers_ratio_all)
    new_bboxes = nearestBox.searchNearestBoundingBoxes(bbox_coordinates, matched_box_indexes, final_img)
    PersonInfo = Image2Text.ocrOutput("img", final_img, new_bboxes)

    utlis.displayMachedBoxes(final_img, new_bboxes)
    utlis.displayAllBoxes(final_img, bbox_coordinates)

    return final_img, np.multiply(predicted_mask, 255), PersonInfo


class ImageViewingWindow(QtW.QWidget):
    def __init__(self, cv2_img) -> None:
        super().__init__()

        self.setWindowTitle("TC ID Card OCR")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

        is_grayscale = len(cv2_img.shape) == 2
        height, width = cv2_img.shape[0], cv2_img.shape[1]
        bytes_per_line = width
        if len(cv2_img.shape) == 3:
            bytes_per_line *= cv2_img.shape[2]


        self.image = get_scaled_image(
            QtG.QImage(
                cv2_img.data,
                width,
                height,
                bytes_per_line,
                QtG.QImage.Format_RGB888 if not is_grayscale else QtG.QImage.Format_Grayscale8
            )
        )

        self.vbox = QtW.QVBoxLayout()
        self.image_viewer_label = QtW.QLabel("")
        self.image_viewer_label.setPixmap(QtG.QPixmap.fromImage(self.image))

        self.hbox_row_1 = QtW.QHBoxLayout()
        self.hbox_row_1.addStretch()
        self.hbox_row_1.addWidget(self.image_viewer_label)
        self.hbox_row_1.addStretch()

        self.vbox.addStretch()
        self.vbox.addLayout(self.hbox_row_1)
        self.vbox.addStretch()

        self.setLayout(self.vbox)


class MainWindow(QtW.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("TC ID Card OCR")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

        self.vbox = QtW.QVBoxLayout()

        self.hbox_row_1 = QtW.QHBoxLayout()
        self.hbox_row_2 = QtW.QHBoxLayout()
        self.hbox_row_3 = QtW.QHBoxLayout()

        self.vbox.addStretch()
        self.vbox.addLayout(self.hbox_row_1)
        self.vbox.addStretch()
        self.vbox.addLayout(self.hbox_row_2)
        self.vbox.addLayout(self.hbox_row_3)


        self.image_filename = None
        self.image = QtG.QImage()
        self.scaled_image = QtG.QImage()
        self.image_viewer_label = QtW.QLabel("")
        self.image_viewer_label.installEventFilter(self)
        self.image_viewer_label.setPixmap(QtG.QPixmap.fromImage(self.image))
        self.hbox_row_1.addStretch()
        self.hbox_row_1.addWidget(self.image_viewer_label)
        self.hbox_row_1.addStretch()

        self.upload_image_button = QtW.QPushButton("Upload Image")
        self.upload_image_button.clicked.connect(self.upload_image_button_click)
        self.hbox_row_2.insertSpacing(0, int(WINDOW_WIDTH * 0.75))
        self.hbox_row_2.addWidget(self.upload_image_button, 0)

        self.hbox_row_3.addStretch()
        self.convert_button = QtW.QPushButton("Convert")
        self.convert_button.clicked.connect(self.convert_button_clicked)
        self.hbox_row_3.addWidget(self.convert_button)

        self.central_widget = QtW.QWidget()
        self.central_widget.setLayout(self.vbox)
        self.setCentralWidget(self.central_widget)

    def upload_image_button_click(self):
        dialog = QtW.QFileDialog()
        dialog.setFileMode(QtW.QFileDialog.FileMode.ExistingFile)

        filter_string = "PNG/JPG/JPEG File (*.png *.jpg *.jpeg)"
        filename, filter = dialog.getOpenFileName(self, "Upload Image", filter=filter_string)
        if len(filename) > 0:
            self.image.load(filename)
            self.image_filename = filename
            self.scaled_image = get_scaled_image(self.image)
            self.image_viewer_label.setPixmap(QtG.QPixmap.fromImage(self.scaled_image))

    def convert_button_clicked(self):
        byte_array = QtC.QByteArray()
        buffer = QtC.QBuffer(byte_array)
        buffer.open(QtC.QIODevice.ReadWrite)

        self.image.save(buffer, "PNG")
        np_array = np.asarray(bytearray(byte_array.data()), dtype="uint8")
        cv2_img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        cv2_ocr_img, cv2_mask_img, info = id_card_ocr(cv2_img, 60, 60, "ssd", "EasyOcr")

        label_text_list = []
        for id, val in info.items():
            label_text_list.append("{}: {}".format(id, val))

        self.image_viewer_label.setText("\n".join(label_text_list))

        self.ocr_viewer = ImageViewingWindow(cv2_ocr_img)
        self.mask_viewer = ImageViewingWindow(cv2_mask_img)

        self.ocr_viewer.show()
        self.mask_viewer.show()


def main():
    app = QtW.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec_()


if __name__ == "__main__":
    main()
