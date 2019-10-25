__doc__ = """
Reference:
- https://stackoverflow.com/questions/46731947/detect-angle-and-rotate-an-image-in-python  # noqa
- https://stackoverflow.com/questions/33698068/align-text-for-ocr
"""

import math

import cv2
import numpy as np
from scipy import ndimage


def load_img(img_fn):
    img = cv2.imread(img_fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def preprocess_img(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 5, 1
    )
    return thresh


def detect_lines(img):
    img_edges = cv2.Canny(img, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0,
                            100, minLineLength=100, maxLineGap=5)

    return lines


def cal_mask(img, thresh, scale_height=30, scale_long=20):
    horizal = thresh
    vertical = thresh

    scale_height = scale_height
    scale_long = scale_long

    long = int(img.shape[1] / scale_long)
    height = int(img.shape[0] / scale_height)

    horizalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (long, 1))
    horizal = cv2.erode(horizal, horizalStructure, (-1, -1))
    horizal = cv2.dilate(horizal, horizalStructure, (-1, -1))

    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))

    return horizal, vertical


def cal_angles(img, lines):
    angles = []
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
    return angles


def rot_img(img, angles):
    median_angle = np.median(angles)
    img_rotated = ndimage.rotate(img, median_angle)
    return img_rotated


def main(img_fp):
    img = load_img(img_fp)
    lines = detect_lines(img)
    angles = cal_angles(img, lines)
    r_img = rot_img(img, angles)
    return r_img

    # OR
    # img = load_img(img_fp)
    # thresh = preprocess_img(img)
    # horizal, _ = cal_mask(img, thresh)
    # lines = detect_lines(horizal)
    # angles = cal_angles(img, lines)
    # r_img = rot_img(img, angles)
    # return r_img


if __name__ == '__main__':
    img_fp = "image-path.png"
    result = main(img_fp)
