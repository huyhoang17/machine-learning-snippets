import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from skimage import measure

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def get_centers(mask):
    """find center points by using contour method
    :return: [(y1, x1), (y2, x2), ...]
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
        else:
            cx, cy = cnt[0][0]
        cy = int(np.round(cy))
        cx = int(np.round(cx))
        centers.append([cy, cx])
    centers = np.array(centers)
    return centers


def make_contours(masks, flatten=True):
    """
    flatten: follow by coco's api
    """
    if masks.ndim == 2:
        masks = np.expand_dims(masks, axis=-1)

    masks = masks.transpose((2, 0, 1))

    segment_objs = []
    for mask in masks:
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            contour = np.flip(contour, axis=1)
            if flatten:
                segmentation = contour.ravel().tolist()
            else:
                segmentation = contour.tolist()
            segment_objs.append(segmentation)

    return segment_objs


def vis_pred_center(pred_center, rad=2, img_size=(512, 512)):

    center_points = get_centers(pred_center.astype(np.uint8))

    img = np.zeros((512, 512))
    pil_img = Image.fromarray(img).convert('RGBA')
    center_canvas = Image.new('RGBA', pil_img.size)
    center_draw = ImageDraw.Draw(center_canvas)

    for point in center_points:
        y, x = point
        # x1, y1, x2, y2
        center_draw.ellipse(
            (x - rad, y - rad, x + rad, y + rad), fill='blue', outline='blue'
        )

    res_img = Image.alpha_composite(pil_img, center_canvas)
    res_img = res_img.convert("RGB")
    res_img = np.asarray(res_img)

    return res_img


def vis_pred_bbox(pred_bbox, cons):
    mask_ = Image.new('1', (512, 512))
    mask_draw = ImageDraw.ImageDraw(mask_, '1')

    for contour in cons:
        mask_draw.polygon(contour, fill=1)

    mask_ = np.array(mask_).astype(np.uint8)
    return mask_ * 255


def load_img(img_fp):
    img = cv2.imread(img_fp, 0)
    img = img / 255
    img = img.astype(np.float32)
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img


if __name__ == '__main__':
    bbox_fp = "./images/pred_bbox.jpg"
    center_fp = "./images/pred_center.jpg"

    pred_bbox = load_img(bbox_fp)
    pred_center = load_img(center_fp)

    start = time.time()
    # get all polygon area in image
    cons = make_contours(pred_bbox)

    # get all center points by contour method
    centers = get_centers(pred_center.astype(np.uint8))

    # checking if polygon contains point
    final_cons = []
    for con in cons:
        polygon = Polygon(zip(con[::2], con[1::2]))
        for center in centers:
            point = Point(center[1], center[0])
            if polygon.contains(point):
                final_cons.append(con)
                break

    print(len(cons), len(final_cons))

    res_img = vis_pred_bbox(pred_bbox, final_cons)
    print(time.time() - start)
    plt.imshow(res_img, cmap='inferno', alpha=0.60)
    plt.imshow(pred_center, cmap='inferno', alpha=0.80)
    plt.savefig('./images/filtered_bbox_center.png', bbox_inches='tight')
