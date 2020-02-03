import cv2


def resize_padding(img, desired_size=640):
    """https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    """

    ratio = float(desired_size) / max(img.shape)
    new_size = tuple([int(dim * ratio) for dim in img.shape[:2]])

    # resize img
    rimg = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # make padding
    color = [0, 0, 0]
    rimg = cv2.copyMakeBorder(rimg, top, bottom, left,
                              right, cv2.BORDER_CONSTANT, value=color)

    return rimg
