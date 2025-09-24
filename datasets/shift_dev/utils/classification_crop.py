from typing import Sequence
from torch import Tensor
from scalabel.label.typing import Box2D, ImageSize
from torchvision.transforms.functional import crop, resize


def get_bbox_dimensions(bbox: Box2D) -> Sequence[int]:
    w = bbox.x2 - bbox.x1
    h = bbox.y2 - bbox.y1
    return w, h

def get_bbox_size(bbox: Box2D) -> int:
    w, h = get_bbox_dimensions(bbox)
    return w * h
    
def squarify_bbox(bbox: Box2D, img_size: ImageSize) -> Sequence[int]:
    """
    Rescales a X1Y1X2Y2 bbox such that it is a square and padded with 5 pixels on each side. Longest side of the bbox
    is used as the side of the square.
    """
    # TODO: check whether this doesn't sometimes give weird bbox if W >> H or vice versa
    # TODO: maybe this should be gray/white padded rather than extra image?

    padding = 5
    
    w, h = get_bbox_dimensions(bbox)

    size = w if w > h else h
    
    size += padding * 2
    half_size = size // 2
    if half_size > 400:
        half_size = 400  # Size of images is 1280 * 800: size can be max 800
        size = 800

    cx, cy = bbox.x1 + w // 2, bbox.y1 + h // 2
    xb, xe, yb, ye = cx - half_size, cx + half_size, cy - half_size, cy + half_size

    if xb < 0:
        xe += -1 * xb
        xb = 0
    if xe >= img_size.width:
        xb -= (xe - img_size.width - 1)
        xe = img_size.width - 1
    if yb < 0:
        ye += -1 * yb
        yb = 0
    if ye >= img_size.height:
        yb -= (ye - img_size.height - 1)
        ye = img_size.height - 1

    return int(yb), int(xb), int(ye - yb), int(xe - xb)

def get_cropped_object_image(image: Tensor, bbox: Box2D, img_size: ImageSize, cropped_img_size: int) -> Tensor:
    top, left, height, width = squarify_bbox(bbox, img_size)
    image = crop(image, top, left, height, width)
    image = resize(image, (cropped_img_size, cropped_img_size))
    return image
