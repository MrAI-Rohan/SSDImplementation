import torch


def transform_cwh_to_xyxy(bboxes, image_size=300, denorm=False):
    """This functin expects a 2D input, even if one bbox to be transformed."""
    cx, cy, w, h = bboxes.T
    xyxy = torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dim=1)
    return xyxy*(image_size) if denorm else xyxy


def transform_xyxy_to_cwh(bboxes, img_size=300, norm=False):
    """This functin expects a 2D input, even if one bbox to be transformed."""
    xmin, ymin, xmax, ymax = bboxes.T
    cwh = torch.stack([(xmax+xmin)/2, (ymax+ymin)/2,
                        xmax-xmin, ymax-ymin], dim=1)
    return cwh/(img_size) if norm else cwh
