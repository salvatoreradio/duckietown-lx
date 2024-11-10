from typing import Tuple


def DT_TOKEN() -> str:
    # TODO: change this to your duckietown token
    dt_token = "dt1-3nT7FDbT7NLPrXykNJmqrV9gSAYJU76VYyp8errDMygutk4-43dzqWFnWd8KBa1yev1g3UKnzVxZkkTbfVxQFcoE5td5uoo6fTTrQngxe1Zpr7noC4"
    return dt_token


def MODEL_NAME() -> str:
    # TODO: change this to your model's name that you used to upload it on google colab.
    # if you didn't change it, it should be "yolov5n"
    return "yolov5n"


def NUMBER_FRAMES_SKIPPED() -> int:
    # TODO: change this number to drop more frames
    # (must be a positive integer)
    return 2


def filter_by_classes(pred_class: int) -> bool:
    """
    Remember the class IDs:

        | Object    | ID    |
        | ---       | ---   |
        | Duckie    | 0     |
        | Cone      | 1     |
        | Truck     | 2     |
        | Bus       | 3     |


    Args:
        pred_class: the class of a prediction
    """
    # Right now, this returns True for every object's class
    # TODO: Change this to only return True for duckies!
    # In other words, returning False means that this prediction is ignored.
    if pred_class == 0:
        return True
    else:
        return False


def filter_by_scores(score: float) -> bool:
    """
    Args:
        score: the confidence score of a prediction
    """
    # Right now, this returns True for every object's confidence
    # TODO: Change this to filter the scores, or not at all
    # (returning True for all of them might be the right thing to do!)
    if score > 0.65:
        return True
    else:
        return False


def filter_by_bboxes(bbox: Tuple[int, int, int, int]) -> bool:
    """
    Args:
        bbox: is the bounding box of a prediction, in xyxy format
                This means the shape of bbox is (leftmost x pixel, topmost y, rightmost x, bottommost y)
    """
    # TODO: Like in the other cases, return False if the bbox should not be considered.
    bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
    roi = [0, 200, 416, 416]
    roi_area = (roi[3] - roi[1]) * (roi[2] - roi[0])  #76.800
    
    x_inter_min = max(bbox[0], roi[0])
    y_inter_min = max(bbox[1], roi[1])
    x_inter_max = min(bbox[2], roi[2])
    y_inter_max = min(bbox[3], roi[3])
    
    inter_width = max(0, x_inter_max - x_inter_min)
    inter_height = max(0, y_inter_max - y_inter_min)
    intersection_area = inter_width * inter_height
    
    union_area = bbox_area + roi_area - intersection_area
    iou = intersection_area / union_area
    
    if bbox_area < 5000:
        return False
    
    if union_area == 0:
        return False
    
    if iou < 0.05:
        return False
    
    else:
        return True
