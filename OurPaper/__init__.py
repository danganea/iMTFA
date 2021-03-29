# Need to import cv2 before detectron2 libraries, else things crash
import cv2
from typing import List


# Number of coordinates needed to predict a BBOX
BBOX_PRED_COORD_NUMBER = 4
