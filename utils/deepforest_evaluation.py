import numpy as np 
import pandas as pd

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


def point_in_box(point, box, scale = 100):
  x = point[0]
  y = point[1]
  w = box[2] - box[0]
  h = box[3] - box[1]
  if(box[0]+w/scale<x<box[2]-w/scale):
    if(box[1]+h/scale<y<box[3]-h/scale):
      return(True)
  else:
    return(False)

def evaluate_pt(X, hand_centers):
  k = 0
  for point in hand_centers:
    for box in X:
      if(point_in_box(point, box)):
        k +=1
  return(k/len(hand_centers))


def to_detection_type(row):
  if row.IoU>=0.5:
      val = 'TP' #a ground truth object mask had no associated predicted object mask.
  # FN: Ground truth mask has no corresponding predicted mask. We failed to identify this object.
  else:
      val = 'FP'
  # TP: Ground truth mask has a corresponding predicted mask which has an IoU that exceeds the threshold value.
  return val

def evaluate_iou(X, gt):
  iou = np.zeros((len(X),len(gt)))
  for i in range(len(X)):
    for j in range(len(gt)):
      iou[i][j] = bb_intersection_over_union(X[i],gt[j])
  iou = np.array(iou)
  res = np.max(iou, axis = 1)

  df = pd.DataFrame(data = res, columns = ['IoU'])
  df['type'] = df.apply(to_detection_type, axis=1)
  return(df)

def precision(df):
  n_TP = df.type[df.type=='TP'].count()
  n_FP = df.type[df.type=='FP'].count()
  return(n_TP/(n_TP + n_FP))

def recall(df, gt):
  n_TP = df.type[df.type=='TP'].count()
  n_FN = df.type[df.type=='FN'].count()
  n_FP = df.type[df.type=='FP'].count()
  return(n_TP/len(gt))