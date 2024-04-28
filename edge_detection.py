import cv2 as cv
import numpy as np
import argparse
import random as rng
rng.seed(12345)
import glob
from math import pi
from area_utils import get_score
import pandas as pd
import ast
from utils import load_image, find_edges


preview = True

def baseline(val):
    # Detect edges using Canny
    canny_output = cv.Canny(val, 100, 200)
    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Draw contours
    if preview:
        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
        #Show in a window
        cv.imshow('Contours', drawing)
        cv.waitKey(0)

    #split into 72 segments... (allow 5 degree increments)
    seg_count = 30
    segments = {}

    middle = np.array(val.shape[:2]) / 2

    for contour in contours:
        for i in range(len(contour)):
            p1 = contour[i][0]
            p2 = contour[(i+1)%len(contour)][0]
            dif = p2 - p1
            dif_len = np.linalg.norm(dif)
            mid_dir = (p2+p1)/2 - middle

            if dif_len < 1:
                continue

            dot = np.abs(mid_dir / np.linalg.norm(mid_dir) @ dif / dif_len)
            #dot = 1 -> perfect match (add vote to line)
            #else -> ignore...

            if dot > 0.9:
                angle = np.arctan2(mid_dir[1], -mid_dir[0])
                index = int(angle * seg_count / (2 * pi))
                if index not in segments: segments[index] = (0, 0)
                
                segments[index] = (segments[index][0] + dif_len, segments[index][1]+angle*dif_len)
    
    angles = []
    for i, (acount, asum) in segments.items():
        if acount < 40: continue
        angle = asum / acount
        if angle < 0: angle += 2*pi
        angles.append(angle)

    angles = sorted(angles)

    i = 0
    while i < len(angles)-1:
        if (angles[i+1] - angles[i]) / 2 / pi * 100 < 3:
            angles[i] = (angles[i] + angles[i+1]) / 2
            angles = angles[:i+1] + angles[i+2:]
        i += 1



    percentages = []
    for i in range(len(angles)-1):
        percentages.append(100 * (angles[i+1] - angles[i]) / (2 * pi))
    if len(angles):
        percentages.append((2 * pi - angles[-1] + angles[0]) / (2 * pi) * 100)
    else:
        percentages.append(100.0)

    return percentages


def load_pie_data(filepath):
    # Load data from CSV file into a DataFrame
    data = pd.read_csv(filepath)

    # Convert string representation of lists back to actual lists
    list_data_features = ["boxes", "start_angles", "end_angles", "angles", "percentages"]
    for column in list_data_features:
        data[column] = data[column].apply(ast.literal_eval)

    return data

train_df = load_pie_data("think-cell-datathon/train.csv")

results = {}
fnames = glob.glob("think-cell-datathon/images/images_processed/*")

all_scores = []

for i, fname in enumerate(range(train_df.size)):
    src = load_image(f"think-cell-datathon/images/images_processed/{train_df.filename[i]}")
    src_gray = find_edges(src)
    src_gray = cv.blur(src_gray, (3,3))

    sums_x = np.nonzero(np.sum(src_gray , 1))
    sums_y = np.nonzero(np.sum(src_gray , 0))

    from_x, to_x = np.min(sums_x), np.max(sums_y)
    from_y, to_y = np.min(sums_y), np.max(sums_y)

    from_x = max(from_x-5, 0)
    from_y = max(from_y-5, 0)

    to_x += 5
    to_y += 5

    src_gray = src_gray[from_x:to_x+1, from_y:to_y+1]

    pred = baseline(src_gray)
    true = train_df.percentages[i]

    all_scores.append(get_score(pred, true))

    #print(f"Score={get_score(pred, true)}")
    
    #print(f"Done {i+1:0000} out of {train_df.size}")

    if i % 100 == 0: print(f"Score={np.mean(all_scores)}")


if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3,3))
# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)
max_thresh = 255
thresh = 100 # initial threshold
cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.waitKey()