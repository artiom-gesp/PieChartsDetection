from PIL import Image
import glob
from skimage import filters
from matplotlib import pyplot as plt
import skimage
import numpy as np
import cv2
import numba
from math import pi
import time
import os
from utils import find_edges, load_image, cv2_show
import json


@numba.njit(fastmath=True)
def compute_score(edges, x, y, r, step):
    if x < r or x + r >= edges.shape[0] or y < r or y + r >= edges.shape[1]: return 0

    samples = int(2 * pi * r)
    score = 0.0
    angles = np.linspace(0, 2 * pi * (1-samples)/samples, samples)
    for i in numba.prange(samples):
        angle = angles[i]
        s, c = np.sin(angle), np.cos(angle)
        e = 0.0
        for curr in np.linspace(max(0, r - step), r + step, int(step + max(step, r))):
            curx, cury = int(x+s*curr), int(y+c*curr)
            if curx < 0 or cury < 0 or curx >= edges.shape[0] or cury >= edges.shape[1]:
                break
            e = max(e, edges[curx, cury])
        score += e
    score /= samples * 255
    return score

@numba.njit(parallel=True)
def find_circle(edges, xs, ys, rs, step):
    maxr = np.max(rs)
    scores = np.zeros_like(xs, np.float32)
    scores_regularized = np.zeros_like(xs, np.float32)
    for i in numba.prange(len(xs)):
        scores[i] = compute_score(edges, xs[i], ys[i], rs[i], step) 
        scores_regularized[i] = scores[i] + rs[i] / maxr * radius_regularization_k
    return scores, scores_regularized
    

@numba.njit(fastmath=True)
def compute_score_arc(edges, x, y, r, step):
    if x < r or x + r >= edges.shape[0] or y < r or y + r >= edges.shape[1]: return 0, (0, 0)

    samples = int(2 * pi * r)
    score = 0.0
    angles = np.linspace(0, 2 * pi * (1-samples)/samples, samples)
    measurements = np.zeros((samples,), np.float32)

    for i in range(samples):
        angle = angles[i]
        s, c = np.sin(angle), np.cos(angle)
        e = 0.0
        for curr in np.linspace(max(0, r - step), r + step, int(step + max(step, r))):
            curx, cury = int(x+s*curr), int(y+c*curr)
            if curx < 0 or cury < 0 or curx >= edges.shape[0] or cury >= edges.shape[1]:
                break
            e = max(e, edges[curx, cury])
        measurements[i] = e

    csum = np.cumsum(measurements / 255.0)
    
    best_a_b = None
    best_dif = 0
    best_score = 0

    for a in range(samples):
        for b in range(samples):
            if a == b: continue
            if b > a:
                dif = (b-a)
                # forward measure
                score = (csum[b] - csum[a]) / dif
            else:
                dif = (samples-a+b)
                score = (csum[-1] - csum[a] + csum[b]) / dif
            dif /= samples

            if score == 1.0 and dif > best_dif:
                best_dif = dif
                best_a_b = (a / samples, b / samples)
                best_score = score
    return (best_score, best_a_b) if best_dif > 0.4 else (0, (0, 0))



@numba.njit(parallel=True)
def find_arc(edges, xs, ys, rs, step):
    maxr = np.max(rs)
    ab_s = np.zeros((xs.shape[0], 2), np.float32)
    scores = np.zeros_like(xs, np.float32)
    scores_regularized = np.zeros_like(xs, np.float32)
    for i in numba.prange(len(xs)):
        score, ab = compute_score_arc(edges, xs[i], ys[i], rs[i], step)
        scores[i] = score 
        scores_regularized[i] = scores[i] + rs[i] / maxr * radius_regularization_k
        ab_s[i] = ab
    return scores, scores_regularized, ab_s



target_folder = "images/images_processed"

image_list = glob.glob("think-cell-datathon/images/images/*")


failed_images = []

preview = False
radius_regularization_k = 0.15

stop_on_fail = False

failed_threshold = 0.9



VARIANT_CIRCLE, VARIANT_ARC = 0, 1




def preview_circle(img, x, y, r):
    preview_image = np.copy(img)
    cv2.circle(preview_image, (int(y), int(x)), int(r), (0,255,0), 2)
    cv2_show(preview_image)

def preview_circles(img, x, y, r):
    preview_image = np.copy(img)
    for i in range(x.shape[0]):
        cv2.circle(preview_image, (int(y[i]),int(x[i])), int(r[i]), (0,255,0), 2)
    cv2_show(preview_image)



STEP_CIRCLE_FIRST, STEP_CIRCLE_SECOND, STEP_ARC = 0, 1, 2

def compute_bounding_circle(original_image, edges, step_images, variant, step_name):
    step = 20

    xs, ys, rs = np.meshgrid(range(0, edges.shape[0], step), range(0, edges.shape[1], step), range(50, max(edges.shape[0], edges.shape[1]) // 2, step))
    
    ms_shape = xs.shape
    xs = xs.ravel()
    ys = ys.ravel()
    rs = rs.ravel()

    steps = 13 if variant == VARIANT_ARC else 6
    convergence_k = 1.25 if variant == VARIANT_ARC else 1.5

    for _ in range(steps):
        if variant == VARIANT_CIRCLE:
            scores, scores_regularized = find_circle(edges, xs, ys, rs, step)
        else:
            scores, scores_regularized, ab_s = find_arc(edges, xs, ys, rs, step)
            
        best_is = np.argsort(scores_regularized)[::-1]

        best_is = best_is[:step_images]

        if preview:
            preview_circles(original_image, xs[best_is], ys[best_is], rs[best_is])

        a_, b_, c_ = np.zeros((0,)), np.zeros((0,)), np.zeros((0))

        for j in best_is:
            new_xs = [xs[j] - step/2, xs[j] + step/2]
            new_ys = [ys[j] - step/2, ys[j] + step/2]
            new_rs = [rs[j] - step/2, rs[j] + step/2]

            a, b, c = np.meshgrid(new_xs, new_ys, new_rs)
            a_ = np.concatenate([a_, a.ravel()])
            b_ = np.concatenate([b_, b.ravel()])
            c_ = np.concatenate([c_, c.ravel()])
        
        xs = np.array(a_)
        ys = np.array(b_)
        rs = np.array(c_)
        
        step /= convergence_k

    best_i = best_is[0]
    best_score = scores[best_i]

    failed = best_score < failed_threshold

    if preview:
        preview_circle(original_image, xs[best_i], ys[best_i], rs[best_i])
    
    if failed:
        print (f"Immediate method failed, trying more advanced version!")
        if step_name == STEP_CIRCLE_FIRST:
            return compute_bounding_circle(original_image, edges, 500, VARIANT_CIRCLE, STEP_CIRCLE_SECOND)
        if step_name == STEP_CIRCLE_SECOND:
            return compute_bounding_circle(original_image, edges, 1000, VARIANT_ARC, STEP_ARC)
        return 0, 0, 1, 0, 0

    return (xs[best_i], ys[best_i], rs[best_i], best_score, ab_s[best_i] if variant==VARIANT_ARC else None)


circle_params_dict = {}

with open("circle_params_dict.json") as file:
    circle_params_dict = json.load(file)

# ... dataset backup.
# stable_dataset = glob.glob("think-cell-datathon/images/images_processed/*")
# with open("stable_dataset_list2222.txt", "w") as backup_file:
#     backup_file.write("\n".join(stable_dataset))

for img_index, img_fname in enumerate(image_list):
    img_base_name = img_fname.split("\\")[-1]

    target_fname = img_fname.replace("images/images", target_folder)
    if os.path.exists(target_fname):
        print (f"Skipping {img_index}")
        continue

    # img = np.average(skimage.io.imread(img_fname), 2)
    # edge_sobel = filters.sobel(img)

    # fix, (ax1, ax2) = plt.subplots(1, 2)

    # ax1.imshow(img)
    # ax2.imshow(edge_sobel)

    # plt.show()
    #import cv2.cv as cv
    original_image = load_image(img_fname)
    edges = find_edges(original_image) #cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    if preview:
        cv2.imshow('edges', edges)
        cv2.waitKey(0)

    
    x, y, r, score, ab_ =  compute_bounding_circle(original_image, edges, 50, VARIANT_CIRCLE, STEP_CIRCLE_FIRST)
    
    circle_params_dict[img_base_name] = (x, y, r)

    if score < failed_threshold:
        print (f"Failed on image {img_fname}!")
        failed_images.append(img_fname)
        with open("failed_images.txt", "w") as f:
            f.write(str(failed_images))
        if stop_on_fail:
            preview_circle(original_image, x, y, r)
        continue
    
    print (f"Done {img_index+1:0000}/{len(image_list)}, {img_fname}, Best score={score}")
    #plt_scores = scores.reshape(ms_shape)

    # if ab_ is not None:
    #     print(f"AB={ab_}. Not saving.")
    #     preview_circle(original_image, x, y, r)
    #     continue

    
    #edges = cv2.Canny(thresh, 100, 200)
    #cv2.imshow('detected ',gray)
    #cimg=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT_ALT, 1, 10000, param1 = 300, param2 = 0.8, minRadius = 0, maxRadius = 0)
    #for i in circles[0,:]:
    #    cv2.circle(img1,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),2)
    
    #print(f"Benchmarking: 1={time2-time1}, 2={time3-time2}, 3={time4-time3}, 4={time5-time4}")

    coords = np.transpose(np.array(np.meshgrid(np.arange(original_image.shape[0]), np.arange(original_image.shape[1]))), (2, 1, 0)) 
    c1 = coords[..., 0] - x
    c2 = coords[..., 1] - y
    mask = np.sqrt(c1*c1 + c2*c2) <= r

    #sums_x = np.nonzero(np.sum(mask, 1))
    #sums_y = np.nonzero(np.sum(mask, 0))

    #from_x, to_x = np.min(sums_x), np.max(sums_y)
    #from_y, to_y = np.min(sums_y), np.max(sums_y)
    
    result = np.where(mask[..., np.newaxis], original_image, 0).astype(np.uint8)
    cv2.imwrite(target_fname, result)

    #for i in range(plt_scores.shape[-1]):
    #    axs[i // 6, i % 6].imshow(plt_scores[..., i].T)
    #plt.show()

with open("circle_params_dict.json", "w") as file:
    json.dump(circle_params_dict, file)  

