import numpy as np
import cv2

def cv2_show(img, header="Bounding circle"):
    cv2.imshow(header, img)
    cv2.waitKey(0)


def non_maximum_supression(heat_map, threshold, merge_distance):
    points = np.array(np.nonzero(heat_map > threshold)).T

    collision = True
    while collision:
        collision = False
        result_points = []
        for i in range(len(points)):
            merge = False
            for j in range(i+1, len(points)):
                dif = points[i] - points[j]
                dist = dif[0]*dif[0] + dif[1]*dif[1]
                if dist <= merge_distance**2: 
                    merge=True
                    collision = True
                    break
            if not merge: # ... if i doesn't collide, add it. If it does, ignore it -> j will get added instead
                result_points.append(points[i])
        points = result_points
    return points
