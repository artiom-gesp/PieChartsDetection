import numpy as np
import cv2


np.random.seed(42)

sample_count = 20
samples_3d = np.random.uniform(0.0, 1.0, (sample_count, 3))
samples_4d = np.random.uniform(0.0, 1.0, (sample_count, 4))


def load_image(fname):
    return cv2.imread(fname, cv2.IMREAD_UNCHANGED)



def find_edges(img):
    result = None
    for i in range(sample_count):
        gray = (img @ samples_3d[i] if img.shape[2] == 3 else img @ samples_4d[i]).astype(np.uint8)
        #cv2.imshow('Gray', gray)
        #cv2.waitKey(0)
        ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(thresh, 100, 200)
        #cv2.imshow('Threshold', edges)
        #cv2.waitKey(0)
        result = edges if result is None else np.maximum(edges, result)
    return result

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


if __name__ == "__main__":
    heat_map = np.zeros([10, 10])
    heat_map[0, 0] = 1
    heat_map[2, 1] = 1
    heat_map[5, 5] = 1
    heat_map[6, 5] = 1
    pts = non_maximum_supression(heat_map, 0.5, 3)

    print(pts)



