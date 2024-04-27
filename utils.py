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