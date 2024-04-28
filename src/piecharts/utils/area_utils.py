import numpy as np
import math

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns clockwise the angle in radians between vectors 'v1' and 'v2'::

    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    
    dot = np.dot(v1_u,v2_u)
    cross = np.cross(v1_u,v2_u)
    
    angle = np.arctan2(cross,dot)
    
    return np.mod((2 * np.pi - angle), (2 * np.pi))


def convert_coord(c, height):
    
    return c[0], height - c[1]


def find_top(centered_arcs):
    
    right_points = [c for c in centered_arcs if c[0] >= 0]
    
    if len(right_points) > 0:
        top = max(right_points, key=lambda c: (c[1], -c[0]))
        return top
    left_points = [c for c in centered_arcs if c[0] < 0]
    return min(left_points, key=lambda c: (c[1], -c[0]))


def compute_all_radius(centers, arc_points):
    radius = []
    
    for c_x, c_y in centers:
        for p_x, p_y in arc_points:
            r_ = np.sqrt((c_x - p_x)**2 + (c_y - p_y)**2)
            if r_ == 94.54099639838793:
                plt.scatter(p_x, height - p_y, c='r')
                plt.scatter(c_x, height - c_y, c='r')
            radius.append(r_)
            
    return radius

def center_arcs(arc_points, center):
    center_x, center_y = center

    centered_arcs = []
    for x, y in arc_points:
        centered_arcs.append((x - center_x, y - center_y))
    return centered_arcs

def get_angle_dict(ref, centered_arcs):
    angle_dict = {}
    for c in centered_arcs:
        angle = angle_between(ref, c)

        angle_dict[angle] = c
    return angle_dict


def get_count(all_r, threshold):
    max_count = 0
    for r_base in all_r:
        count = 0
        for r in all_r:
            if abs((r-r_base)/r_base) <= threshold:
                count += 1
        if count > max_count:
            max_count = count
            record_r = r_base
    return record_r, max_count


def binary_search(all_r, tar_count):
    lt = 0.05
    lr = 0.2
    record_r, count = get_count(all_r, (lt+lr)/2)
    while (lr-lt) > 1e-3:
        if count < tar_count:
            lt = (lr+lt)/2
        else:
            lr = (lr+lt)/2
        record_r, count = get_count(all_r, (lt + lr) / 2)
    return record_r, lt


def sector_area(start, end):
    
    start_x, start_y = start
    end_x, end_y = end


    angle = angle_between(start, end)

    return angle


def cross(a):
    center_x, center_y = center
    left_x, left_y = a
    x1 = left_x - center_x
    y1 = left_y - center_y
    theta_y = math.degrees(math.acos((-y1 / math.sqrt(x1 * x1 + y1 * y1))))
    if x1 < 0:
        theta_y = 360 - theta_y
    return theta_y


def cal_dis(a, b):
    return np.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2))

def multi_center(center_points, arc_points):
    rads = compute_all_radius(center_points, arc_points)
    r_star, t = binary_search(rads, len(arc_points))
    
    global center
    center = center_points[0]
    key_points = sorted(arc_points, key=cross, reverse=True)
    angles = []
    
    for i in range(len(key_points)):
        key_point = key_points[i]
        for j in range(len(center_points)):
            r_ = cal_dis(key_point, center_points[j])
            if abs((r_-r_star)/r_star) <= t:
                tar_center = center_points[j]
                break
        r_ = cal_dis(key_points[(i+1)%len(key_points)], tar_center)
        if abs((r_ - r_star) / r_star) <= t:
            arcs_ = [key_points[i], key_points[(i+1)%len(key_points)]]
            
            centered_arcs_ = center_arcs(arcs_, tar_center)
            angle = sector_area(*centered_arcs_)
            angles.append(angle)
    percentages = [100 * angle / (2 * np.pi) for angle in angles]
    return percentages

    
def arcs_to_sectors(centers, arc_points):
    if len(centers) == 1:
        
        centered_arcs = center_arcs(arc_points, centers[0])

        top_point = find_top(centered_arcs)

        angle_dict = get_angle_dict(top_point, centered_arcs)

        clock_wise_sorted_arcs = [angle_dict[angle] for angle in sorted(angle_dict.keys())]
        clock_wise_sorted_arcs.append(top_point)

        angles = []
        for i in range(len(clock_wise_sorted_arcs) - 1):

            angle = sector_area(clock_wise_sorted_arcs[i], clock_wise_sorted_arcs[i + 1])
            angles.append(angle)

        percentages = [100 * angle / (2 * np.pi) for angle in angles]
        return percentages
    else:
        return multi_center(centers, arc_points)


def get_sectors(centers, arc_points, height):
    centers_set = set()

    arcs_set = set()

    for center in centers:
        centers_set.add(convert_coord(center, height))
    for arc in arc_points:
        arcs_set.add(convert_coord(arc, height))

    arcs = list(arcs_set)
    centers = list(centers_set)[0:1]

    sectors = arcs_to_sectors(centers, arcs)
    return sectors

def get_score(predicted, truth):

    s = {}
    def score(i, j):
        if (i, j) in s:
            return s[(i, j)]
        if i == -1 or j == -1:
            return 0


        s_ = max(score(i - 1, j), score(i, j - 1), score(i - 1, j - 1) + 1 - np.abs((predicted[i] - truth[j]) / truth[j]))
        s[(i, j)] = s_
        return s_

    return score(len(predicted) - 1, len(truth) - 1) / len(truth)  
