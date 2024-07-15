import math
from springc_utils import *
from matplotlib import pyplot as plt

def convert_cossin_to_angle(values):
    # values: (N, 2)
    values = toNumpy(values)
    degrees = []
    for cos, sin in values:
        radians = math.atan2(sin, cos)
        degree = math.degrees(radians)
        degrees.append(round(degree))
    return degrees
    

def show_diff_angle_per1(gt_degrees, diff_degrees):
    angle_bins = np.arange(-180, 181, 1)
    angle_diff_dict = {i: [] for i in angle_bins}

    for gt_angle, diff in zip(gt_degrees, diff_degrees):
        angle_diff_dict[gt_angle].append(diff)

    mean_diffs = [np.mean(angle_diff_dict[angle]) if angle_diff_dict[angle] else 0 for angle in angle_bins]
    plt.figure(figsize=(12, 6))
    plt.bar(angle_bins, mean_diffs, width=1, align='center', edgecolor='black')
    plt.title('Mean Difference for Each Degree Bin (-180, 180)')
    plt.xlabel('True Angle (Degrees)')
    plt.ylabel('Mean Difference')
    plt.grid(True)
    plt.savefig("./assets/diff_angles.png")
