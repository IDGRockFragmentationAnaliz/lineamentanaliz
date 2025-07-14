from contextlib import nullcontext
from pathlib import Path

import numpy as np
import cv2
import scipy as sp
import json
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.vectorized import contains
from pyrockshape import shape_load

from pyrocksegmentation.basic_segmentator import Segmentator
from pyrockstats.distrebutions import lognorm, weibull, paretoexp
from pyrockstats.bootstrap.ks_statistics import get_ks_distribution
from pyrockstats.bootstrap.ks_statistics import get_confidence_value
from pyrockstats.empirical import ecdf
from distrebution_test import DistributionTest



def main():
    lines_type = "120"
    root_path = Path("D:/1.ToSaver/profileimages/ShapeBaikal/lineaments")
    root_path = root_path / ("FABDEM_" + lines_type)
    json_base_segmentation = root_path / ("areas_" + lines_type + ".json")
    json_path_ensemble_segmentation = root_path / ("ensemble_areas_" + lines_type + ".json")
    shape_areas_path = root_path / "areas"
    
    with open(str(json_base_segmentation), 'r') as file:
        data_base = json.load(file)
    
    polies, bbox = shape_load(shape_areas_path)
    
    for i, poly in enumerate(polies):
        print("area number:", i)
        s = get_border_aras(poly, data_base)
        data = get_ks_test_data(s)
        print("lognorm:")
        print_test_data(data["test_data"]["lognorm"])
        print("paretoexp:")
        print_test_data(data["test_data"]["paretoexp"])
        print("weibull:")
        print_test_data(data["test_data"]["weibull"])
        
        headers = []
        distributions = ['lognorm', 'paretoexp', 'weibull']
        metrics = ['Hypothesis', 'd', 'p-value', 'theta1', 'theta2']

def get_ks_test_data(s):
    s = np.delete(s, np.argmax(s))
    
    areas = s
    
    xmin = np.min(areas)
    xmax = np.max(areas)
    
    models = {"lognorm": lognorm, "paretoexp": paretoexp, "weibull": weibull}
    tests = {name: DistributionTest(areas, model) for name, model in models.items()}
    
    values, e_freq = ecdf(areas)
    x = np.logspace(np.log10(xmin), np.log10(xmax), 100)
    alpha = 0.05
    data = {
        "x": x.tolist(),
        "xmin": xmin,
        "xmax": xmax,
        "alpha": alpha,
        "test_data": {name: test.get_data(x, alpha) for name, test in tests.items()},
        "ecdf": {"values": values.tolist(), "freqs": e_freq.tolist()},
        "theta": {name: tests[name].theta for name, test in tests.items()},
    }
    return data


def print_test_data(test_data):
    ks_test = test_data['ks_test']
    d = test_data['d']
    p_value = test_data['p_value']
    theta = test_data["theta"]
    print(
        f"hypothesis: {ks_test}, "
        f"d: {d:.3f}, "
        f"p_value: {p_value:.3f}",
        f"theta: {theta[0]:.3f}, {theta[1]:.3f}",
    )


def get_border_aras(poly, data):
    poly = Polygon(poly)
    areas = np.array(data["areas"])
    centers = np.array(data["centers"])
    mask = contains(poly, centers[:, 0], centers[:, 1])
    return areas[mask]


if __name__ == "__main__":
    main()