import math
from pathlib import Path
import cv2
import numpy as np
from pyrockstats.distrebutions import lognorm, weibull, paretoexp

import matplotlib.pyplot as plt
from pyrockstats.empirical import ecdf

import json

from pyrockshape import shape_load
from scipy.io import savemat

from lineaments_map import LineamentsMap
from tqdm import tqdm


def main():
    #lines_str_names = ["30",]
    lines1_str = "120"
    root_path = Path("D:/1.ToSaver/profileimages/ShapeBaikal/lineaments")
    shape_path = root_path / ("FABDEM_" + lines1_str)

    with open(str(root_path / "config.json"), 'r') as file:
        data = json.load(file)
        buffer = data[lines1_str]["buffer"]

    line_map = LineamentsMap(shape_path, buffer)
    data = {}
    for i in tqdm(range(500)):
        areas, centers = line_map.get_areas_corrected(rate=0.4, return_centers=False)
        samples_set = {
            "areas": np.int64(areas).tolist(),
            "centers": np.int64(centers).tolist() if centers is not None else None
        }
        data[i] = samples_set

    with open((root_path / ("areas_" + lines1_str + ".json")), 'w') as json_file:
        json.dump(data, json_file, indent=4)

    #print(data["areas"])
    #savemat((root_path / ("areas_" + lines1_str + ".mat")), data)


if __name__ == '__main__':
    main()
