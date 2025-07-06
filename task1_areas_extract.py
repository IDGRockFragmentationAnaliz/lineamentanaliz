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

from tools import get_config


def main():
    lines_types = ["30", "120", "400", "800"]
    root_path = Path(get_config()["data_root"])
    assembl_len = 500
    
    lines_type = "120"
    lines_type_folder = root_path / ("FABDEM_" + lines_type)
    
    shape_path = lines_type_folder / ("lineaments_" + lines_type)
    
    with open(str(root_path / "config.json"), 'r') as file:
        data = json.load(file)
        buffer = data[lines_type]["buffer"]*2  # radius -> diameter
    
    base_file_name = "areas_" + lines_type + ".json"
    file_path = (lines_type_folder / base_file_name)
    if file_path.is_file() is False:
        data = get_base_data(shape_path, buffer)
        with open(str(file_path), 'w+') as json_file:
            json.dump(data, json_file, indent=4)
    else:
        print("Файл базовой сегментации существует, переход к следующему шагу.")
    
    data = get_synthetic_data(shape_path, buffer, assembl_len=assembl_len, rate=0.4)
    ensemble_file_name = "ensemble_areas_" + lines_type + ".json"
    file_path = lines_type_folder / ensemble_file_name
    with open(str(file_path), 'w+') as json_file:
        json.dump(data, json_file, indent=4)
    

def get_synthetic_data(path, buffer, assembl_len=10, rate=0.4):
    """
    Код для извлечения замкнутых областей и линеаментов.
    rate: отвечает за величину оценки ситматической ошибки
    assembl_len: отвечает за величину за величину выходного ансамбля
    """
    line_map = LineamentsMap(path, buffer)
    data = {}
    for i in tqdm(range(assembl_len)):
        areas, centers = line_map.get_areas_corrected(
            rate=rate,
            return_centers=True,
            exclude_mincircle=True
        )
        samples_set = {
            "areas": np.int64(areas).tolist(),
            "centers": np.int64(centers).tolist() if centers is not None else None
        }
        data[i] = samples_set
    return data


def get_base_data(path, buffer):
    """
    Код для извлечения замкнутых областей и линеаментов.
    rate: отвечает за величину оценки ситматической ошибки
    assembl_len: отвечает за величину за величину выходного ансамбля
    """
    line_map = LineamentsMap(path, buffer)
    areas, centers = line_map.get_areas_corrected(
        return_centers=True,
        exclude_mincircle=True
    )
    data = {
        "areas": np.int64(areas).tolist(),
        "centers": np.int64(centers).tolist() if centers is not None else None
    }
    return data


def get_set():
    areas, centers = line_map.get_areas_corrected(
        rate=rate,
        return_centers=True,
        exclude_mincircle=True
    )
    samples_set = {
        "areas": np.int64(areas).tolist(),
        "centers": np.int64(centers).tolist() if centers is not None else None
    }
    return samples_set


if __name__ == '__main__':
    main()
