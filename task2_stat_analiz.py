from pathlib import Path
import cv2
import numpy as np
from pyrockstats.distrebutions import lognorm, weibull, paretoexp
from shapely.geometry import Point, Polygon
from shapely.vectorized import contains

import matplotlib.pyplot as plt
from pyrocksegmentation.basic_segmentator import Segmentator
from pyrocksegmentation import Extractor
from pyrockstats.empirical import ecdf
from pyrockstats.empirical.empirical_distrebution import empirical_cdf_gen, ecdf_rvs
import pyrockstats.bootstrap.ks_statistics as ks_statistics

from tqdm import tqdm
import json

from pyrockshape import shape_load
from scipy.io import savemat, loadmat
from tools import get_config
from pyrockshape import shape_load


def main():
    lines_types = ["30", "120", "400", "800"]
    root_path = Path(get_config()["data_root"])
    
    lines_type = "30"
    lines_type_folder = root_path / ("FABDEM_" + lines_type)
    ensemble_file_name = "ensemble_areas_" + lines_type + ".json"
    
    pth = Path("D:/1.ToSaver/profileimages/ShapeBaikal/lineaments/FABDEM_30/areas")
    polies, bbox = shape_load(pth)
    
    with open(str(lines_type_folder / ensemble_file_name), 'r') as file:
        data = json.load(file)
    
    
    xs = np.logspace(3, 9, 500)
    bins = np.logspace(3, 9, 50)
    
    min_area = np.pi * ((75) ** 2)
    max_area = 10 ** 8
    
    areas_0 = np.array(data["0"]["areas"])
    areas_0 = areas_0[areas_0 > min_area]
    
    values_0, e_freq_0 = ecdf(areas_0, xmin=min_area, xmax=max_area)
    
    
    distribution = {
        "lognorm": lognorm(*lognorm.fit(areas_0, xmin=min_area, xmax=max_area)),
        "weibull": weibull(*weibull.fit(areas_0, xmin=min_area, xmax=max_area)),
        "paretoexp": paretoexp(*paretoexp.fit(areas_0, xmin=min_area, xmax=max_area))
    }
    
    cdf = {
        "empirical": empirical_cdf_gen(values_0, e_freq_0)(xs),
        "lognorm": distribution["lognorm"].cdf(xs, xmin=min_area, xmax=max_area),
        "weibull": distribution["weibull"].cdf(xs, xmin=min_area, xmax=max_area),
        "paretoexp": distribution["paretoexp"].cdf(xs, xmin=min_area, xmax=max_area)
    }
    
    pdf = {
        "empirical": np.histogram(areas_0, bins=bins, density=True)[0],
        "lognorm": distribution["lognorm"].pdf(bins, xmin=min_area, xmax=max_area),
        "weibull": distribution["weibull"].pdf(bins, xmin=min_area, xmax=max_area),
        "paretoexp": distribution["paretoexp"].pdf(bins, xmin=min_area, xmax=max_area)
    }
    
    ks = np.zeros(len(data))
    for i, key in tqdm(enumerate(data)):
        poly = Polygon(polies[1])
        areas = np.array(data[key]["areas"])
        centers = np.array(data[key]["centers"])
        mask = contains(poly, centers[:, 0], centers[:, 1])
        areas = areas[mask]
        ks[i] = ks_statistics.get_ks_estimation(areas, values_0, e_freq_0, xmin=min_area, xmax=max_area)
    
    confidance = ks_statistics.get_confidence_value(ks, 0.05)
    print(confidance)
    fig = plt.figure(figsize=(12, 6))
    axs = [fig.add_subplot(1, 2, 1),
           fig.add_subplot(1, 2, 2)]
    
    axs[0].fill_between(xs, cdf["empirical"] - confidance, cdf["empirical"] + confidance,
                        color="gray")
    axs[0].plot(xs, cdf["empirical"], color="black")
    axs[0].plot(xs, cdf["lognorm"], color="red")
    axs[0].plot(xs, cdf["weibull"], color="blue")
    axs[0].plot(xs, cdf["paretoexp"], color="green")
    axs[0].set_xscale('log')
    # axs[0].set_yscale('log')
    axs[0].grid("on")
    axs[0].set_ylim([0, 1])
    #
    axs[1].stairs(pdf["empirical"], bins, color="gray")
    axs[1].plot(bins, pdf["lognorm"], color="red")
    axs[1].plot(bins, pdf["weibull"], color="blue")
    axs[1].plot(bins, pdf["paretoexp"], color="green")
    axs[1].set_xscale('log')
    axs[1].set_xlim([min_area, max_area])
    plt.show()


if __name__ == '__main__':
    main()
