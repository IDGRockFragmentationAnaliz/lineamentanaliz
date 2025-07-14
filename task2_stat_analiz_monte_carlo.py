from pathlib import Path
import cv2
import numpy as np
from pyrockstats.distrebutions import lognorm, weibull, paretoexp
from shapely.geometry import Point, Polygon
from shapely.vectorized import contains

import matplotlib as mpl
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
from estimator import Estimator


def main():
    lines_type = "120"
    root_path = Path("D:/1.ToSaver/profileimages/ShapeBaikal/lineaments")
    root_path = root_path / ("FABDEM_" + lines_type)
    json_base_segmentation = root_path / ("areas_" + lines_type + ".json")
    json_path_ensemble_segmentation = root_path / ("ensemble_areas_" + lines_type + ".json")
    shape_areas_path = root_path / "areas"
    
    with open(str(json_base_segmentation), 'r') as file:
        data_base = json.load(file)
    
    with open(str(json_path_ensemble_segmentation), 'r') as file:
        data_ensemble = json.load(file)
    
    polies, bbox = shape_load(shape_areas_path)
    
    for i, poly in enumerate(polies):
        print("area number:",i)
        areas = get_border_aras(poly, data_base)
        ensemble = {}
        for number in data_ensemble:
            ensemble[number] = get_border_aras(poly, data_ensemble[number])
        
        ks = Estimator.ks_stats_estimate(areas, ensemble)
        estimator = {
            "lognorm": Estimator(areas, lognorm),
            "paretoexp": Estimator(areas, paretoexp),
            "weibull": Estimator(areas, weibull)
        }
        
        confidance = Estimator.get_confidance(ks, 0.05)
        print("confidance", confidance)
        p_lognorm = estimator["lognorm"].get_p_value(ks)
        p_paretoexp = estimator["paretoexp"].get_p_value(ks)
        p_weibull = estimator["weibull"].get_p_value(ks)
        
        print("p_lognorm:", p_lognorm,
              "d:", estimator["lognorm"].get_ks_norm()
              )
        print("p_paretoexp:", p_paretoexp,
              "d:", estimator["paretoexp"].get_ks_norm()
              )
        print("p_weibull:", p_weibull,
              "d:", estimator["weibull"].get_ks_norm()
              )
        continue
    
    return
    
    
    print(theta)
    exit()
    
    font_path = Path(".") / "assets" / "timesnewromanpsmt.ttf"
    custom_font = mpl.font_manager.FontProperties(fname=font_path, size=16)
    
    fig = plt.figure(figsize=(6, 6))
    axs = [fig.add_subplot(1, 1, 1)]
    
    axs[0].fill_between(xs, cdf["empirical"] - confidance, cdf["empirical"] + confidance,
                        color="gray")
    axs[0].plot(xs, cdf["empirical"], color="black", label="1")
    axs[0].plot(xs, cdf["lognorm"], color="red", label="2")
    axs[0].plot(xs, cdf["weibull"], color="blue", label="3")
    axs[0].plot(xs, cdf["paretoexp"], color="green", label="4")
    axs[0].set_xscale('log')
    # axs[0].set_yscale('log')
    axs[0].legend(loc='lower right', fontsize=16, prop=custom_font)
    axs[0].grid("on")
    axs[0].set_ylim([0, 1])
    axs[0].set_xlim([xmin, xmax])
    axs[0].set_xlabel(r's, мкм$^\mathregular{2}$', fontproperties=custom_font, size=16)
    for label in axs[0].get_xticklabels():
        label.set_fontproperties(custom_font)
        label.set_size(16)
    for label in axs[0].get_yticklabels():
        label.set_fontproperties(custom_font)
        label.set_size(16)
    axs[0].set_title("Область: " + str(region_number),
                     fontproperties=custom_font, size=16)
    
    # axs[1].stairs(pdf["empirical"], bins, color="gray")
    # axs[1].plot(bins, pdf["lognorm"], color="red")
    # axs[1].plot(bins, pdf["weibull"], color="blue")
    # axs[1].plot(bins, pdf["paretoexp"], color="green")
    # axs[1].set_xscale('log')
    # axs[1].set_xlim([min_area, max_area])
    save_pic = Path("D:/1.ToSaver/profileimages/ShapeBaikal/lineaments/pictures/FABDEM_" + lines_type)
    file_path = save_pic / ("region_" + str(region_number))
    fig.savefig(str(file_path.with_suffix(".png")), dpi=300, bbox_inches='tight')
    #plt.show()


def get_border_aras(poly, data):
    poly = Polygon(poly)
    areas = np.array(data["areas"])
    centers = np.array(data["centers"])
    mask = contains(poly, centers[:, 0], centers[:, 1])
    return areas[mask]


if __name__ == '__main__':
    main()
