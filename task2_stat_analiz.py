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
from pyrockshape import shape_load



def main():
    lines_types = ["30", "120", "400", "800"]
    root_path = Path(get_config()["data_root"])
    region_number = 10
    lines_type = "120"
    lines_type_folder = root_path / ("FABDEM_" + lines_type)
    ensemble_file_name = "ensemble_areas_" + lines_type + ".json"
    
    pth = Path("D:/1.ToSaver/profileimages/ShapeBaikal/lineaments/FABDEM_30/areas")
    polies, bbox = shape_load(pth)
    
    with open(str(lines_type_folder / ensemble_file_name), 'r') as file:
        data = json.load(file)
    
    for name in data:
        print(name)
    
    
    min_area = np.pi * ((300) ** 2)
    max_area = 10 ** 10
    
    xs = np.logspace(np.log10(min_area), np.log10(max_area), 500)
    bins = np.logspace(np.log10(min_area), np.log10(max_area), 50)
    
    
    def get_border_ar(key):
        poly = Polygon(polies[region_number])
        areas = np.array(data[key]["areas"])
        centers = np.array(data[key]["centers"])
        mask = contains(poly, centers[:, 0], centers[:, 1])
        return areas[mask]
        
    areas_0 = get_border_ar("1")
    mask = areas_0 > min_area
    areas_0 = areas_0[mask]
    
    values_0, e_freq_0 = ecdf(areas_0, xmin=min_area, xmax=max_area)
    
    theta = {
        "lognorm": lognorm.fit(areas_0, xmin=min_area, xmax=max_area),
        "weibull": weibull.fit(areas_0, xmin=min_area, xmax=max_area),
        "paretoexp": paretoexp.fit(areas_0, xmin=min_area, xmax=max_area)
    }
    
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
        poly = Polygon(polies[region_number])
        areas = np.array(data[key]["areas"])
        centers = np.array(data[key]["centers"])
        mask = contains(poly, centers[:, 0], centers[:, 1])
        areas = areas[mask]
        ks[i] = ks_statistics.get_ks_estimation(areas, values_0, e_freq_0, xmin=min_area, xmax=max_area)
    
    confidance = ks_statistics.get_confidence_value(ks, 0.05)
    ks_norm = {
        "lognorm": np.max(np.abs(cdf["empirical"] - cdf["lognorm"])),
        "weibull": np.max(np.abs(cdf["empirical"] - cdf["weibull"])),
        "paretoexp": np.max(np.abs(cdf["empirical"] - cdf["paretoexp"]))
    }
    
    def get_p_value(name):
        ks_values, ks_freq = ecdf(ks)
        p_idx = np.where(ks_values > ks_norm[name])[0]
        if len(p_idx) == 0:
            p_idx = len(ks_values) - 1
        else:
            p_idx = p_idx[0]
        return 1 - ks_freq[p_idx]
    
    p_value = {
        "lognorm": get_p_value("lognorm"),
        "weibull": get_p_value("weibull"),
        "paretoexp": get_p_value("paretoexp")
    }
    
    #print("p_value", p_value)
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
    axs[0].set_xlim([min_area, max_area])
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


if __name__ == '__main__':
    main()
