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
        areas = get_border_aras(poly, data_base)
        
        ensemble = {}
        for number in data_ensemble:
            ensemble[number] = get_border_aras(poly, data_ensemble[number])
        
        ks = Estimator.ks_stats_estimate(areas, ensemble)
        values, freqs = ecdf(ks)
        plt.plot(values, freqs)
        plt.show()
        
        exit()
        fig = plt.figure(figsize=(12, 4))
        axs = [fig.add_subplot(1, 1, 1)]
        #
        obj = Estimator(areas, lognorm)
        values, freqs = obj.get_empirical()
        distribution = obj.get_distribution()
        ks_norm = obj.get_ks_norm()
        print("lognorm: ",ks_norm)
        #
        axs[0].plot(np.log10(values), distribution.cdf(values))
        axs[0].plot(np.log10(values), kse.cdf(values))
        #
        obj = Estimator(areas, paretoexp)
        values, freqs = obj.get_empirical()
        distribution = obj.get_distribution()
        axs[0].plot(np.log10(values), distribution.cdf(values))
        ks_norm = obj.get_ks_norm()
        print("paretoexp: ",ks_norm)
        #
        obj = Estimator(areas, weibull)
        values, freqs = obj.get_empirical()
        distribution = obj.get_distribution()
        axs[0].plot(np.log10(values), distribution.cdf(values))
        ks_norm = obj.get_ks_norm()
        print("weibull: ",ks_norm)
        #
        plt.show()
        exit()
    
    return
    
    # exit()
    # xmin = np.pi * ((300) ** 2)
    # xmax = 10 ** 10
    
    
    
    xs = np.logspace(np.log10(xmin), np.log10(xmax), 500)
    bins = np.logspace(np.log10(xmin), np.log10(xmax), 50)
    
    
    def get_border_ar(key):
        poly = Polygon(polies[region_number])
        areas = np.array(data[key]["areas"])
        centers = np.array(data[key]["centers"])
        mask = contains(poly, centers[:, 0], centers[:, 1])
        return areas[mask]
        
    areas_0 = get_border_ar("1")
    mask = areas_0 > xmin
    areas_0 = areas_0[mask]
    
    values_0, e_freq_0 = ecdf(areas_0, xmin=xmin, xmax=xmax)
    
    theta = {
        "lognorm": lognorm.fit(areas_0, xmin=xmin, xmax=xmax),
        "weibull": weibull.fit(areas_0, xmin=xmin, xmax=xmax),
        "paretoexp": paretoexp.fit(areas_0, xmin=xmin, xmax=xmax)
    }
    
    distribution = {
        "lognorm": lognorm(*lognorm.fit(areas_0, xmin=xmin, xmax=xmax)),
        "weibull": weibull(*weibull.fit(areas_0, xmin=xmin, xmax=xmax)),
        "paretoexp": paretoexp(*paretoexp.fit(areas_0, xmin=xmin, xmax=xmax))
    }
    
    cdf = {
        "empirical": empirical_cdf_gen(values_0, e_freq_0)(xs),
        "lognorm": distribution["lognorm"].cdf(xs, xmin=xmin, xmax=xmax),
        "weibull": distribution["weibull"].cdf(xs, xmin=xmin, xmax=xmax),
        "paretoexp": distribution["paretoexp"].cdf(xs, xmin=xmin, xmax=xmax)
    }
    
    pdf = {
        "empirical": np.histogram(areas_0, bins=bins, density=True)[0],
        "lognorm": distribution["lognorm"].pdf(bins, xmin=xmin, xmax=xmax),
        "weibull": distribution["weibull"].pdf(bins, xmin=xmin, xmax=xmax),
        "paretoexp": distribution["paretoexp"].pdf(bins, xmin=xmin, xmax=xmax)
    }
    
    ks = np.zeros(len(data))
    for i, key in tqdm(enumerate(data)):
        poly = Polygon(polies[region_number])
        areas = np.array(data[key]["areas"])
        centers = np.array(data[key]["centers"])
        mask = contains(poly, centers[:, 0], centers[:, 1])
        areas = areas[mask]
        ks[i] = ks_statistics.get_ks_estimation(areas, values_0, e_freq_0, xmin=xmin, xmax=xmax)
    
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
