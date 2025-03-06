from pathlib import Path
import cv2
import numpy as np
from pyrockstats.distrebutions import lognorm, weibull, paretoexp

import matplotlib.pyplot as plt
from pyrocksegmentation.basic_segmentator import Segmentator
from pyrocksegmentation import Extractor
from pyrockstats.empirical import ecdf
from pyrockstats.empirical.empirical_distrebution import empirical_cdf_gen, ecdf_rvs

from tqdm import tqdm
import json

from pyrockshape import shape_load
from scipy.io import savemat, loadmat


def main():
    lines1_str = "120"
    root_path = Path("D:/1.ToSaver/profileimages/ShapeBaikal/lineaments")
    mat_path = root_path / ("areas_" + lines1_str + ".mat")

    with open(str(root_path / ("areas_" + lines1_str + ".json")), 'r') as file:
        data = json.load(file)
    
    xs = np.logspace(3, 9, 500)
    bins = np.logspace(3, 9, 50)
    
    cdf = {"ecdf": {}, "lognorm": {}, "weibull": {}, "paretoexp": {}}
    pdf = {"epdf": {}, "lognorm": {}, "weibull": {}, "paretoexp": {}}
    min_area_0 = np.pi * ((75) ** 2)
    max_area_0 = 10 ** 8
    for i, key in tqdm(enumerate(data)):
        areas = np.array(data[key]["areas"])
        #areas = areas[areas >= min_area_0]
        areas = areas[areas <= max_area_0]
        
        min_area = np.min(areas)
        max_area = np.max(areas)
        
        
        areas = areas[areas < max_area]
        
        values, e_freq = ecdf(areas)
        areas = ecdf_rvs(values, e_freq, len(areas))
        values, e_freq = ecdf(areas)
        e_cdf = empirical_cdf_gen(values, e_freq)
        
        cdf["ecdf"][i] = e_cdf(xs)
        pdf["epdf"][i], bins = np.histogram(areas, bins, density=True)
        #
        theta = lognorm.fit(areas, xmin=min_area, xmax=max_area)
        theta = (theta[0], theta[1])
        distribution = lognorm(*theta)
        cdf["lognorm"][i] = distribution.cdf(xs, xmin=min_area, xmax=max_area)
        pdf["lognorm"][i] = distribution.pdf(xs, xmin=min_area, xmax=max_area)
        #
        distribution = weibull(*weibull.fit(areas, xmin=min_area, xmax=max_area))
        cdf["weibull"][i] = distribution.cdf(xs, xmin=min_area, xmax=max_area)
        pdf["weibull"][i] = distribution.pdf(xs, xmin=min_area, xmax=max_area)
        #
        distribution = paretoexp(*paretoexp.fit(areas, xmin=min_area, xmax=max_area))
        cdf["paretoexp"][i] = distribution.cdf(xs, xmin=min_area, xmax=max_area)
        pdf["paretoexp"][i] = distribution.pdf(xs, xmin=min_area, xmax=max_area)
        #
        
        if i == 10:
            break
    
    #buffer = data[lines1_str]["buffer"]
    #centers = data["centers"][3:]

    # theta = {
    # 	"lognorm": lognorm.fit(areas, xmin=10**5, xmax=10**7),
    # 	"weibull": weibull.fit(areas, xmin=10**5, xmax=10**7),
    # 	"paretoexp": paretoexp.fit(areas, xmin=10**8),
    # }
    # print(theta)

    # theta["weibull"] = (theta["weibull"][0], theta["weibull"][1])

    # cdf["lognorm"] = lognorm(*theta["lognorm"]).cdf(x)
    # cdf["weibull"] = lognorm(*theta["weibull"]).cdf(x)
    # cdf["paretoexp"] = lognorm(*theta["paretoexp"]).cdf(x)

    fig = plt.figure(figsize=(12, 6))
    axs = [fig.add_subplot(1, 2, 1),
           fig.add_subplot(1, 2, 2)]
    for i in range(len(cdf["ecdf"])):
        axs[0].plot(xs, cdf["ecdf"][i], color="grey")
    for i in range(len(cdf["ecdf"])):
        axs[0].plot(xs, cdf["lognorm"][i], color="red")
        #axs[0].plot(xs, cdf["weibull"][i], color="blue")
        #axs[0].plot(xs, cdf["paretoexp"][i], color="green")
    axs[0].set_xscale('log')
    # axs[0].set_yscale('log')
    axs[0].grid("on")
    axs[0].set_ylim([0, 1])
    #
    axs[1].stairs(pdf["epdf"][0], bins, color="grey")
    axs[1].plot(xs, pdf["lognorm"][0], color="red")
    axs[1].plot(xs, pdf["weibull"][0], color="blue")
    axs[1].plot(xs, pdf["paretoexp"][0], color="green")
    axs[1].plot([min_area_0, min_area_0],[0, np.max(pdf["epdf"][0])])
    
    axs[1].set_xscale('log')
    axs[1].set_ylim([0, np.max(pdf["epdf"][0])])
    plt.show()


if __name__ == '__main__':
    main()
