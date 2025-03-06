from pathlib import Path
import cv2
import numpy as np
from pyrockstats.distrebutions import lognorm, weibull, paretoexp

import matplotlib.pyplot as plt
from pyrocksegmentation.basic_segmentator import Segmentator
from pyrocksegmentation import Extractor
from pyrockstats.empirical import ecdf

import json

from pyrockshape import shape_load
from scipy.io import savemat, loadmat


def main():
	lines1_str = "30"
	root_path = Path("D:/1.ToSaver/profileimages/ShapeBaikal/lineaments")
	mat_path = root_path / ("areas_" + lines1_str + ".mat")
	data = loadmat(str(mat_path), squeeze_me=True)
	areas = data["areas"][3:]
	centers = data["centers"][3:]
	
	sum_area = np.sum(areas)

	x_min = np.log10(np.min(areas))
	x_max = np.log10(np.max(areas))

	bins = np.logspace(x_min, x_max, 30)
	hist, bins = np.histogram(areas, bins)
	density = hist/sum_area

	log_bins = np.log(bins)
	log_center = np.exp((log_bins[:-1] + log_bins[1:]) / 2)

	fig = plt.figure(figsize=(7, 7))
	axs = [fig.add_subplot(1, 1, 1)]
	axs[0].stairs(density, bins)
	axs[0].plot(log_center, density)
	axs[0].set_xscale('log')
	axs[0].set_yscale('log')
	plt.show()

	data = {
		"bins": bins,
		"density": density
	}

	savemat((root_path / ("FABDEM_hist_" + lines1_str + ".mat")), data)


if __name__ == '__main__':
	main()
	
