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
	root_path = Path("D:/1.ToSaver/profileimages/ShapeBaikal/lineaments")
	list_names = ["30", "120", "400", "800"]

	data = {}
	for name in list_names:
		bins, density = get_data((root_path / ("FABDEM_hist_" + name + ".mat")))
		bins, density = bin_corr(bins, density)
		data[name] = {
			"bins": list(bins),
			"density": list(density),
			"units": "m"
		}


	with open((root_path / "lineaments_hists.json"), 'w') as json_file:
		json.dump(data, json_file, indent=4)

	fig = plt.figure(figsize=(7, 7))
	axs = [fig.add_subplot(1, 1, 1)]
	axs[0].stairs(data["30"]["density"], data["30"]["bins"])
	axs[0].stairs(data["120"]["density"], data["120"]["bins"])
	axs[0].stairs(data["400"]["density"], data["400"]["bins"])
	axs[0].stairs(data["800"]["density"], data["800"]["bins"])
	axs[0].set_xscale('log')
	axs[0].set_yscale('log')
	plt.show()


def bin_corr(bins, density):
	arg_d_min = np.argmin(density)
	if density[arg_d_min] == 0:
		bins = bins[0:arg_d_min+1]
		density = density[0:arg_d_min]
	return bins, density


def get_data(path: Path):
	data = loadmat(str(path), squeeze_me=True)
	bins = data["bins"]
	density = data["density"]
	return bins, density


if __name__ == '__main__':
	main()
