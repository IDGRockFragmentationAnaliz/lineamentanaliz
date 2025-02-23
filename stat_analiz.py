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
	
	xs = np.logspace(5, 9, 500)
	cdf = {"ecdf": {}, "lognorm": {}, "weibull": {}, "paretoexp": {}}
	for i, key in tqdm(enumerate(data)):
		areas = np.array(data[key]["areas"])
		min_area = np.min(areas)
		max_area = np.max(areas)
		
		max_area = 10**9
		areas = areas[areas < max_area]
		
		values, e_freq = ecdf(areas)
		areas = ecdf_rvs(values, e_freq, len(areas))
		values, e_freq = ecdf(areas)
		e_cdf = empirical_cdf_gen(values, e_freq)
		
		cdf["ecdf"][i] = e_cdf(xs)
		#
		distribution = lognorm(*lognorm.fit(areas, xmin=min_area, xmax=max_area))
		cdf["lognorm"][i] = distribution.cdf(xs, xmin=min_area, xmax=max_area)
		#
		distribution = weibull(*weibull.fit(areas, xmin=min_area, xmax=max_area))
		cdf["weibull"][i] = distribution.cdf(xs, xmin=min_area, xmax=max_area)
		#
		distribution = paretoexp(*paretoexp.fit(areas, xmin=min_area, xmax=max_area))
		cdf["paretoexp"][i] = distribution.cdf(xs, xmin=min_area, xmax=max_area)
		
		# if i == 10:
		# 	break
	

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


	fig = plt.figure(figsize=(7, 7))
	axs = [fig.add_subplot(1, 1, 1)]
	for i in range(len(cdf["ecdf"])):
		axs[0].plot(xs, cdf["ecdf"][i], color="grey")
		axs[0].plot(xs, cdf["lognorm"][i], color="red")
		axs[0].plot(xs, cdf["weibull"][i], color="blue")
		axs[0].plot(xs, cdf["paretoexp"][i], color="green")
	# axs[0].plot(x, cdf["lognorm"], color="red")
	# axs[0].plot(x, cdf["weibull"], color="green")
	# axs[0].plot(x, cdf["paretoexp"], color="blue")
	axs[0].set_xscale('log')
	# axs[0].set_yscale('log')
	axs[0].grid("on")
	axs[0].set_ylim([0, 1])
	plt.show()


if __name__ == '__main__':
	main()
