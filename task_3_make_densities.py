from pathlib import Path
import cv2
import numpy as np
from pyrockstats.distrebutions import lognorm, weibull, paretoexp

import matplotlib.pyplot as plt
from pyrocksegmentation.basic_segmentator import Segmentator
from pyrocksegmentation import Extractor
from pyrockstats.empirical import ecdf

from shapely.geometry import Point, Polygon
from shapely.vectorized import contains

import json

from pyrockshape import shape_load
from scipy.io import savemat, loadmat


def main():
	lines1_str = "120"
	root_path = Path("D:/1.ToSaver/profileimages/ShapeBaikal/lineaments")
	root_path = root_path / ("FABDEM_" + lines1_str)
	json_path = root_path / ("areas_" + lines1_str + ".json")
	
	with open(str(json_path), 'r') as file:
		data = json.load(file)
		
	
	pth = Path("D:/1.ToSaver/profileimages/ShapeBaikal/lineaments/FABDEM_120/areas")
	polies, bbox = shape_load(pth)
	density_data = {}
	for i, poly in enumerate(polies[0:]):
		areas = get_border_areas(poly, data)
		density_data[i] = get_density_data(areas)
	
	with open("./data/lineaments_densities.json", 'w+') as json_file:
		json.dump(density_data, json_file, indent=4)


def get_density_data(s):
	# вычисление размера пикселя
	s = np.delete(s, np.argmax(s))
	xmin = np.min(s)
	xmax = np.max(s)

	# Начальное число бинов
	n_bins = 10
	min_bins = 7  # минимальное допустимое число бинов

	# Логарифмические бины
	hist = None
	bins = None
	while True:
		bins = np.logspace(np.log10(xmin), np.log10(xmax), n_bins)
		hist, bins = np.histogram(s, bins=bins)
		if np.all(hist > 0):
			break
		elif n_bins > min_bins:
			n_bins -= 1
		else:
			break

	# маска для ненулевых значений гистограммы
	mask = hist > 0

	# Вычисляем плотность
	bin_widths = np.diff(bins)
	rho = np.log10(hist[mask]) - np.log10(bin_widths[mask] * np.sum(s))# - 4*np.log10(pix2m)

	# Средние точки бинов
	s_rho = (bins[:-1] + bins[1:]) / 2
	s_rho = np.log10(s_rho[mask])# + 2*np.log10(pix2m)

	# Преобразуем в список для JSON-сериализации
	data = {
		"s": s_rho.tolist(),
		"rho": rho.tolist(),
		"unit": "log m2"
	}
	return data


def get_border_areas(poly, data):
	poly = Polygon(poly)
	areas = np.array(data["areas"])
	centers = np.array(data["centers"])
	mask = contains(poly, centers[:, 0], centers[:, 1])
	return areas[mask]


if __name__ == '__main__':
	main()
	
