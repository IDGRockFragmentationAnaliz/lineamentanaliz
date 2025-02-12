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


def main():
	lines1_str = "300_500"
	root_path = Path("D:/1.ToSaver/profileimages/ShapeBaikal/lineaments")
	shape_path = root_path / ("FABDEM_" + lines1_str)

	with open(str(root_path / "config.json"), 'r') as file:
		data = json.load(file)
		buffer = data[lines1_str]

	line_map = LineamentsMap(shape_path, buffer)
	areas, centers = line_map.get_areas()
	image = line_map.image

	#area_min = np.log10(np.min(areas))
	#area_max = np.log10(np.max(areas))

	fig = plt.figure(figsize=(14, 9))
	axs = [
		fig.add_subplot(1, 1, 1)
	]
	axs[0].imshow(image)
	axs[0].plot(centers[:, 0], centers[:, 1], ".")
	plt.show()


class LineamentsMap:
	def __init__(self, lines_folder: Path, buffer=1):
		lines, bbox = shape_load(lines_folder)
		shift = np.array((bbox[0], bbox[1]))
		dx = bbox[2] - bbox[0]
		dy = bbox[3] - bbox[1]
		shape_x = 5000
		scale = shape_x / dx
		shape_y = int(dy * scale)

		shape = (shape_y, shape_x)
		buffer = int(buffer * scale)

		lines = [np.int32((line - shift) * scale) for line in lines]

		self.shift = shift
		self.buffer = buffer
		self.scale = scale
		self.image = self.make_image(lines, shape)

	def make_image(self, lines, shape):
		image = np.zeros(shape, np.uint8)
		image = cv2.polylines(image, lines, False, (255,), self.buffer)
		return image

	def get_areas(self):
		s, c = Extractor(Segmentator(self.image).run()).extruct_centers()
		return s, c

	def get_areas_corrected(self):
		s, c = self.get_areas()
		s = s/(self.scale**2)
		c = c/self.scale + self.shift
		return s, c


if __name__ == '__main__':
	main()



