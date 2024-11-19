from pathlib import Path
import cv2
import numpy as np
from pyrockstats.distrebutions import lognorm, weibull, paretoexp

from rocknetmanager.tools.shape_load import shape_lines_load
import matplotlib.pyplot as plt
from pyrocksegmentation.basic_segmentator import Segmentator
from pyrocksegmentation import Extractor
from pyrockstats.empirical import ecdf

root_path = Path("D:/1.ToSaver/profileimages/ShapeBaikal")
shape_path = (
	root_path / "9_200ГГК" / "all.shp")

lines, bbox = shape_lines_load(shape_path)

shape = (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1)
new_shape = (1000, 1000)

p2m2 = shape[0]*shape[1]/(new_shape[0]*new_shape[1])
p2km2 = p2m2/(1000*1000)
line_width = 3

def get_dists(areas):
	dist = {}
	x, cdf = ecdf(areas)
	theta = lognorm.fit(areas)
	print(theta)
	dist["lognorm"] = lognorm(*theta)
	theta = paretoexp.fit(areas)
	dist["paretoexp"] = paretoexp(*theta)
	theta = weibull.fit(areas)
	dist["weibull"] = weibull(*theta)
	dist["ecdf"] = {"x": x, "cdf": cdf}
	return dist

lines_1 = []
lines_2 = []
lines_3 = []
lines_4 = []
lines_5 = []
lines_6 = []

for i, line in enumerate(lines):
	lines[i] = (line * [1, -1] - bbox[0:2]) / (shape[0], shape[1]) * new_shape
	lines[i] = lines[i].astype(np.int32)
	pos_c = np.mean(lines[i], axis=0).astype(np.int32)
	if pos_c[0] + 500 < pos_c[1]:
		lines_1.append(lines[i])
	if (
		(pos_c[0] + 500 > pos_c[1]) and
		(pos_c[0] + 200 < pos_c[1]) and
		(pos_c[0] < 300)
	):
		lines_2.append(lines[i])
	if (
		(pos_c[0] + 500 > pos_c[1]) and
		(pos_c[0] + 200 < pos_c[1]) and
		(pos_c[0] > 300)
	):
		lines_3.append(lines[i])
	if (
		(pos_c[0] + 200 > pos_c[1]) and
		(pos_c[0] - 100 < pos_c[1]) and
		(pos_c[0] > 600)
	):
		lines_4.append(lines[i])
	if (
		(pos_c[0] + 200 > pos_c[1]) and
		(pos_c[0] - 100 < pos_c[1]) and
		(pos_c[0] < 600) and
		(pos_c[0] > 300)
	):
		lines_5.append(lines[i])
	if (
		(pos_c[0] + 200 > pos_c[1]) and
		(pos_c[0] - 100 < pos_c[1]) and
		(pos_c[0] < 300) and
		(pos_c[0] > 000)
	):
		lines_6.append(lines[i])


zero_image = np.zeros(new_shape, np.uint8)
image = zero_image.copy()
image = cv2.polylines(image, lines, False, 255, line_width)


def get_areas(lines):
	zero_image = np.zeros(new_shape, np.uint8)
	image = cv2.polylines(zero_image.copy(), lines, False, 255, line_width)
	return Extractor(Segmentator(image).run()).extruct()


crops_lines = {
	"red": lines_1,
	"green": lines_2,
	"blue": lines_3,
	"aqua": lines_4,
	"magenta": lines_5,
	"yellow": lines_6}


areas = {}
dists = {}
for key, value in crops_lines.items():
	areas = get_areas(value)
	print("len",len(areas))
	dists[key] = get_dists(areas)

image = cv2.merge((image, image, image))
image = cv2.polylines(image, lines_1, False, (255, 0, 0), line_width)
image = cv2.polylines(image, lines_2, False, (0, 255, 0), line_width)
image = cv2.polylines(image, lines_3, False, (0, 0, 255), line_width)
image = cv2.polylines(image, lines_4, False, (0, 255, 255), line_width)
image = cv2.polylines(image, lines_5, False, (255, 0, 255), line_width)
image = cv2.polylines(image, lines_6, False, (255, 255, 0), line_width)


def mini_plot(ax, dist, name="stats"):
	x = dist["ecdf"]["x"]
	cdf = dist["ecdf"]["cdf"]
	ax.plot(x*p2m2/(1000*1000), cdf, color="black", label="ecdf")
	ax.plot(x*p2m2/(1000*1000), dist["lognorm"].cdf(x), color="blue", label="lognorm")
	ax.plot(x*p2m2/(1000*1000), dist["paretoexp"].cdf(x), color="red", label="paretoexp")
	ax.plot(x*p2m2/(1000*1000), dist["weibull"].cdf(x), color="green", label="weibull")
	ax.set_xscale('log')
	ax.legend(loc='lower right')
	ax.set_xlim([1, 10 ** 3])
	ax.set_title(name)


fig = plt.figure(figsize=(14, 9))
axs = [
	fig.add_subplot(2, 2, 1),
	fig.add_subplot(2, 2, 2),
	fig.add_subplot(2, 2, 3),
	fig.add_subplot(2, 2, 4),
]
axs[0].imshow(image, origin='lower')
mini_plot(axs[1], dists["red"], "red")
mini_plot(axs[2], dists["green"], "green")
mini_plot(axs[3], dists["blue"], "blue")
# mini_plot(axs[1], dists["aqua"], "aqua")
# mini_plot(axs[2], dists["magenta"], "magenta")
# mini_plot(axs[3], dists["yellow"], "yellow")
fig.savefig("1")
plt.show()
