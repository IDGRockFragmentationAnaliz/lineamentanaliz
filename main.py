from pathlib import Path
import cv2
import numpy as np
from pyrockstats.distrebutions import lognorm, weibull, paretoexp

from rocknetmanager.tools.shape_load import shape_lines_save
import matplotlib.pyplot as plt
from pyrocksegmentation.basic_segmentator import Segmentator
from pyrocksegmentation import Extractor
from pyrockstats.empirical import ecdf

root_path = Path("D:/1.ToSaver/profileimages/ShapeBaikal")
shape_path = (
	root_path / "FABDEM_600_1000_Фильтрация" / "Линеаменты_FABDEM_600_1000.shp"
)

lines, bbox = shape_lines_save(shape_path)

exit()
shape = (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1)
new_shape = (1000, 1000)

for i, line in enumerate(lines):
	lines[i] = (line * [1, -1] - bbox[0:2])/(shape[0], shape[1])*new_shape
	lines[i] = lines[i].astype(np.int32)

image = np.zeros(new_shape, np.uint8)
image = cv2.polylines(image, lines, False, 255, 3)

marcked_areas = Segmentator(image).run()
areas = Extractor(marcked_areas).extruct()

dist = {}
x, cdf = ecdf(areas)
theta = lognorm.fit(areas)
dist["lognorm"] = lognorm(*theta)
theta = paretoexp.fit(areas)
dist["paretoexp"] = paretoexp(*theta)
theta = weibull.fit(areas)
dist["weibull"] = weibull(*theta)

fig = plt.figure(figsize=(14, 9))
axs = [
	fig.add_subplot(2, 2, 1),
	fig.add_subplot(2, 2, 2),
	fig.add_subplot(2, 2, 3)
]
axs[0].imshow(image)
axs[0].set_title("edges")
axs[1].imshow(marcked_areas)
axs[1].set_title("segments")
axs[2].plot(x, cdf, color="black", label="ecdf")
axs[2].plot(x, dist["lognorm"].cdf(x), color="blue", label="lognorm")
axs[2].plot(x, dist["paretoexp"].cdf(x), color="red", label="paretoexp")
axs[2].plot(x, dist["weibull"].cdf(x), color="green", label="weibull")
axs[2].set_xscale('log')
axs[2].legend(loc='lower right')
axs[2].set_xlim([10, 10**3])
axs[2].set_title("stats")
fig.savefig("600-1000.png")
plt.show()


