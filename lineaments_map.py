import math
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
from scipy.io import savemat
from time import time


class LineamentsMap:

    LINE_COLOR = (255,)
    DEFAULT_SHAPE_X = 3600*4

    def __init__(self, lines_folder: Path, buffer=1):
        lines, bbox = shape_load(lines_folder)
        (self.scale,
         self.shape,
         self.pixel_buffer,
         self.shift) = self._get_pixel_map_config(bbox, buffer)
        self.min_area = math.pi * ((self.pixel_buffer / 2) ** 2)
        self.lines = [self._world_to_pixels(line) for line in lines]

    @classmethod
    def _get_pixel_map_config(cls, bbox, buffer):
        """
            takes:
                bbox   Границы карты в координатах
                buffer Размер буфера в усл.ед.

            returns:
                scale (float): Масштаб (пиксели на единицу координат).
                shape (tuple): Размеры изображения (height, width).
                buffer (int): Размер буфера в пикселях.
                shift (np.array): Сдвиг координат.
        """
        shift = np.array((bbox[0], bbox[1]))
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        shape_x = cls.DEFAULT_SHAPE_X
        scale = shape_x / width
        shape_y = int(height * scale)
        shape = (shape_y, shape_x)
        buffer = int(buffer * scale)
        print(buffer)
        return scale, shape, buffer, shift

    def _world_to_pixels(self, coord):
        return np.int32((coord - self.shift) * self.scale)

    def _pixels_to_world(self, coord):
        return coord / self.scale + self.shift

    def exclude_lines(self, rate=0):
        """Filter lines using random sampling."""
        if not 0 <= rate < 1:
            raise ValueError("Rate must be in [0, 1) range")

        is_not_excluded = (np.random.rand(len(self.lines)) > rate).tolist()
        lines = [line for cond, line in zip(is_not_excluded, self.lines) if cond]
        return lines

    def make_image_template(self):
        return np.zeros(self.shape, np.uint8)

    def make_image(self, lines):
        image = self.make_image_template()
        image = cv2.polylines(image, lines, False, self.LINE_COLOR, self.pixel_buffer)
        return image

    def get_areas(self, rate=0, return_centers=True, exclude_mincircle=True):
        if rate == 0:
            lines = self.lines
        else:
            lines = self.exclude_lines(rate)
        centers = None

        image = self.make_image(lines)
        segmentator = Segmentator(image)
        segments = segmentator.run()
        #img = segmentator.get_segment_image()
        #cv2.imwrite("./data/image_3.png", img)
        
        # fig = plt.figure(figsize=(6, 6))
        # axs = [fig.add_subplot(1, 1, 1)]
        # axs[0].imshow(img)
        # plt.show()
        
        #fig.savefig("./data/" + "all_stat" + ".png", dpi=300, bbox_inches='tight')
        
        if return_centers is True:
            areas, centers = Extractor(segments).extruct_centers()
        else:
            areas = Extractor(segments).extruct_areas()
        
        if exclude_mincircle is True:
            mask = areas > self.min_area
            areas = areas[mask]
            centers = centers[mask] if centers is not None else None

        return areas, centers

    def get_areas_corrected(self, rate=0, return_centers=True, exclude_mincircle=True):
        s, centers = self.get_areas(rate, return_centers, exclude_mincircle)
        s = s / (self.scale ** 2)
        centers = centers / self.scale + self.shift if centers is not None else None
        return s, centers
