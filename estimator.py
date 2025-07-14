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
from pyrockstats.bootstrap.ks_statistics import get_confidence_value


class Estimator:
    def __init__(self, x, model=lognorm, xmin=None, xmax=None):
        self.x = x
        self.xmin = xmin
        self.xmax = xmax
        self.model = model
        self.distribution = None
        self.theta = None
        self.values = None
        self.freqs = None
        self.ks_norm = None
        self.p_value = None
    
    def get_theta(self):
        if self.theta is not None:
            return self.theta
        self.theta = self.model.fit(self.x, xmin=self.xmin, xmax=self.xmax)
        return self.theta
    
    def get_distribution(self):
        if self.distribution is not None:
            return self.distribution
        theta = self.get_theta()
        self.distribution = self.model(*theta)
        return self.distribution
    
    def get_empirical(self):
        if self.values is not None and self.freqs is not None:
            return self.values, self.freqs
        self.values, self.freqs = ecdf(self.x, xmin=self.xmin, xmax=self.xmax)
        return self.values, self.freqs
    
    def get_ks_norm(self):
        if self.ks_norm is not None:
            return self.ks_norm
        values, freqs = self.get_empirical()
        distribution = self.get_distribution()
        self.ks_norm = np.max(np.abs(distribution.cdf(values) - freqs))
        return self.ks_norm
    
    def get_p_value(self, ks):
        if self.p_value is not None:
            return self.p_value
        values, freqs = ecdf(ks)
        d = self.get_ks_norm()
        idx = np.searchsorted(values, d, side='right')
        if idx == 0:
            p_value = 1.0  # Если D меньше всех значений в self.ks
        else:
            p_value = 1.0 - freqs[idx - 1]
        self.p_value = p_value
        return self.p_value

    
    @classmethod
    def ks_stats_estimate(cls, x, ensamble, xmin=None, xmax=None):
        values, freqs = ecdf(x, xmin=xmin, xmax=xmax)
        cdf = empirical_cdf_gen(values, freqs)
        ks = np.empty((len(ensamble),))
        for number in ensamble:
            values, freqs = ecdf(ensamble[number])
            ks[int(number)] = np.max(np.abs(cdf(values) - freqs))
        return ks
    
    @classmethod
    def get_confidance(cls, ks, alpha):
        return get_confidence_value(ks, alpha)
    
    
