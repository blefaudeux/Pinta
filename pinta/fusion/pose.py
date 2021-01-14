#!/usr/bin/python3
from pykalman import UnscentedKalmanFilter
import numpy as np

# See https://pykalman.github.io/

class Pose:
    def __init__(self):
        initialState = np.zeros((6,1))
        initialCov = np.diag(np.ones((6,1))[:,0])
        self.filter = UnscentedKalmanFilter(initial_state_mean=initialState, initial_state_covariance=initialCov, n_dim_obs=6)

    def updateAcc(self, acc):
        pass

    def updateGyr(self, gyr):
        pass

    def getEstimate(self):
        return np.zeros((6,1))
