import numpy as np
from .processor import Processor


class NormSub(Processor):
    def calibrate(self, est_dist):
        estimates = np.copy(est_dist)
        sum_estimte = np.sum(estimates)
        #print(sum_estimte)
        n = self.n
        while (np.fabs(sum(estimates) - n) > 1) or (estimates < 0).any():
            sum_neg = np.sum(estimates[estimates < 0])
            #print(sum_neg)
            estimates[estimates < 0] = 0
            total = sum(estimates)
            mask = estimates > 0
            diff = (n - total) / sum(mask)
            estimates[mask] += diff
        return estimates
