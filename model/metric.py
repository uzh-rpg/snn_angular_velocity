import numpy as np


def medianRelativeError(pred: np.ndarray, gt: np.ndarray):
    # pred/gt: (samplesize, 3, time) tensor

    gt_norm = np.linalg.norm(gt, ord=2, axis=1).flatten()
    diff_norm = np.linalg.norm(gt - pred, ord=2, axis=1).flatten()

    # Do not take groundtruth into account that is less than 1 deg/s.
    one_deg_ps = 1/180*np.pi
    is_valid = (np.abs(gt_norm) >= one_deg_ps)
    diff_norm = diff_norm[is_valid]
    gt_norm = gt_norm[is_valid]
    rel_errors = np.divide(diff_norm, gt_norm)
    return np.median(rel_errors)

def rmse(pred: np.ndarray, gt: np.ndarray, deg: bool=True):
    # pred/gt: (samplesize, 3, time) tensor

    rmse = np.mean(np.sqrt(np.sum(np.square(pred - gt), axis=1).flatten()))
    if deg:
        rmse = rmse/np.pi*180
    return rmse
