import numpy as np


def normalize_meanbystd(v_vector):
    mean = np.mean(v_vector)
    sd = np.std(v_vector)
    return (v_vector - mean)/sd
