import numpy as np



def normalize_stdsample(m_matrrix):
    mu = np.mean(m_matrrix, axis=0)
    sigma = np.std(m_matrrix, axis=0,ddof=1)
    norm_x = (m_matrrix - mu)/sigma
    return (mu, sigma, norm_x)

def normalize_stdpopulation(m_matrrix):
    mu = np.mean(m_matrrix, axis=0)
    sigma = np.std(m_matrrix, axis=0)
    norm_x = (m_matrrix - mu)/sigma
    return (mu, sigma, norm_x)
