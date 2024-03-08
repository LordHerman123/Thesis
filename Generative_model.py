import numpy as np

import scipy.stats as stats
import matplotlib.pyplot as plt

'''
The following function has this as input
sigma_0 = the std at 0 degrees of frame rotation
sigma_90 = the std at 90 degrees of frame rotation
tau = the change of verticle & horizontal in context to the frame (aka turning your head?)
oto = the combined uncertainty of the vestibular and prior information
theta = the current line rotation
'''
#check if it runs similar 
def analytical_solution(sigma_0,sigma_90,tau,oto,theta):
    mu_oto = 0
   # Calculate how the vertical and horizontal change depending on frame rotation
    sigma_ver = sigma_0 - (1 - np.cos(np.pi / 180 * abs(2 * theta))) * tau * (sigma_0 - sigma_90)
    sigma_hor = sigma_90 + (1 - np.cos(np.pi / 180 * abs(2 * theta))) * (1 - tau) * (sigma_0 - sigma_90)

    # Ensures that values are at most 1e-3
    sigma_ver = np.maximum(1e-3, sigma_ver)
    sigma_hor = np.maximum(1e-3, sigma_hor)

    # Analytical solution for total remaining variance after integrating the horizontal and vertical visibility signals
    sigma_vis = np.sqrt((sigma_ver**2 * sigma_hor**2) / (sigma_ver**2 + sigma_hor**2))

    # Analytical solution for combining horizontal and vertical signal
    weight1 = sigma_hor**2 / (sigma_ver**2 + sigma_hor**2)
    weight2 = sigma_ver**2 / (sigma_ver**2 + sigma_hor**2)

    # Represents the mean of the visual information
    mu_vis = weight1 * theta + weight2 * (theta - np.sign(theta) * 90)

    weight_vis = oto**2 / (sigma_vis**2 + oto**2)

    # Next, include the otolith information and prior signals
    weight_oto = sigma_vis**2 / (sigma_vis**2 + oto**2)

    mu_vertical = weight_vis * mu_vis + weight_oto * mu_oto
    sigma_vertical = np.sqrt((sigma_vis**2 * oto**2) / (sigma_vis**2 + oto**2))

    return mu_vertical, sigma_vertical

def sample_response(mu,sigma,degree):
    normal_dist = stats.norm(loc = mu, scale=sigma)
    return np.random.binomial(1,normal_dist.cdf(degree))


