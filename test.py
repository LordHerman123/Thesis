
import warnings

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as tt

from pymc3 import Model, Normal, Slice, sample, Uniform, Binomial
from pymc3.distributions import Interpolated
from scipy import stats


plt.style.use("seaborn-darkgrid")
print(f"Running on PyMC3 v{pm.__version__}")