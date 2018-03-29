from gefry3.problem import *
from gefry3.classes import *

import warnings

try:
    from gefry3 import plots
except ImportError as e:
    warnings.warn("Exception raised importing plots package, skipped (you probably need to install matplotlib and/or seaborn)", ImportWarning)
