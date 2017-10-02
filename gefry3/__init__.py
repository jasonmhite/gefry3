from gefry3.problem import *
from gefry3.classes import *

try:
    from gefry3 import plots
except ImportError as e:
    print("Exception raised importing plots package, skipped (you probably need to install matplotlib and/or seaborn)")
    print("\n\nException raised: {}".format(e)) 
