import sframe                            # see below for install instruction
import matplotlib.pyplot as plt          # plotting
import numpy as np                       # dense matrices
from scipy.sparse import csr_matrix      # sparse matrices
%matplotlib inline

wiki = sframe.SFrame('..//w2-a1//people_wiki.gl/')
wiki = wiki.add_row_number()
