'''
H(X) = -(Pi * log2Pi)
假定一个系统的输出有2种情况：【0.9， 0.1】,另外一个是【0.7，0.3】，还有一个是【0.5，0.5】，哪个熵最大
'''
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
Pa = np.array([0.9, 0.1])
Pb = np.array([0.7, 0.3])
Pc = np.array([0.5, 0.5])
EntropyA = - (np.log2(Pa) * Pa).sum()
EntropyB = - (np.log2(Pb) * Pb).sum()
EntropyC = - (np.log2(Pc) * Pc).sum()

logging.info(EntropyA)
logging.info(EntropyB)
logging.info(EntropyC)  # entropy max
