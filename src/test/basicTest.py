# Testing LCModel function for BasicModel (Placebo)

import sys
import numpy as np
import unittest

sys.path.append('../main')

from ModelDSBase import ModelDSBaseWrapper
from ModelDSBase import basicModelParamsToMap

class basicTest(unittest.TestCase):

	def testBasic(self):
		X = np.random.random((1000,10))
		y = np.random.random((1000,1))
		percentiles = [25,50,75,100]

		lcm=ModelDSBaseWrapper('baseTest', X, y, percentiles, parameters=basicModelParamsToMap(1,2,3,4))

		lcm.train()

		lc = lcm.getLearningCurves()

		self.assertTrue( (lc[0,:].max() == 1) & (lc[0,:].mean() == 1) )
		self.assertTrue( (lc[1,:].max() == 0) & (lc[1,:].mean() == 0) )

if __name__ == '__main__':
    unittest.main()
