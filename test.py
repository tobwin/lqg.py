import numpy as np
import lqg

system = lqg.LQG(5)  # initialize LQG system with time horizon T = 5
system.define('ABCQRVWX', np.matrix(np.eye(2)))  # define matrices
data = system.sample(10)  # simulate 10 system runs
