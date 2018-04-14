lqg = LQG(5)
lqg.define('ABCQRVWX', np.matrix(np.eye(2)))
data = lqg.sample(10)
