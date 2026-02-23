import numpy as np


class CEM:
  def __init__(
      self,
      func,
      d,
      maxits=500,
      N=100,
      Ne=10,
      argmin=True,
      v_min=None,
      v_max=None,
      init_scale=1,
      sampleMethod='Gaussian',
      init_mu=None,
      init_sigma=None,
  ):
    self.func = func
    self.d = int(d)
    self.maxits = int(maxits)
    self.N = int(N)
    self.Ne = int(Ne)
    self.reverse = not argmin
    self.v_min = None if v_min is None else np.asarray(v_min, dtype=float).reshape(self.d)
    self.v_max = None if v_max is None else np.asarray(v_max, dtype=float).reshape(self.d)
    self.init_coef = float(init_scale)
    self.init_mu = None if init_mu is None else np.asarray(init_mu, dtype=float).reshape(self.d)
    self.init_sigma = None if init_sigma is None else np.asarray(init_sigma, dtype=float).reshape(self.d)

    assert sampleMethod in ('Gaussian', 'Uniform')
    self.sampleMethod = sampleMethod

    if self.v_min is not None and self.v_max is not None:
      self.v_min, self.v_max = np.minimum(self.v_min, self.v_max), np.maximum(self.v_min, self.v_max)

  def eval(self, instr=None):
    """Evaluate and return the best solution."""
    if self.sampleMethod == 'Gaussian':
      return self.evalGaussian(instr)
    if self.sampleMethod == 'Uniform':
      return self.evalUniform(instr)
    raise ValueError("Unsupported sample method")

  def evalUniform(self, instr=None):
    t, _min, _max = self.__initUniformParams()

    while t < self.maxits:
      x = self.__uniformSampleData(_min, _max)
      s = self.__sortSample(self.__functionReward(instr, x))
      x = np.array([s[i][0] for i in range(len(s))])
      _min, _max = self.__updateUniformParams(x)
      t += 1

    return (_min + _max) / 2.0

  def evalGaussian(self, instr=None):
    t, mu, sigma = self.__initGaussianParams()

    while t < self.maxits:
      x = self.__gaussianSampleData(mu, sigma)
      s = self.__sortSample(self.__functionReward(instr, x))
      x = np.array([s[i][0] for i in range(len(s))])
      mu, sigma = self.__updateGaussianParams(x)
      t += 1

    return mu

  def __initGaussianParams(self):
    t = 0
    mu = np.zeros(self.d) if self.init_mu is None else self.init_mu.copy()
    if self.init_sigma is None:
      sigma = np.ones(self.d) * self.init_coef
    else:
      sigma = np.maximum(self.init_sigma.copy(), 1e-12)
    return t, mu, sigma

  def __updateGaussianParams(self, x):
    mu = x[0:self.Ne, :].mean(axis=0)
    sigma = np.maximum(x[0:self.Ne, :].std(axis=0), 1e-12)
    return mu, sigma

  def __gaussianSampleData(self, mu, sigma):
    sample_matrix = np.zeros((self.N, self.d))
    for j in range(self.d):
      sample_matrix[:, j] = np.random.normal(loc=mu[j], scale=sigma[j], size=(self.N,))
      if self.v_min is not None and self.v_max is not None:
        sample_matrix[:, j] = np.clip(sample_matrix[:, j], self.v_min[j], self.v_max[j])
    return sample_matrix

  def __initUniformParams(self):
    t = 0
    if self.v_min is not None:
      _min = self.v_min.copy()
    else:
      _min = -np.ones(self.d)
    if self.v_max is not None:
      _max = self.v_max.copy()
    else:
      _max = np.ones(self.d)
    return t, _min, _max

  def __updateUniformParams(self, x):
    _min = np.amin(x[0:self.Ne, :], axis=0)
    _max = np.amax(x[0:self.Ne, :], axis=0)
    return _min, _max

  def __uniformSampleData(self, _min, _max):
    sample_matrix = np.zeros((self.N, self.d))
    for j in range(self.d):
      sample_matrix[:, j] = np.random.uniform(low=_min[j], high=_max[j], size=(self.N,))
    return sample_matrix

  def __functionReward(self, instr, x):
    # Preferred: f(X) -> costs
    try:
      vals = self.func(x)
      vals = np.asarray(vals, dtype=float).reshape(-1)
      return zip(x, vals)
    except TypeError:
      pass

    # Backward-compatible: f(instr_batch, X) -> costs
    if instr is None:
      instr = np.zeros(self.d, dtype=float)
    bi = np.reshape(instr, [1, -1])
    bi = np.repeat(bi, self.N, axis=0)
    vals = np.asarray(self.func(bi, x), dtype=float).reshape(-1)
    return zip(x, vals)

  def __sortSample(self, s):
    return sorted(s, key=lambda x: x[1], reverse=self.reverse)


def func(a1, a2):
  c = a1 - a2
  return [_c[0] * _c[0] + _c[1] * _c[1] for _c in c]


if __name__ == '__main__':
  cem = CEM(func, 2, sampleMethod='Uniform', v_min=[-5., -5.], v_max=[5., 5.])
  t = np.array([1, 2])
  v = cem.eval(t)
  print(v, func(t.reshape([-1, 2]), v.reshape([-1, 2])))
