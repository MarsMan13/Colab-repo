import numpy as np

def im2col(input_data, FH, FW, stride=1, pad=0):
  N, C, H, W = input_data.shape
  OH = (H + 2 * pad - FH) // stride + 1
  OW = (W + 2 * pad - FW) // stride + 1
  img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
  col = np.zeros((N, OH, OW, C, FH, FW))

  for h in range(OH):
    ii = h * stride
    for w in range(OW):
      jj = w * stride
      col[:, h, w, :, :, :] = img[:, :, ii:ii+FH, jj:jj+FW]
  col = col.reshape(N*OH*OW, -1)
  return col

def col2im(col, input_shape, FH, FW, stride=1, pad=0):
  N, C, H, W = input_shape
  OH = (H + 2 * pad - FH) // stride + 1
  OW = (W + 2 * pad - FW) // stride + 1
  col = col.reshape(N, OH, OW, C, FH, FW)
  img = np.zeros((N, C, H+2*pad, W+2*pad))
  for h in range(OH):
    ii = h * stride
    for w in range(OW):
      jj = w * stride
      img[:,:,ii:ii+FH, jj:jj+FW] = col[:, h, w, :, :, :]
  return img[:, :, pad:pad+H, pad:pad+W]

def gaussianInit(input_size, output_size, weight_init_std=0.01):
  return np.random.randn(input_size, output_size) * weight_init_std

def xavierInit(input_size, output_size, weight_init_std=0.01):
  return np.random.randn(input_size, output_size) / np.sqrt(input_size)

def heInit(input_size, output_size, weight_init_std=0.01):
  return np.random.randn(input_size, output_size) * np.sqrt( 2 / input_size)

##### SIDE LAYER #########################

class Sigmoid:
  def forward(self, x):
    self.y = 1. / (1 + np.exp(-x))
    return self.y
  
  def backward(self, dout):
    return dout * self.y * (1. - self.y)

class Relu:
  def __init__(self):
    self.mask = None

  def forward(self, X):
    self.mask = (X <= 0)
    self.out = X.copy()
    self.out[self.mask] = 0
    return self.out
  
  def backward(self, dout):
    dout[self.mask] = 0
    dx = dout
    return dx

def softmax(x):
  C = np.max(x)
  exps = np.exp(x - C)
  if x.ndim == 1:
    sum_exps = np.sum(exps)
    return exps / sum_exps
  sum_exps = np.sum(exps, axis=1)
  return (exps.T / sum_exps).T

def cross_entropy_error(y, t):
  if y.ndim == 1:
    y = y.reshape(1, y.size)
    t = t.reshape(1, t.size)
  batch_size = t.shape[0]
  return -np.sum( t * np.log(y + 1e-4) ) / batch_size

class BinaryCrossEntropy:
  def __init__(self):
    self.loss = None
    self.x = None
    self.t = None

  def forward(self, x, t):
    self.x = x
    self.t = t
    self.loss = -(t * np.log(x) + (1 - t) * np.log(1 - x))
    self.loss = np.sum(self.loss)
    return self.loss
  
  def backward(self, dout=1):
    return - (self.t/self.x - (1 - self.t)/(1- self.x)) * dout

class SoftmaxWithLoss:
  def __init__(self):
    self.loss = None
    self.y = None
    self.t = None
  
  def forward(self, x, t):
    self.t = t
    self.y = softmax(x)
    self.loss = cross_entropy_error(self.y, self.t)
    return self.loss
  
  def backward(self, dout=1):
    batch_size = self.t.shape[0]
    dx = (self.y - self.t) / batch_size
    return dx

##### CORE LAYER #########################

class Affine:
  def __init__(self, W, b):
    self.W = W
    self.b = b
    self.x = None
    self.original_x_shape = None
    self.dW = None
    self.db = None

  def forward(self, x):
    self.original_x_shape = x.shape
    self.x = x.reshape(x.shape[0], -1)
    return np.dot(self.x, self.W) + self.b
  
  def backward(self, dout):
    self.db = np.sum(dout, axis=0)
    self.dw = np.dot(self.x.T, dout)
    dx = np.dot(dout, self.W.T)
    return dx.reshape(self.original_x_shape)

class Convolution:
  def __init__(self, W, b, stride=1, pad=0):
    self.W = W
    self.b = b
    self.stride = stride
    self.pad = pad

  def forward(self, x):
    self.x = x
    N, C, H, W = self.x.shape
    FN, C, FH, FW = self.W.shape
    out_h = (H + 2*self.pad - FH) // self.stride + 1
    out_w = (W + 2*self.pad - FW) // self.stride + 1
    self.col = im2col(x, FH, FW, self.stride, self.pad)
    self.col_W = self.W.reshape(FN, -1).T
    out = np.dot(self.col, self.col_W) + self.b

    out = out.reshape(N, out_h, out_w, FN).transpose(0,3,1,2)
    return out
  
  def backward(self, dout):
    N, C, H, W = self.x.shape
    FN, C, FH, FW = self.W.shape
    out_h = (H + 2*self.pad - FH) // self.stride + 1
    out_w = (W + 2*self.pad - FW) // self.stride + 1
    # step1
    dout = dout.transpose(0,2,3,1).reshape(-1, FN)
    # step2 for db
    self.db = np.sum(dout, axis=0)
    # step3 for self.col * self.col_W
    dcol = np.dot(dout, self.col_W.T)
    dcol_W = np.dot(self.col.T, dout)
    # step4 for dx, dw
    dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
    self.dw = dcol_W.T.reshape(FN, C, FH, FW)
    return dx

class Pooling:
  def __init__(self, pool_h, pool_w, stride=1, pad=0):
    self.pool_h = pool_h
    self.pool_w = pool_w
    self.stride = stride
    self.pad = pad
  
  def forward(self, x):
    N, C, H, W = x.shape
    out_h = (H - self.pool_h) // self.stride + 1
    out_w = (W - self.pool_w) // self.stride + 1
    col = im2col(x, self.pool_h, self.pool_w, stride=self.stride, pad=self.pad)
    col = col.reshape(-1, self.pool_h * self.pool_w)
    self.arg_max = np.argmax(col, axis=1)
    out = np.max(col, axis=1)
    out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)
    self.x = x
    return out

  def backward(self, dout):
    N, C, H, W = self.x.shape
    out_h = (H - self.pool_h) // self.stride + 1
    out_w = (W - self.pool_w) // self.stride + 1
    dout = dout.transpose(0,2,3,1).reshape(N, out_h, out_w, C)

    dmax = np.zeros((dout.size, self.pool_h * self.pool_w))
    dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
    dmax = dmax.reshape(dout.shape + (self.pool_h * self.pool_w,))
    dcol = dmax.reshape(dmax.shape[0], dmax.shape[1], dmax.shape[2], -1)
    dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
    return dx

class Substractor:
  def forward(self, x1, x2):
    return x1 - x2
  def backward(self, dout):
    # print("dout:",dout)
    return dout, -dout

class Flattener:
  def forward(self, x):
    self.batch_size = x.shape[0]
    self.origin_shape = x.shape
    self.out = x.reshape(self.batch_size, -1)
    return self.out
  
  def backward(self, dout):
    dout = dout.reshape(self.origin_shape)
    return dout
