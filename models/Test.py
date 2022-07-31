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