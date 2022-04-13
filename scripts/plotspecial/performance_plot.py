'''
BSD 3-Clause License

Copyright (c) 2020, Sam Relton
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
'''
On a profile, a point (x; y) means that the respective model is within x factor of the best result for a fraction y of the instances.
e.g. point (1.20, 0.60) on curve of model M means that M yields 20% more runtime than the smallest runtime achieved for 60% of the dataset.
The model closest to the top left corner is interpreted as the model with best performance.
'''
import numpy as np
from utils import *


def perfprof(ax,
             data,
             markerspecs=None,
             colorspecs=None,
             linewidth=1.6,
             thmax=None,
             tol=np.double(1e-8),
             legendnames=None,
             legendpos="lower right",
             fontsize=18,
             tickfontsize=14,
             legendfontsize=14,
             ppfix=False,
             ppfixmin=np.double(1e-18),
             xlim=1.30,
             ppfixmax=np.finfo(np.double).eps / 2):
  data = np.asarray(data).astype(np.double)

  if ppfix:
    data = np.array(data >= ppfixmax, dtype=np.int) * data + \
        np.array(data < ppfixmax, dtype=np.int) * \
        (ppfixmin + data*(ppfixmax - ppfixmin)/ppfixmax)

  minvals = np.min(data, axis=1)
  if thmax is None:
    thmax = np.max(np.max(data, axis=1) / minvals)
  m, n = data.shape  # m tests cases, n alternatives

  for alt in range(n):  # for each alternative
    col = data[:, alt] / minvals  # performance ratio
    col = col[~np.isnan(col)]  # remove nans

    if len(col) == 0:
      continue

    theta = np.unique(col)
    r = len(theta)
    myarray = np.repeat(col, r).reshape(len(col), r) <= \
        np.repeat(theta, len(col)).reshape((len(col), r), order='F')
    myarray = np.array(myarray, dtype=np.double)
    prob = np.sum(myarray, axis=0) / m

    # Get points to print staircase plot
    k = np.array(np.floor(np.arange(0, r, 0.5)), dtype=np.int)
    x = theta[k[1:]]
    y = prob[k[0:-1]]

    # check endpoints
    if x[0] >= 1 + tol:
      x = np.append([1, x[0]], x)
      y = np.append([0, 0], y)
    if x[-1] < thmax - tol:
      x = np.append(x, thmax)
      y = np.append(y, y[-1])

    ax.plot(x,
            y,
            color=colorspecs[alt],
            marker=markerspecs[alt],
            linewidth=linewidth,
            label=legendnames[alt],
            mfc='none',
            markersize=10,
            alpha=0.75)
    ax.tick_params(labelsize=tickfontsize)

  # create legend
  ax.legend(loc=legendpos, fontsize=legendfontsize)

  # set xlim
  #ax.set_xlim([1, thmax])
  ax.set_xlim([1, xlim])
  ax.set_ylim([0, 1.01])
  ax.axhline(1.0, c="gray", ls="--", lw=0.8)


def performance_profile(ax, res, mats, algs, type, attrib, mapaname, xlim, fontsize):
  # populate series
  tostack = []
  markerspecs = [MARKERS[a] for a in algs]
  colorspecs = [COLORS[a] for a in algs]
  labels = [MAPANAME_LEGEND[a] for a in algs] if mapaname else algs

  for m in mats:
    alg_vals = []
    for a in algs:
      alg_vals.append(res[m][type][a][attrib])
    tostack.append(alg_vals)

  data = np.vstack(tuple(tostack))
  perfprof(ax,
           data,
           markerspecs=markerspecs,
           colorspecs=colorspecs,
           legendnames=labels,
           linewidth=2,
           xlim=xlim,
           legendfontsize=fontsize,
           tickfontsize=fontsize)
