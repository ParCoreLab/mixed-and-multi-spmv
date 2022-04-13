from numpy import arange
from utils import *


def split_comparison(ax, res, mats, algs):
  c = 'gray'
  x = []
  y = []
  for m in mats:
    x.append(res[m][SPMV][ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT]['percentage'] * 100.0)
    y.append(res[m][SPMV][ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT]['percentage'] * 100.0)
  ax.scatter(x, y, color=c, alpha=0.7, edgecolors='darkgray')
  ax.set_xlim([0, 100])
  ax.set_ylim([0, 100])
  ax.axhline(10.0, c="gray", ls="--", lw=0.8)
  ax.axhline(90.0, c="gray", ls="--", lw=0.8)

  x = arange(100)
  y = arange(100)
  ax.plot(x, y, linewidth=0.5, linestyle='--', color='gray')
