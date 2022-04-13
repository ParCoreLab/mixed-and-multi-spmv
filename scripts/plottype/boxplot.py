from math import nan, isnan
from utils import ATTRIB_DELTA, ATTRIB_ERROR, ATTRIB_SPEEDUP, ATTRIB_TIME, ATTRIB_GFLOPS, ATTRIB_GAMMA
from utils import JACOBI, SPMV, ANAME_DOUBLES_CUSP, HATCHES, COLORS
import matplotlib.pyplot as plt

ROUND_DIGITS = 3  # for single percentage


def boxplot(ax, res, mats, algs, type, attrib, barwidth, fontsize, legend, outliers: bool):
  # init
  vals = {}
  for alg in algs:
    vals[alg] = []
    for m in mats:
      if attrib == ATTRIB_ERROR or attrib == ATTRIB_DELTA or attrib == ATTRIB_SPEEDUP or attrib == ATTRIB_GAMMA:
        if isnan(res[m][type][alg][attrib]):
          print(m)
        vals[alg].append(res[m][type][alg][attrib])
      elif attrib == ATTRIB_TIME:
        vals[alg].append(res[m][type][ANAME_DOUBLES_CUSP][attrib] / res[m][type][alg][attrib])
      elif attrib == ATTRIB_GFLOPS:
        vals[alg].append(res[m][type][alg][attrib] / res[m][type][ANAME_DOUBLES_CUSP][attrib])

  # prepare plotting values
  x = []
  lbl = []
  colors = []
  for a in vals:
    lbl.append(a)
    x.append(vals[a])
    colors.append(COLORS[a])

  # boxplot
  boxplt = ax.boxplot(x, labels=lbl, showmeans=True, showfliers=outliers, meanline=False)
  i = 0
  for b in boxplt['means']:
    plt.setp(b, markerfacecolor=colors[i], marker='D', markeredgecolor='black')
    i += 1
  i = 0
  for b in boxplt['medians']:
    plt.setp(b, color=colors[i], linewidth=2.5)
    i += 1

  # set labels
  if attrib == ATTRIB_ERROR or attrib == ATTRIB_DELTA or attrib == ATTRIB_GAMMA:
    ax.set_ylabel("log(" + attrib + ")")
  elif attrib == ATTRIB_TIME or attrib == ATTRIB_SPEEDUP:
    ax.set_ylabel("speedup")
  elif attrib == ATTRIB_GFLOPS:
    ax.set_ylabel("gflops improvement")
  ax.set_xticklabels(lbl, rotation=55, ha="right", fontsize="x-small")

  if attrib == ATTRIB_TIME or attrib == ATTRIB_GFLOPS or attrib == ATTRIB_SPEEDUP:
    ax.axhline(1.0, c="gray", ls="--", lw=0.8)
    #ax.set_ylim([0, 2.5]) # TODO: remove

  if attrib == ATTRIB_ERROR or attrib == ATTRIB_DELTA:
    ax.set_yscale("log")
