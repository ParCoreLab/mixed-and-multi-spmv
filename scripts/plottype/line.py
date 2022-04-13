from utils import *

ROUND_DIGITS = 3


def line(ax, res, mats, algs, type, xattrib, yattrib, legend, usemarkers, mapaname):
  for alg in algs:
    x = []
    y = []
    for m in mats:
      # x attributes
      if xattrib == ATTRIB_SPLIT_PERCENTAGE:
        x.append(round(res[m][type][alg][ATTRIB_SPLIT_PERCENTAGE], ROUND_DIGITS))
      elif xattrib == ATTRIB_ERROR or xattrib == ATTRIB_SPEEDUP or xattrib == ATTRIB_GAMMA:
        x.append(res[m][type][alg][xattrib])
      elif xattrib == ATTRIB_TIME:
        x.append(res[m][type][ANAME_DOUBLES_CUSP][ATTRIB_TIME] / res[m][type][alg][ATTRIB_TIME])
      elif xattrib == ATTRIB_GFLOPS:
        x.append(res[m][type][alg][ATTRIB_GFLOPS] / res[m][type][ANAME_DOUBLES_CUSP][ATTRIB_GFLOPS])
      elif xattrib == ATTRIB_INSTANCE or xattrib == ATTRIB_MATRIX:
        x.append(len(x) + 1)
      elif xattrib == ATTRIB_NNZ or xattrib == ATTRIB_AVG_NNZ:
        x.append(res[m]['info'][xattrib])

      # y attributes
      if yattrib == ATTRIB_SPLIT_PERCENTAGE:
        y.append(round(res[m][type][alg][ATTRIB_SPLIT_PERCENTAGE], ROUND_DIGITS))
      elif yattrib == ATTRIB_ERROR or yattrib == ATTRIB_SPEEDUP or yattrib == ATTRIB_GAMMA:
        y.append(res[m][type][alg][yattrib])
      elif yattrib == ATTRIB_TIME:
        y.append(res[m][type][ANAME_DOUBLES_CUSP][ATTRIB_TIME] / res[m][type][alg][ATTRIB_TIME])
      elif yattrib == ATTRIB_GFLOPS:
        y.append(res[m][type][alg][ATTRIB_GFLOPS] / res[m][type][ANAME_DOUBLES_CUSP][ATTRIB_GFLOPS])
      elif yattrib == ATTRIB_INSTANCE:
        y.append(len(y) + 1)
      elif yattrib == ATTRIB_NNZ or yattrib == ATTRIB_AVG_NNZ:
        y.append(res[m]['info'][yattrib])

    ax.plot(x,
            y,
            c=COLORS[alg],
            label=MAPANAME_LEGEND[alg].replace('\n', ' ') if mapaname else alg,
            marker=MARKERS[alg] if usemarkers else ('.' if type == SPMV else None),
            alpha=0.7)

  # legend
  if legend:
    ax.legend(
        ncol=2,
        fontsize="medium",
        bbox_to_anchor=(0.0, 1.12, 1.0, 0.112),
        loc="lower left",
        mode="expand",
        borderaxespad=0.0,
    )

  # x attribute modifiers
  if xattrib == ATTRIB_ERROR or xattrib == ATTRIB_DELTA or xattrib == ATTRIB_GAMMA:
    ax.set_xscale("log")
  elif xattrib == ATTRIB_GFLOPS or xattrib == ATTRIB_TIME or xattrib == ATTRIB_SPEEDUP:
    ax.axvline(1.0, c="gray", ls="--", lw=0.8)
  elif xattrib == ATTRIB_MATRIX:
    ax.set_xticks(x)
    ax.set_xticklabels(mats, rotation=30, ha="right", fontsize="x-small")
  elif xattrib == ATTRIB_INSTANCE:
    ax.set_xticks([])
    ax.set_xticklabels([])

  # y attribute modifiers
  if yattrib == ATTRIB_ERROR or yattrib == ATTRIB_DELTA or yattrib == ATTRIB_GAMMA:
    ax.set_yscale("log")
  elif yattrib == ATTRIB_GFLOPS or yattrib == ATTRIB_TIME or yattrib == ATTRIB_SPEEDUP:
    ax.axhline(1.0, c="gray", ls="--", lw=0.8)

  # labels and titles
  ax.set(xlabel=xattrib, ylabel=yattrib, title="{0} vs {1}".format(xattrib, yattrib))