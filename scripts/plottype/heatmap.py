from numpy import empty, arange
from utils import ATTRIB_SPLIT_PERCENTAGE, ATTRIB_ITERATION

ROUND_DIGITS = 3


def heatmap(ax, res, mats, algs, type, attrib, legend):
  num_mats = len(mats)
  num_algs = len(algs)

  # Fill values
  vals = empty((num_mats, num_algs))
  m_i = 0
  for m in mats:
    a_i = 0
    for alg in algs:
      vals[m_i, a_i] = res[m][type][alg][attrib]
      a_i += 1
    m_i += 1

  ax.imshow(vals, interpolation="nearest", aspect="auto", cmap="cividis")
  ax.set_yticks(arange(num_mats))
  ax.set_xticks(arange(num_algs))
  ax.set_yticklabels(mats, fontsize="x-small")
  ax.set_xticklabels(algs, fontsize="x-small")

  # Loop over data dimensions and create text annotations.
  for i in range(num_mats):
    for j in range(num_algs):
      if attrib == ATTRIB_SPLIT_PERCENTAGE:
        ax.text(j, i, str(round(vals[i, j], 2)), ha="center", va="center", color="white", backgroundcolor="black")
      elif attrib == ATTRIB_ITERATION:
        ax.text(j, i, str(vals[i, j]), ha="center", va="center", color="white", backgroundcolor="black")
      else:
        ax.text(j, i, "%.2E" % vals[i, j], ha="center", va="center", color="white", backgroundcolor="black")

  if legend:
    pass  # no legend for this guy anyways

  ax.set_title(type + " " + attrib)