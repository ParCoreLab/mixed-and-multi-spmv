from numpy import arange
from utils import ATTRIB_DELTA, ATTRIB_ERROR, ATTRIB_SPEEDUP, ATTRIB_TIME, ATTRIB_GFLOPS
from utils import JACOBI, SPMV, ANAME_DOUBLES_CUSP, HATCHES, COLORS
from statistics import median

ROUND_DIGITS = 2


def tuplebar(ax, res, mats, type, attrib, barwidth, fontsize, legend, algtuples, aggregation="avg"):
  assert (aggregation in ['avg', 'median'])
  aggrsplits = {}

  # obtain values
  values_bar = {}
  rects = {}
  for algtup in algtuples:
    for alg in algtup:
      values_bar[alg] = []
      aggrsplits[alg] = 0
      val = []
      for m in mats:
        # sum values
        if attrib == ATTRIB_SPEEDUP:
          val.append(res[m][type][alg][ATTRIB_SPEEDUP])
        elif attrib == ATTRIB_TIME:
          val.append(res[m][type][ANAME_DOUBLES_CUSP][attrib] / res[m][type][alg][attrib])
        elif attrib == ATTRIB_GFLOPS:
          val.append(res[m][type][alg][attrib] / res[m][type][ANAME_DOUBLES_CUSP][attrib])
        elif attrib == ATTRIB_ERROR:
          val.append(res[m][type][alg][ATTRIB_ERROR])
        # sum
        aggrsplits[alg] += round((res[m][type][alg]["singles"] / res[m]["info"]["nnz"]) * 100, ROUND_DIGITS)
      # averaging
      aggrsplits[alg] /= len(mats)
      # round
      aggrsplits[alg] = round(aggrsplits[alg], ROUND_DIGITS)
      if aggregation == "avg":
        values_bar[alg].append(sum(val) / len(val))
      elif aggregation == "median":
        values_bar[alg].append(median(val))

  # draw bar rectangles
  x = arange(len(algtuples))
  x_i = 0
  xticks = []
  xlabels = []
  for algtup in algtuples:
    offset = -(len(algtup) - 1) * barwidth / 2
    for alg in algtup:
      rects[alg] = ax.bar(
          x[x_i] + offset,
          values_bar[alg],
          barwidth,
          label=alg,
          color=COLORS[alg],
          hatch=HATCHES[alg],
      )
      if (alg[:2] == 'DD'):
        ax.text(  # single percentage text
            x[x_i] + offset,
            values_bar[alg][0],
            " " + str(aggrsplits[alg]),
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=fontsize,
            color="black",
        )
      offset += barwidth
    alg = algtup[int(len(algtup) / 2)]
    if len(algtuples) % 2 == 0:
      rect = rects[alg].patches[0]
      xticks.append(rect.get_x())
    else:
      rect = rects[alg].patches[0]
      xticks.append(rect.get_x() - rect.get_width() / 2)
    xlabels.append(algtup[0][:-4])
    x_i += 1

  # axis labels and settings
  if attrib == ATTRIB_TIME or attrib == ATTRIB_SPEEDUP or attrib == ATTRIB_GFLOPS:
    ax.set_ylabel("speedup")
    ax.axhline(1.0, c="gray", ls="--", lw=0.8)
  elif attrib == ATTRIB_ERROR:
    ax.set_ylabel("error (log)")
    #ax.set_yscale('log')

  ax.set_xticks(xticks)
  ax.set_xticklabels(xlabels, rotation=55, ha="right", fontsize="medium")

  if legend:
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)

  ax.set_title(type + " " + attrib)
