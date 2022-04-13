from numpy import arange
from utils import *

ROUND_DIGITS = 3  # for single percentage


def aggregatebar(ax, res, mats, algs, type, attrib, barwidth, fontsize, legend, usehatches, mapaname):
  # obtain values
  rects = {}
  values_bar = {}
  split_percentages = {}
  for alg in algs:
    values_bar[alg] = 0
    split_percentages[alg] = 0
    for m in mats:
      # bar y values
      if attrib == ATTRIB_ERROR or attrib == ATTRIB_DELTA or attrib == ATTRIB_SPEEDUP:
        values_bar[alg] += res[m][type][alg][attrib]
      elif attrib == ATTRIB_TIME:
        values_bar[alg] += res[m][type][ANAME_DOUBLES_CUSP][attrib] / res[m][type][alg][attrib]
      elif attrib == ATTRIB_GFLOPS:
        values_bar[alg] += res[m][type][alg][attrib] / res[m][type][ANAME_DOUBLES_CUSP][attrib]
      split_percentages[alg] += res[m][type][alg][ATTRIB_SPLIT_PERCENTAGE]
    split_percentages[alg] /= len(mats)
    values_bar[alg] /= len(mats)

  # draw bar rectangles
  x = arange(len(algs))
  alg_i = 0

  for alg in algs:
    # generate rectangle
    rects[alg] = ax.bar(
        x[alg_i],
        values_bar[alg],
        barwidth,
        label=alg,
        color=COLORS[alg],
        hatch=HATCHES[alg] if usehatches else None,
    )
    ax.text(  # single percentage text
        x[alg_i],
        values_bar[alg],
        " {0:.3f}".format(values_bar[alg]),
        ha="center",
        va="bottom",
        rotation=90,
        fontsize=fontsize,
        color="black",
    )
    alg_i += 1

  percent_text = ""
  for alg in algs:
    percent_text += "{0}: {1:.1f}%\n".format(alg, 100.0 * split_percentages[alg])
  ax.text(0.05, 0.92, percent_text, ha='left', va='top', transform=ax.transAxes)
  # axis labels and settings
  if attrib == ATTRIB_ERROR or attrib == ATTRIB_DELTA:
    ax.set_ylabel("log(" + attrib + ")")
    ax.set_yscale("log")
  elif attrib == ATTRIB_TIME or attrib == ATTRIB_GFLOPS or attrib == ATTRIB_SPEEDUP:
    ax.set_ylabel("speedup")
    ax.axhline(1.0, c="gray", ls="--", lw=0.8)
    ax.set_ylim([0.9, 1.2])

  # ax.set_xlabel('Matrix')
  ax.set_xticks(x)
  ax.set_xticklabels(algs if not mapaname else [MAPANAME_XAXIS[alg] for alg in algs],
                     rotation=0,
                     ha="center",
                     fontsize="medium")

  if legend:
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='medium')
