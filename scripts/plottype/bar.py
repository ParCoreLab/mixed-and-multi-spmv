from numpy import arange
from utils import ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE, ATTRIB_DELTA, ATTRIB_ERROR, ATTRIB_SPEEDUP, ATTRIB_TIME, ATTRIB_GFLOPS
from utils import JACOBI, SPMV, ANAME_DOUBLES_CUSP, CARDIAC, COLORS, MAPANAME_LEGEND

ROUND_DIGITS = 3  # for single percentage


def bar(ax, res, mats, algs, type, attrib, barwidth, fontsize, legend, mapaname):
  # obtain values
  values_bar = {}
  values_singles = {}
  values_iterations = {}
  rt_bar = {}
  rects = {}
  for alg in algs:
    values_bar[alg] = []
    values_singles[alg] = []
    values_iterations[alg] = []
    rt_bar[alg] = []
  for m in mats:
    for alg in algs:
      # runtime
      rt_bar[alg].append(res[m][type][alg][ATTRIB_TIME])

      # bar y values
      if attrib == ATTRIB_ERROR or attrib == ATTRIB_DELTA or attrib == ATTRIB_SPEEDUP:
        values_bar[alg].append(res[m][type][alg][attrib])
      elif attrib == ATTRIB_TIME:
        values_bar[alg].append(res[m][type][ANAME_DOUBLES_CUSP][attrib] / res[m][type][alg][attrib])
      elif attrib == ATTRIB_GFLOPS:
        values_bar[alg].append(res[m][type][alg][attrib] / res[m][type][ANAME_DOUBLES_CUSP][attrib])

      # iteration counts
      if type == JACOBI:
        values_iterations[alg].append(str(res[m][type][alg]["iter"]))
      else:
        values_iterations[alg].append(str(res[m]['info']["spmv_iter"]))

      # single percentages
      values_singles[alg].append(str(round((res[m][type][alg]["singles"] / res[m]["info"]["nnz"]) * 100, ROUND_DIGITS)))

  # draw bar rectangles
  x = arange(len(mats))
  offset = -(len(mats) - 1) * barwidth / 2
  alg_i = 0
  if len(algs) == 1:
    tick_target = 0
  else:
    tick_target = int(len(algs) / 2) - 1
  for alg in list(reversed(algs)):
    lbl = MAPANAME_LEGEND[alg] if mapaname else alg
    if alg == ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE and type != SPMV:
      lbl = "1-step"  #: row-wise composite"
    # generate rectangle
    rects[alg] = ax.bar(
        x - offset,
        values_bar[alg],
        barwidth,
        label=lbl,
        #hatch=HATCHES[alg],
        color=COLORS[alg])
    offset += barwidth

    # draw the tick on the corresponding middle algorithm
    if alg_i == tick_target:
      if len(algs) % 2 == 0:
        xticks = [rect.get_x() for rect in rects[alg]]  # no need to adjust for even
      else:
        xticks = [rect.get_x() - rect.get_width() / 2 for rect in rects[alg]]  # need to adjust half to the left for odd

    alg_i += 1

  # axis labels and settings
  if attrib == ATTRIB_ERROR or attrib == ATTRIB_DELTA:
    ax.set_yscale("log")
  elif attrib == ATTRIB_SPEEDUP or attrib == ATTRIB_TIME or attrib == ATTRIB_GFLOPS:
    if type == CARDIAC:
      # y axis limits
      ax.set_ylim([0.75, 1.45])
      # horizontal lines
      for y in [0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]:
        ax.axhline(y, c="gray", ls="--", lw=0.5)
      ax.axhline(1.0, c="gray", ls="-", lw=0.8)

    else:
      # y axis limits
      ax.set_ylim([0.8, 1.13])
      # horizontal lines
      for y in [0.8, 0.85, 0.9, 0.95, 1.05, 1.1]:
        ax.axhline(y, c="gray", ls="--", lw=0.5)
      ax.axhline(1.0, c="gray", ls="-", lw=0.8)
  ax.set_xticks(xticks)
  if type == CARDIAC:
    ax.set_xticklabels([m[:len("heartXX")] for m in mats], rotation=20, ha="right", fontsize=fontsize)
  else:
    ax.set_xticklabels(mats, rotation=20, ha="right", fontsize=fontsize)
  # draw legend
  if legend:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        #ncol=2,
        fontsize=fontsize,
        #bbox_to_anchor=(0.0, 1.12, 1.0, 0.112),
        loc="upper left" if type == CARDIAC else "lower left",
        #mode="expand",
        #borderaxespad=0.0,
        framealpha=0.95)

  # draw labels
  for alg in algs:
    m_i = 0
    for rect, singles, iters in zip(rects[alg], values_singles[alg], values_iterations[alg]):
      ## COMMON ##
      # ax.text( # single percentage text
      #   rect.get_x() + rect.get_width() / 2,
      #   values_bar[alg][m_i],
      #   " " + "{0:.1f} ms".format(rt_bar[alg][m_i]),
      #   ha="center",
      #   va="bottom",
      #   rotation=90,
      #   fontsize=fontsize,
      #   color="black",
      # )
      ## JACOBI ##
      #if type == JACOBI:
      #ax.text( # iterations text
      #  rect.get_x() + rect.get_width() / 2,
      #  values_bar[alg][m_i],
      #  iters + " ",
      #  ha="center",
      #  va="top",
      #  rotation=90,
      #  fontsize=fontsize,
      #  color="black",
      #)
      ## SPMV ##
      # if attrib == ATTRIB_SPEEDUP:
      #   ax.text( # iterations text
      #     rect.get_x() + rect.get_width() / 2,
      #     values_bar[alg][m_i],
      #     singles + " ",
      #     ha="center",
      #     va="top",
      #     rotation=90,
      #     fontsize=fontsize,
      #     color="black",
      #   )

      m_i += 1

    if attrib == ATTRIB_ERROR or attrib == ATTRIB_DELTA:
      ax.set_yscale("log")

    # add parameters textbox to the side
  #if show_params:
  #  paramstext = pu.prep_params_text(
  #    res[mats[0]]["parameters"], algs, eps=epsilon_text
  #  )
  #  ax.annotate(
  #    paramstext,
  #    xy=(1.05, 0.97),
  #    xycoords="axes fraction",
  #    horizontalalignment="left",
  #    verticalalignment="top",
  #    fontsize="small",
  #    bbox=dict(fc="w", ec="gray", lw=0.5),
  #  )