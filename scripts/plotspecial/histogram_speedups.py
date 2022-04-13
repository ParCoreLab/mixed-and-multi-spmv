from numpy import NaN
from utils import *

# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html


def histogram_speedups(ax, res, mats, algs, type, legend):
  #bins = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
  bins = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
  #bins = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35,  1.4, 1.45, 1.5, 1.55, 1.6]

  xs = []
  c = []
  for alg in algs:
    x = []
    for m in mats:
      x.append(res[m][type][alg][ATTRIB_SPEEDUP])
    xs.append(x)
    c.append(COLORS[alg])  # add color
  labels = [MAPANAME_LEGEND[a] for a in algs] if legend else None

  arrs, _, patches = ax.hist(xs, color=c, bins=bins, label=labels, cumulative=False, density=False, stacked=False)

  # legend
  if legend:
    ax.legend(
        #ncol=1,
        fontsize="x-large",
        #bbox_to_anchor=(0.0, 1.12, 1.0, 0.112),
        loc="center right",
        #mode="expand",
        #borderaxespad=0.0,
    )

  # speedup 1.0 line
  ax.axvline(1.0, c="gray", ls="--", lw=0.8)

  axtwin = ax.twinx()

  # find single percentages in each bin
  middle_bin_x = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
  single_percentages_E = {i: [] for i in range(len(bins) - 1)}
  single_percentages_R = {i: [] for i in range(len(bins) - 1)}
  single_percentages_Eb = {i: [] for i in range(len(bins) - 1)}
  single_percentages = {}
  for alg in algs:
    single_percentages[alg] = {i: [] for i in range(len(bins) - 1)}
    # calculate bins
    for m in mats:
      for i in range(len(middle_bin_x)):
        speedup = res[m][type][alg][ATTRIB_SPEEDUP]
        if speedup >= bins[i] and speedup < bins[i + 1]:
          #print("E {0} > {1}".format(sE, bins[i]))
          single_percentages[alg][i].append(res[m][type][alg][ATTRIB_SPLIT_PERCENTAGE] * 100.0)
          break
    # average them
    for i in range(len(middle_bin_x)):
      arr = single_percentages[alg][i]
      single_percentages[alg][i] = NaN if len(arr) == 0 else sum(arr) / len(arr)
    single_percentages[alg] = [single_percentages[alg][i] for i in range(len(middle_bin_x))]

    axtwin.plot(middle_bin_x, single_percentages[alg], color=COLORS[alg], marker=MARKERS[alg])
  axtwin.set_ylabel("Percentage of NNZ stored in FP32 (Lines)", fontsize='x-large')
  axtwin.set_ylim([0.0, 100.0])  # 0-1, 100+1
  #axtwin.text(middle_bin_x[-1], single_percentages_E[-1], "Entry-wise\nSplit", ha='left', va='center')

  # add x labels for bins
  xtick_labels = ["[{0}, {1})".format(bins[i], bins[i + 1]) for i in range(len(middle_bin_x))]
  ax.set_xticks(middle_bin_x)
  ax.set_xticklabels(xtick_labels, fontsize='large')
