from numpy import arange
from utils import *


def jacobi_steps_bar(ax, res, mats, algs, barwidth, fontsize, legend):
  values_bar = {}
  values_singles = {}
  values_iterations = {}
  rects = {}
  for a in algs:
    values_bar[a] = {'s': [], 'm': [], 'd': []}
    values_singles[a] = []
    values_iterations[a] = []
    rects[a] = {}
  for m in mats:
    for a in algs:
      [ssP, msP, dsP] = [0, 0, 0]  # step time percentages
      if res[m][JACOBI][a]["converged"]:
        curstep = 0
        if "steps" in res[m][JACOBI][a]["evaluations"] and len(res[m][JACOBI][a]["evaluations"]["steps"]) > 0:
          curtime = 0.0
          curstep = 1
          totaltime = res[m][JACOBI][a]["time"]
          for i in range(len(res[m][JACOBI][a]["evaluations"]["steps"])):
            step = res[m][JACOBI][a]["evaluations"]["steps"][i]
            if step['cur'] == 1 and step['next'] == 2:  # single step -> mixed step
              ssP = (step['time'] - curtime) / (totaltime) * 100.0
              curstep += 1
            elif step['cur'] == 1 and step['next'] == 3:  # single step -> double step
              ssP = (step['time'] - curtime) / (totaltime) * 100.0
              curstep += 2
            elif step['cur'] == 2 and step['next'] == 3:  # mixed step -> double step
              msP = (step['time'] - curtime) / (totaltime) * 100.0
              curstep += 1
            else:
              print("Unknown step:\n\t", step, "\nMatrix\t", m)
              assert False
            curtime = step['time']
        if curstep == 1:  # finished at single
          ssP = 100.0
        elif curstep == 2:  # finished at mixed
          msP = 100.0 - ssP
        else:  # finished at double
          dsP = 100.0 - ssP - msP
      if a == ANAME_SINGLES_MIXED_ENTRYWISE_SPLIT:
        ssP = dsP
        dsP = 0
      elif a == ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT or a == ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT:
        msP = dsP
        dsP = 0
      values_bar[a]['s'].append(ssP)
      values_bar[a]['m'].append(msP)
      values_bar[a]['d'].append(dsP)

  # draw bar rectangles
  x = arange(len(mats))
  offset = -(len(mats) - 1) * barwidth / 2
  alg_i = 0
  if len(algs) == 1:
    tick_target = 0
  else:
    tick_target = int(len(algs) / 2) - 1

  for alg in algs:
    # generate rectangle
    rects[alg]['s'] = ax.bar(x - offset, values_bar[alg]['s'], barwidth, label=alg, color=COLORS[ANAME_SINGLES_DR_CUSP])
    rects[alg]['m'] = ax.bar(x - offset,
                             values_bar[alg]['m'],
                             barwidth,
                             label=alg,
                             color=COLORS[ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT],
                             bottom=values_bar[alg]['s'])
    rects[alg]['d'] = ax.bar(x - offset,
                             values_bar[alg]['d'],
                             barwidth,
                             label=alg,
                             color=COLORS[ANAME_DOUBLES_CUSP],
                             bottom=[a + b for a, b in zip(values_bar[alg]['s'], values_bar[alg]['m'])])
    offset += barwidth

    # draw the tick on the corresponding middle algorithm
    if alg_i == tick_target:
      if len(algs) % 2 == 0:
        xticks = [rect.get_x() for rect in rects[alg]['s']]  # no need to adjust for even
      else:
        xticks = [rect.get_x() - rect.get_width() / 2 for rect in rects[alg]['s']
                 ]  # need to adjust half to the left for odd

    alg_i += 1

  # add texts
  for alg in algs:
    for rectS in rects[alg]['s']:
      ax.text(
          rectS.get_x() + rectS.get_width() / 2,
          0,
          " " + alg,
          ha="center",
          va="bottom",
          rotation=90,
          fontsize=fontsize,
          color="black",
      )

  # add x labels
  ax.set_xticks(xticks)
  ax.set_xticklabels(mats, rotation=55, ha="right", fontsize="x-small")

  # percentage limits
  ax.set_ylim([0, 100])
