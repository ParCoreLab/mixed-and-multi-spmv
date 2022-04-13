from utils import JACOBI, FMT_MIN_ITER, FMT_MAX_ITER, FMT_ALL, COLORS, MARKERS

_verbose = False


def jacobi_iterations(ax, res, m, algs, attrib, fmt, legend):
  epsilon = res[m]["info"]["epsilon"]
  if "iters" in res[m][JACOBI][algs[0]]["evaluations"]:
    if len(res[m][JACOBI][algs[0]]['evaluations']['iters']) == 0:
      print("No iterations for {0}".format(m))
      return
  try:
    if fmt == FMT_MAX_ITER:
      iter_limit = max([res[m][JACOBI][a]["iter"] for a in algs if res[m][JACOBI][a]["converged"]])
    elif fmt == FMT_MIN_ITER:
      iter_limit = min([res[m][JACOBI][a]["iter"] for a in algs if res[m][JACOBI][a]["converged"]])
    elif fmt == FMT_ALL:
      iter_limit = res[m]['info']['jacobi_iter']
    elif fmt > 0:
      iter_limit = fmt
  except:
    print("Exception in", m)
    iter_limit = res[m]['info']["jacobi_iter"]
  if _verbose:
    print("Printing up to", iter_limit)
  for a in algs:
    ax.scatter(
        res[m][JACOBI][a]["evaluations"]["iters"][:iter_limit],
        res[m][JACOBI][a]["evaluations"][attrib][:iter_limit],
        color=COLORS[a],
        marker=MARKERS[a],
    )

  if legend:
    ax.legend(algs, bbox_to_anchor=(1.05, 1), loc='upper left')

  # indicate steps
  for a in algs:
    if "steps" in res[m][JACOBI][a]["evaluations"] and len(res[m][JACOBI][a]["evaluations"]["steps"]) > 0:
      curtime = 0.0
      curstep = 1
      totaltime = res[m][JACOBI][a]["time"]
      [ssP, msP, dsP] = [0, 0, 0]  # step time percentages
      for i in range(len(res[m][JACOBI][a]["evaluations"]["steps"])):
        step = res[m][JACOBI][a]["evaluations"]["steps"][i]
        if step['cur'] == 1 and step['next'] == 2:  # single step -> mixed step
          ssP = (step['time'] - curtime) / (totaltime) * 100
          curstep += 1
        elif step['cur'] == 1 and step['next'] == 3:  # single step -> double step
          ssP = (step['time'] - curtime) / (totaltime) * 100
          curstep += 2
        elif step['cur'] == 2 and step['next'] == 3:  # mixed step -> double step
          msP = (step['time'] - curtime) / (totaltime) * 100
          curstep += 1
        else:
          print("Unknown step:\n\t", step, "\nMatrix\t", m)
          assert False

        # draw steps
        if iter_limit >= step['iter']:
          ax.axvline(step['iter'], c=COLORS[a], ls="-", lw=1.1)
        curtime = step['time']
      if curstep == 1:
        # finished at single
        ssP = 100
      elif curstep == 2:
        # finished at mixed
        msP = 100 - ssP
      else:
        # finished at double
        dsP = 100 - ssP - msP

      #assert(100 == ssP + msP + dsP)
      if _verbose:
        print("Steps for {3}: S {0} - M {1} - D {2}".format(round(ssP, 2), round(msP, 2), round(dsP, 2), a))

  ax.set_xlabel("iterations" + " (" + str(res[m]["info"]["jacobi_iter"]) + " max)")
  ax.set_ylabel("log(" + str(attrib) + ")")
  ax.set_title(m)

  # draw epsilon line
  ax.axhline(epsilon, c="gray", ls="--", lw=0.8)

  ax.set_yscale("log")