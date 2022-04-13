from utils import ANAME_DOUBLES_CUSP, JACOBI, FMT_MAX_ITER, FMT_MIN_ITER, FMT_ALL, COLORS, MARKERS

_DELTA_BREAKPOINT_U = 1.15
_DELTA_BREAKPOINT_L = 0.85


def delta_breakpoints(axs, res, m, algs, fmt, legend):

  # define the num iters to display
  if fmt == FMT_MAX_ITER:
    iter_limit = max([res[m][JACOBI][a]["iter"] for a in algs if res[m][JACOBI][a]["converged"]])
    print("Printing up to", iter_limit)
  elif fmt == FMT_MIN_ITER:
    iter_limit = min([res[m][JACOBI][a]["iter"] for a in algs if res[m][JACOBI][a]["converged"]])
    print("Printing up to", iter_limit)

  if legend:
    pass  # no legend for this guy now

  for a in algs:
    iters = []
    deltas = []
    diffs = [-1]
    # positions
    if fmt == FMT_ALL:
      iters = res[m][JACOBI][a]["evaluations"]["iters"]
      deltas = res[m][JACOBI][a]["evaluations"]["deltas"]
    else:
      iters = res[m][JACOBI][a]["evaluations"]["iters"][:iter_limit]
      deltas = res[m][JACOBI][a]["evaluations"]["deltas"][:iter_limit]

    for i in range(1, len(deltas)):
      if (deltas[i - 1] != 0):
        diffs.append(deltas[i] / deltas[i - 1])
      else:
        diffs.append(diffs[-1])  # we expect this not to occur in the first iteration

    # show deltas
    axs[0].scatter(iters, deltas, color=COLORS[a], marker=MARKERS[a])
    axs[0].set_yscale('log')
    # show difference and breakpoint
    axs[1].scatter(iters[1:], diffs[1:], color=COLORS[a], marker=MARKERS[a])
    axs[1].axhline(_DELTA_BREAKPOINT_U, color='black', ls="--", lw=0.8)
    axs[1].axhline(_DELTA_BREAKPOINT_L, color='black', ls="--", lw=0.8)
    axs[1].set_yscale('log')

    # draw steps
    if "steps" in res[m][JACOBI][a]["evaluations"] and len(res[m][JACOBI][a]["evaluations"]["steps"]) > 0:
      for i in range(len(res[m][JACOBI][a]["evaluations"]["steps"])):
        step = res[m][JACOBI][a]["evaluations"]["steps"][i]
        if iter_limit >= step['iter']:
          axs[0].axvline(step['iter'], c=COLORS[a], ls="-", lw=0.9)
          axs[1].axvline(step['iter'], c=COLORS[a], ls="-", lw=0.9)