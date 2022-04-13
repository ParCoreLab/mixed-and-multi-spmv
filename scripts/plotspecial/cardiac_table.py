from math import floor

from numpy.core.fromnumeric import sort
from utils import *

_TITLE = ['Matrix', 'FP32', 'Baseline', '1-Step', '2-Step', '3-Step']
_MATRIX_SPC = 20
_REST_SPC = 17


def _latex_preprocess(line, residuals_from_this_line):
  # change xxxe-y to xxx \times 10^{-y} (or +y respectively)
  for i in range(residuals_from_this_line, len(line)):
    v = line[i].split('e')
    v[1] = v[1].replace('+', '').replace('0', '')
    if v[1] == '':
      v = v[0]
    else:
      v = '$' + v[0] + ' \\times 10^{' + v[1] + '}$'
    line[i] = v
  return line


def _print_line(contents):
  print("│", end='')
  print("{0:<{1}}│".format(contents[0], _MATRIX_SPC), end='')
  for c in contents[1:]:
    print("{0:<{1}}│".format(c, _REST_SPC), end='')
  print("")


def cardiac_table(res, mats, save_as, save_to_path):
  DECIMAL_PTS_PERCENTAGE = 1
  DECIMAL_PTS_RESIDUAL = 2

  # HEADER
  chosen_title = _TITLE
  _HLINE = ['─' * _MATRIX_SPC] + ['─' * _REST_SPC] * (len(chosen_title) - 1)
  if save_as != None:
    lines = [' & '.join(chosen_title) + ' \\\\']
    lines.append('\\midrule')
  else:
    _print_line(_HLINE)
    _print_line(chosen_title)
    _print_line(_HLINE)

  # BODY
  for m in mats:
    line = []
    line.append(m[:len("heartXX")])  # heartXX
    #line.append("{0}".format(floor(res[m]['info'][ATTRIB_NNZ])))    # NNZ
    line.append("{0:.{1}e}".format(res[m][CARDIAC][ANAME_SINGLES_DR_CUSP][ATTRIB_ERROR],
                                   DECIMAL_PTS_RESIDUAL))  # FP32-64
    line.append("{0:.{1}e}"
      #" {2:.{3}f}%"
      .format(
      res[m][CARDIAC][ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE][ATTRIB_ERROR], DECIMAL_PTS_RESIDUAL
      #,res[m][CARDIAC][ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE][ATTRIB_SPLIT_PERCENTAGE] * 100, DECIMAL_PTS_PERCENTAGE
      )) # baseline
    line.append("{0:.{1}e}"
      #" {2:.{3}f}%"
      .format(
      res[m][CARDIAC][ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT][ATTRIB_ERROR], DECIMAL_PTS_RESIDUAL
      #,res[m][CARDIAC][ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT][ATTRIB_SPLIT_PERCENTAGE] * 100, DECIMAL_PTS_PERCENTAGE
      )) # 1-step
    line.append("{0:.{1}e}".format(res[m][CARDIAC][ANAME_MULTI_DD_R_DOUBLE][ATTRIB_ERROR],
                                   DECIMAL_PTS_RESIDUAL))  # 2-step
    line.append("{0:.{1}e}".format(res[m][CARDIAC][ANAME_MULTI_SINGLE_DD_R_DOUBLE][ATTRIB_ERROR],
                                   DECIMAL_PTS_RESIDUAL))  # 3-step
    if save_as != None:
      #lines.append(' & '.join(_latex_preprocess(line, 1)) + ' \\\\')
      lines.append(' & '.join(line) + ' \\\\')
    else:
      _print_line(line)

  # FOOTER and SAVE
  if save_as != None:
    # save lines as csv
    path = save_to_path + '/' + save_as
    print("Saved to", path)
    f = open(path, 'w')
    latex_text = '\n'.join(lines).replace('_', '\\_').replace('%', '\\%')
    f.write(latex_text)
    f.close()
  else:
    _print_line(_HLINE)
