from json import load
import pathlib
from utils import *


def print_valid_matrices(res):
  mats = [m for m in list(res.keys()) if res[m]['error'] == None]
  print("[{0} / {1}] valid matrices:\n{2}".format(len(mats), len(list(res.keys())), mats))
  for m in [m for m in list(res.keys()) if res[m]['error'] != None]:
    print("Skipped {0}: {1}".format(m, res[m]['error']))


def print_empty_rows(res):
  mats = [m for m in list(res.keys()) if res[m]['error'] == None]
  matsnonempty = [m for m in mats if res[m]['info']['empty_rows'] != 0]
  print("[{0} / {1}] matrices have empty rows:\n{2}".format(len(matsnonempty), len(mats), matsnonempty))


def print_outliers(res):
  mats = [m for m in list(res.keys()) if res[m]['error'] == None]
  alg = 'DD (RO)'
  for m in mats:
    if res[m][SPMV][alg]['speedup'] > 2:
      print("{0:>20} has speedup {1}\t{2:>7} x {3:>7}\t{4} entries.".format(m, res[m][SPMV][alg]['speedup'],
                                                                            res[m]['info']['rows'],
                                                                            res[m]['info']['cols'],
                                                                            res[m]['info']['nnz']))
  nzlim = 100000
  less = len([m for m in mats if res[m]['info']['nnz'] < nzlim])
  more = len(mats) - less
  print("{0} / {1} have less, {2} / {1} have more than {3} non zeros.".format(less, len(mats), more, nzlim))


def print_aggr_speedups(res, mats, alg, type):
  for a in [alg, ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE]:  #, ANAME_SINGLES_DR_CUSP]:
    spups = [res[m][type][a][ATTRIB_SPEEDUP] for m in mats]
    print("{0}: MAX SPEEDUP: {1}\n"
          "{0}: AVG SPEEDUP: {2}\n"
          "{0}: SPEEDUPs > 1.0: {3}".format(a, max(spups),
                                            sum(spups) / len(spups), len([s for s in spups if s > 1.0])))


def print_jacobi_candidates(res):
  mats = [m for m in list(res.keys()) if res[m]['error'] == None]
  alg = ANAME_DOUBLES_CUSP
  idx = 3
  cnt = 0
  ans_mats = []
  for m in mats:
    #if res[m][JACOBI][alg]['evaluations']['deltas'][idx] < res[m][JACOBI][alg]['evaluations']['deltas'][0]:
    if res[m][JACOBI][alg]['error'] < 1:
      print("{0:>11} delta: {1:>11} error: {2:>11} nnz: {3:>11} iter: {4:>11}".format(
          m, res[m][JACOBI][alg]['delta'], res[m][JACOBI][alg]['error'], res[m]['info']['nnz'],
          res[m][JACOBI][alg]['iter']))
      cnt += 1
      ans_mats.append(m)
  print("{0} out of {1} matrices!".format(cnt, len(mats)))
  return ans_mats


def print_split_amounts(res):
  # Split amounts
  # round((res[m][type][alg]["singles"] / res[m]["info"]["nnz"])* 100, ROUND_DIGITS)
  mats = [m for m in list(res.keys()) if res[m]['error'] == None]
  SINGLEPERCENT = lambda m, alg: round(
      res[m][SPMV][alg]["singles"] / (res[m][SPMV][alg]['singles'] + res[m][SPMV][alg]['doubles']) * 100, 3)

  print("Entrywise Splits > 0:\nDD {0} / {1}\n".format(
      sum([(1 if SINGLEPERCENT(m, ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT) > 0 else 0) for m in mats]), len(mats)))
  print("Entrywise Avg. Singles:\nDD {0}\n".format(
      sum([SINGLEPERCENT(m, ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT) for m in mats]) / len(mats)))
  print("Entrywise Min. Singles:\nDD {0}\n".format(
      min([
          SINGLEPERCENT(m, ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT)
          for m in mats
          if SINGLEPERCENT(m, ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT) > 0
      ])))
  print("Entrywise Max. Singles:\nDD {0}\n".format(
      max([
          SINGLEPERCENT(m, ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT)
          for m in mats
          if SINGLEPERCENT(m, ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT) < 100
      ])))
  print("Rowwise Splits > 0:\nDD {0} / {1}\n".format(
      sum([(1 if SINGLEPERCENT(m, ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT) > 0 else 0) for m in mats]), len(mats)))
  print("Rowwise Avg. Singles:\nDD {0}\n".format(
      sum([SINGLEPERCENT(m, ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT) for m in mats]) / len(mats)))
  print("Rowwise Min. Singles:\nDD {0}\n".format(
      min([
          SINGLEPERCENT(m, ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT)
          for m in mats
          if SINGLEPERCENT(m, ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT) > 0
      ])))
  print("Rowwise Max. Singles:\nDD {0}\n".format(
      max([
          SINGLEPERCENT(m, ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT)
          for m in mats
          if SINGLEPERCENT(m, ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT) < 100
      ])))


def print_jacobi_summary(res, mats):
  for m in mats:
    print("\n--- {0} ({1} nz)---".format(m, res[m]['info']['nnz']))
    for a in res[m][JACOBI]:
      print("{0: <9}\tSTEP:{1: <9}\tITER: {2: <4}\tSPLIT: {3: <12}\tSPUP: {4: <9}\tERR: {5: <9}\tGAMMA: {6}".format(
          a, str([s['iter'] for s in res[m][JACOBI][a]['evaluations']['steps']]), res[m][JACOBI][a]['iter'],
          res[m][JACOBI][a]['percentage'], res[m][JACOBI][a]['speedup'], res[m][JACOBI][a]['error'],
          res[m][JACOBI][a]['gamma']))


def print_those_above_and_below_1_speedup(res, mats):
  #mats = [m for m in list(res.keys()) if res[m]['error'] == None]
  sp_e_more = 0
  sp_e_less = 0
  sp_r_more = 0
  sp_r_less = 0
  for m in mats:
    sp_e = res[m][SPMV][ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT][ATTRIB_SPEEDUP]
    sp_r = res[m][SPMV][ANAME_DOUBLES_MIXED_ROWWISE_SPLIT][ATTRIB_SPEEDUP]
    if sp_e > 1.0:
      sp_e_more += 1
    else:
      sp_e_less += 1
    if sp_r > 1.0:
      sp_r_more += 1
    else:
      sp_r_less += 1
  print("ENTRYWISE: {0} above 1.0\t{1} below 1.0\n".format(sp_e_more, sp_e_less))
  print("ROWWISE: {0} above 1.0\t{1} below 1.0\n".format(sp_r_more, sp_r_less))


if __name__ == "__main__":
  input_relative_path = "evaluations/in/paper/spmv-all-opt1-p99-2k-simula"  # "evaluations/in/jacobi.json"
  input_path = str(pathlib.Path(__file__).parent.resolve()) + "/../" + input_relative_path + '.json'

  with open(input_path) as f:
    res = load(f)
    print_speedups_per_algorithm(res)
  pass