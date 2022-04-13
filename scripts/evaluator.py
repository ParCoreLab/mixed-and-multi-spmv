#!/usr/bin/env python3

import gzip
import json
import os
import requests
import shutil
import subprocess
import sys
import tarfile
import random
import string

from datetime import datetime
from os import path, listdir
from os.path import dirname, isfile, join, realpath
from typing import Dict, Optional, List

from plots import Plots

SAVE_TO_PATH = ""  # global variable to save results
SAVE_PER_RUN = 15  # in the online case, save results every 100 matrices


# Parses the output of an evaluation.
def parseOutput(output: subprocess.CompletedProcess) -> Optional[Dict]:
  # to read NaN numbers
  _NANNABLE_NUMBER = lambda xstr: float('NaN') if xstr.lower() in ['nan', '-nan', 'inf', '-inf'] else float(xstr)

  # Note that in Linux, it uses "\n" for enter, whereas in Windows it uses "\n\r".
  lines = output.stdout.decode("utf-8").split("\n")
  _prefix = "scrp "
  lines = [l[len(_prefix):] for l in lines if l.startswith(_prefix)]

  ans = {"spmv": {}, "jacobi": {}, "cardiac": {}, "error": None, "info": None, "parameters": None}
  for line in lines:
    tokens = list(map(lambda s: s.strip(), line.split("||")))
    typeToken = tokens[0]
    # scrp info || <matrix>  || <rows> || <cols> || <nnz> || <spmv_iters> || <jacobi_iters_max> || <epsilon>
    if typeToken == "info":
      ans["info"] = {
          "path": tokens[1],
          "code": output.returncode,
          "rows": int(tokens[2]),
          "cols": int(tokens[3]),
          "is_square": int(tokens[2]) == int(tokens[3]),
          "nnz": int(tokens[4]),
          "spmv_iter": int(tokens[5]),
          "jacobi_iter": int(tokens[6]),
          "cardiac_iter": int(tokens[7]),
          "is_symmetric": tokens[8] == '1',
          "run_option": int(tokens[9]),
          "empty_rows": int(tokens[10]),
          "avg_nz_inrow": float(tokens[11]),
          "min_nz_inrow": int(tokens[12]),
          "max_nz_inrow": int(tokens[13]),
          "dd_range": float(tokens[14]),
          "dd_percentage": float(tokens[15]),
          "dd_shrink": float(tokens[16]),
          "mattype": tokens[17]
      }
    # scrp error || message
    elif typeToken == "error":
      ans['error'] = tokens[1]
    # scrp spmv || title || time || speedup || error || doubles || singles || percentage
    elif typeToken == "spmv":
      ans["spmv"][tokens[1]] = {
          "time": float(tokens[2]),
          "speedup": float(tokens[3]),
          "error": _NANNABLE_NUMBER(tokens[4]),
          "doubles": int(tokens[5]),
          "singles": int(tokens[6]),
          "percentage": float(tokens[7])
      }
    # scrp cardiac || title || time || speedup || error || doubles || singles || percentage
    elif typeToken == "cardiac":
      ans["cardiac"][tokens[1]] = {
          "time": float(tokens[2]),
          "speedup": float(tokens[3]),
          "error": _NANNABLE_NUMBER(tokens[4]),
          "doubles": int(tokens[5]),
          "singles": int(tokens[6]),
          "percentage": float(tokens[7])
      }
    # scrp jacobi || title || time || speedup || error || gamma || iters || doubles || singles || percentage
    elif typeToken == "jacobi":
      ans["jacobi"][tokens[1]] = {
          "time": float(tokens[2]),
          "speedup": float(tokens[3]),
          "error": _NANNABLE_NUMBER(tokens[4]),
          "delta": _NANNABLE_NUMBER(tokens[5]),
          "doubles": int(tokens[6]),
          "singles": int(tokens[7]),
          "percentage": float(tokens[8]),
      }
    # scrp skip || <error> || <delta>
    elif typeToken == "skip":
      ans['error'] = "FP64 Jacobi is bad. (e: {0})".format(tokens[1])
    else:
      raise Exception("Unknown Type Token:" + typeToken)

  # an unknown error would mean that everything is empty
  if (ans['error'] == None and ans['info'] == None):
    ans['error'] = "Unknown error."
  return ans


def _save_plots(res, directory, filename):
  if not res:
    # no results
    print("\nfailure there are no results.")
  else:
    # save results
    Plots(res, save_to_path=directory, save_dict_as=filename)


# Evaluate Cardiac simulation matrices
def evaluate_cardiac(executable: str, cmdlineopts: str, filename: str) -> Optional[Dict]:
  assert (SAVE_TO_PATH != "")

  # hardcode matrix and vector paths
  stem = dirname(dirname(realpath(__file__)))  # dir of this script
  matrixVectorPaths = [
      #('heart01_A_transviso_forward_dt0005_fv.mtx', 'heart01_b_ap_0005ms.mtx'), // too small
      #('heart01_A_uniform_forward_dt0001_fv.mtx', 'heart01_b_ap_0005ms.mtx'), // too small
      #('heart02_A_transviso_forward_dt0005_fv.mtx', 'heart02_b_ap_0005ms.mtx'), // transviso
      ('heart02_A_uniform_forward_dt0001_fv.mtx', 'heart02_b_ap_0005ms.mtx'),
      #('heart03_A_transviso_forward_dt0005_fv.mtx', 'heart03_b_ap_0005ms.mtx'), // transviso
      ('heart03_A_uniform_forward_dt0001_fv.mtx', 'heart03_b_ap_0005ms.mtx'),
      #('heart04_A_transviso_forward_dt0005_fv.mtx', 'heart04_b_ap_0005ms.mtx'), // transviso
      ('heart04_A_uniform_forward_dt0001_fv.mtx', 'heart04_b_ap_0005ms.mtx'),
      ('heart05_A_uniform_forward_dt0001_fv.mtx', 'heart05_b_ap_0005ms.mtx'),
      ('heart06_A_uniform_forward_dt0001_fv.mtx', 'heart06_b_ap_0005ms.mtx'),
      ('heart07_A_uniform_forward_dt0001_fv.mtx', 'heart07_b_ap_0005ms.mtx'),
  ]

  currentRuns = 1
  totalRuns = len(matrixVectorPaths)
  all_results = {}

  for matrixPath, vectorPath in matrixVectorPaths:
    matrixPath = stem + "/res/cardiac/" + matrixPath
    vectorPath = stem + "/res/cardiac/" + vectorPath
    # check if matrix and vector both exists
    if (not (isfile(matrixPath) and isfile(vectorPath))):
      print("Matrix Path", matrixPath, "and Vector Path", vectorPath, "are invalid. Skipping...")
      continue

    # run
    cmd = "%s -m %s -x %s -s --cardiac %s" % (executable, matrixPath, vectorPath, cmdlineopts)
    print("[", currentRuns, "/", totalRuns, "]: ", cmd)
    output = subprocess.run(cmd.split(), capture_output=True)
    parsed = parseOutput(output)

    # save
    mtx_name = path.basename(matrixPath)[:-4]
    all_results[mtx_name] = parsed
    if parsed == None and 'error' not in parsed:
      print("Unknown error in", mtx_name)
      all_results[mtx_name] = {
          "spmv": {},
          "jacobi": {},
          "centralitytime": {},
          "error": "Unknown error.",
          "info": None,
          "parameters": None
      }
    elif parsed['error'] != None:
      print("Error in", mtx_name, ":", parsed['error'])
    else:
      print("Success.")
    currentRuns += 1

  _save_plots(all_results, SAVE_TO_PATH, filename)


# Evaluate SpMV and Jacobi locally with this script.
def evaluate_local(executable: str, matrixPaths: List[str], cmdlineopts: str, filename: str) -> Optional[Dict]:
  assert (SAVE_TO_PATH != "")
  currentRuns = 1
  totalRuns = len(matrixPaths)
  all_results = {}

  for matrixPath in matrixPaths:
    # run
    cmd = "%s -m %s -s %s" % (executable, matrixPath, cmdlineopts)
    print("[", currentRuns, "/", totalRuns, "]: ", cmd)
    output = subprocess.run(cmd.split(), capture_output=True)
    parsed = parseOutput(output)

    # save
    mtx_name = path.basename(matrixPath)[:-4]
    all_results[mtx_name] = parsed
    if parsed == None and 'error' not in parsed:
      print("Unknown error in", mtx_name)
      all_results[mtx_name] = {
          "spmv": {},
          "jacobi": {},
          "centralitytime": {},
          "error": "Unknown error.",
          "info": None,
          "parameters": None
      }
    elif parsed['error'] != None:
      print("Error in", mtx_name, ":", parsed['error'])
    else:
      print("Success.")
    currentRuns += 1

  _save_plots(all_results, SAVE_TO_PATH, filename)


# Evaluate SpMV and Jacobi with matrices downloaded one by one from SuiteSparse with this script.
def evaluate_online(executable: str, mm_index_path: str, cmdlineopts: str, filename: str) -> Optional[Dict]:
  assert (SAVE_TO_PATH != "")
  mm_url = "http://sparse-files.engr.tamu.edu/MM/%s.tar.gz"

  key = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
  stem = dirname(dirname(realpath(__file__)))  # dir of this script
  save_to = join(stem, "evaluations", "all", key, "cache")
  extract_to = join(stem, "evaluations", "all", key)
  os.makedirs(save_to, exist_ok=True)

  currentRuns = 1
  all_results = {}

  mm_index = [l.strip() for l in open(mm_index_path, "r").read().split("\n")]
  totalRuns = len(mm_index)

  for m in mm_index:
    try:
      mtx_name = m.split("/")[1].strip()

      # downlaod matrix
      url = mm_url % m
      res = requests.head(url)
      assert res.status_code == 200, "mm.index file corrupted? got %s from url: %s" % (res.status_code, url % m)
      size = res.headers["Content-Length"]
      out_ar = join(save_to, "%s.tar.gz" % mtx_name)
      res = requests.get(url, stream=True)
      print("\nFetching %s -> %s (%s bytes)..." % (url, out_ar, size))
      f = open(out_ar, "wb")
      shutil.copyfileobj(res.raw, f)
      f.close()

      # extract matrix
      extracted_tmp = join(extract_to, mtx_name, "%s.mtx" % mtx_name)
      extracted_mtx = join(extract_to, "%s.mtx" % mtx_name)
      tf = tarfile.open(out_ar, "r|gz")
      sys.stdout.write("Extracting %s -> %s ... " % (out_ar, extracted_mtx))
      sys.stdout.flush()
      os.chdir(extract_to)
      tf.extractall()
      shutil.move(extracted_tmp, extracted_mtx)
      shutil.rmtree(join(extract_to, mtx_name))  # remove subdirectory and any possible junk
      tf.close()
      extracted_size = path.getsize(extracted_mtx)
      sys.stdout.write(" OK (%s bytes)\n" % extracted_size)

      # run evaluation
      cmd = "%s -m %s -s %s" % (executable, extracted_mtx, cmdlineopts)
      print("[", currentRuns, "/", totalRuns, "]: ", cmd)
      output = subprocess.run(cmd.split(), capture_output=True)
      parsed = parseOutput(output)

      # remove
      os.remove(out_ar)
      os.remove(extracted_mtx)

      # save
      all_results[mtx_name] = parsed
      if parsed == None and 'error' not in parsed:
        print("Unknown error in", mtx_name)
        all_results[mtx_name] = {
            "spmv": {},
            "jacobi": {},
            "timings": {},
            "error": "Unknown error.",
            "info": None,
            "parameters": None
        }
      elif parsed['error'] != None:
        print("Error in", mtx_name, ":", parsed['error'])
      else:
        print("Success.")

      if (currentRuns % SAVE_PER_RUN == 0):
        print("Saving partial results... [{0} / {1}]".format(currentRuns, totalRuns))
        _save_plots(all_results, SAVE_TO_PATH, filename)
      currentRuns += 1
    except Exception as err:
      #raise err
      print("{0} exception in {1}. Skipping.".format(str(type(err)), m))

  _save_plots(all_results, SAVE_TO_PATH, filename)


if __name__ == "__main__":
  dir_path = dirname(realpath(__file__))  # dir of this file
  SAVE_TO_PATH = dir_path + "/../evaluations/out"
  EXECUTABLE_DIR = "../bin"
  EXECUTABLE_NAME = "spmv"
  DO_CARDIAC = False
  CMDLINE_OPTIONS = []
  MATRIX_PATHS = []
  MM_INDEX_PATH = ""
  i = 1
  FILENAME = None
  while i < len(sys.argv):
    if sys.argv[i] == "-d":  # use matrices in a directory
      i += 1
      MATRIX_DIR = sys.argv[i]
      MATRIX_PATHS = [
          MATRIX_DIR + "/" + f for f in listdir(MATRIX_DIR) if isfile(join(MATRIX_DIR, f)) and f.endswith(".mtx")
      ]
    elif sys.argv[i] == "-a":  # evaluate online via mm.index
      i += 1
      MM_INDEX_PATH = sys.argv[i]
    elif sys.argv[i] == "-s":  # save to a given path
      i += 1
      SAVE_TO_PATH = sys.argv[i]
    elif sys.argv[i] == "-f":  # save as given filename
      i += 1
      FILENAME = sys.argv[i]
    elif sys.argv[i] == "-c":  # cmdline args to the executable
      i += 1
      CMDLINE_OPTIONS = sys.argv[i:]
      break
    elif sys.argv[i] == "-m":  # single matrix input
      i += 1
      MATRIX_PATHS.append(sys.argv[i])
    elif sys.argv[i] == "--cardiac":  # do cardiac simulation
      DO_CARDIAC = True
    else:
      print("Usage: python3 " + sys.argv[0] + "\n"
            "-s <path>        Save under the given path.\n"
            "-c <args...>     Pass cmdline args to the executable.\n"
            "-m <path>        Give a single matrix input.\n"
            "-d <path>        Use matrices in a given directory.\n"
            "-a <path>        Use matrices online, indexed by the given file (mm.index). Takes a long time!\n"
            "-f <name>        Filename to save the results dict.\n"
            "--cardiac        Special command for Cardiac Simulation evaluations.\n")
      exit(0)
    i += 1

  # Run a new evaluation
  if "win32" in sys.platform:
    EXECUTABLE_NAME += ".exe"  # need to specify .exe for Windows

  EXECUTABLE_NAME = dir_path + "/" + EXECUTABLE_DIR + "/" + EXECUTABLE_NAME
  # Initial checks
  if not path.exists(EXECUTABLE_NAME):
    print("error No such executable", EXECUTABLE_NAME)
    exit(-1)

  if any((not path.exists(matrix_path)) for matrix_path in MATRIX_PATHS):
    print("error No such matrix path, check the names. Paths:\n", MATRIX_PATHS)
    exit(-1)

  CMDLINE_OPTIONS = " ".join(CMDLINE_OPTIONS)
  if FILENAME == None:
    FILENAME = datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".json"
  if DO_CARDIAC:
    print("Evaluating cardiac simulation")
    evaluate_cardiac(EXECUTABLE_NAME, CMDLINE_OPTIONS, FILENAME)
  elif not MM_INDEX_PATH:
    print("Evaluating matrices:", MATRIX_PATHS)
    evaluate_local(EXECUTABLE_NAME, MATRIX_PATHS, CMDLINE_OPTIONS, FILENAME)
  else:
    print("Evaluating all matrices from:", MM_INDEX_PATH)
    evaluate_online(EXECUTABLE_NAME, MM_INDEX_PATH, CMDLINE_OPTIONS, FILENAME)
