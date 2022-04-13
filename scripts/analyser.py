from typing import List
from scipy.io import mmread
from scipy.sparse import csr_matrix
from statistics import stdev, median
import os
import numpy as np
import matplotlib.pyplot as plt

FLT_MAX = 3.4028235 * 1e+38
DIR_PATH = os.path.dirname(os.path.realpath(__file__))  # path of this file's directory


# Return the percentage of singles
def get_percentage(nzS, nzD):
  return nzS / (nzS + nzD) * 100.0


# Print a CSR matrix
def print_csr(csr):
  for i in range(csr.shape[0]):
    for j in range(csr.indptr[i], csr.indptr[i + 1]):
      print("{0}\t{1}\t{2}".format(i + 1, csr.indices[j] + 1, csr.data[j]))


# Row-wise precision decisions
def datadriven_rowwise(csr, r: float = 1, p: float = 99.0):
  hasPostFloatValue: bool = False
  row_is_single: List[bool] = [None] * csr.shape[0]
  nzS: int = 0
  nzD: int = 0
  for i in range(csr.shape[0]):
    # count occurences in range
    inrange = 0
    outrange = 0
    for j in range(csr.indptr[i], csr.indptr[i + 1]):
      if (csr.data[j] < -FLT_MAX or csr.data[j] > FLT_MAX):
        hasPostFloatValue = True
        break
      if (csr.data[j] <= r and csr.data[j] >= -r):
        inrange += 1
      else:
        outrange += 1

    # decide precision
    if hasPostFloatValue:
      row_is_single[i] = False
    else:
      if (inrange + outrange <= 100):
        # less than 100 values
        if (outrange <= 100 - p):
          # at most p exceptions
          row_is_single[i] = True
        else:
          row_is_single[i] = False
      else:
        # check percentage
        rowp = 100 * (inrange / (inrange + outrange))
        if (rowp >= p):
          # single row
          row_is_single[i] = True
        else:
          # double row
          row_is_single[i] = False

  # calculate nnz
  for i in range(len(row_is_single)):
    if row_is_single[i]:
      nzS += csr.indptr[i + 1] - csr.indptr[i]
    else:
      nzD += csr.indptr[i + 1] - csr.indptr[i]
  return [nzS, nzD, row_is_single]


# Entrywise precision decisions
def datadriven_entrywise(csr):
  r = 1.0  # range
  nzS = 0
  nzD = 0
  for i in range(csr.shape[0]):
    for j in range(csr.indptr[i], csr.indptr[i + 1]):
      # check range
      if (csr.data[j] <= r and csr.data[j] >= -r):
        # single entry
        nzS += 1
      else:
        # double entry
        nzD += 1
  return (nzS, nzD)


# Matrix-adaptive range calculation.
def absmean_shrink(csr, shrinkfactor: float = 10) -> float:
  return (sum([abs(c) for c in csr.data]) / csr.nnz) / shrinkfactor


# Extracts the diagonal and returns it togther with the new matrix.
def extract_diag(csr):
  numdiag: int = 0
  nz_i: int = 0
  newnz: int = csr.nnz - csr.shape[0]
  newrowptr: List[int] = [0] * (csr.shape[0] + 1)
  newvals: List[int] = [0] * newnz
  newcols: List[int] = [0] * newnz
  diag: List[csr.dtype] = [0] * csr.shape[0]

  # create matrix
  newrowptr[0] = 0
  for i in range(csr.shape[0]):
    newrowptr[i + 1] = newrowptr[i]
    for j in range(csr.indptr[i], csr.indptr[i + 1]):
      if i != csr.indices[j]:
        newcols[nz_i] = csr.indices[j]
        newvals[nz_i] = csr.data[j]
        newrowptr[i + 1] += 1
        nz_i += 1
      else:
        numdiag += 1
        diag[i] = csr.data[j]
  assert (nz_i == csr.nnz - csr.shape[0])
  assert (numdiag == csr.shape[0])
  return [csr_matrix((newvals, newcols, newrowptr), shape=csr.shape), diag]


def examine_matrix(matrix_name: str, extract_diagonal: bool):
  print("Reading {0} from disk.".format(matrix_name))
  fullpath = DIR_PATH + "/../res/" + matrix_name + ".mtx"  # path of input matrix
  mat = mmread(fullpath)  # coo
  mat = mat.tocsr()  # csr
  print("{3} read ({1}x{2}) with {0} nnz.".format(mat.nnz, mat.shape[0], mat.shape[1], matrix_name))
  if extract_diagonal:
    print("Extracting diagonal...")
    [mat, _] = extract_diag(mat)
    print("Diagonal extracted!")

  absdata = [abs(c) for c in mat.data]
  #std = stdev(absdata)
  avg = sum(absdata) / mat.nnz
  r = avg / 0.5  # shrink factor 10
  #r = avg + std
  #r = median(absdata)
  print("r: {0}\navg: {1}\nstdev: {2}\n".format(r, avg, 0))
  nzS, nzD, _ = datadriven_rowwise(mat, r=r)
  print("Percentage:", get_percentage(nzS, nzD))

  print("")
  #_, _, _ = plt.hist([abs(c) for c in mat.data], bins='auto')
  #plt.show()


def try_cardiac_matrices():
  for h in [3]:
    for s in ["uniform_forward_dt0001_fv"]:  # , "transviso_forward_dt0005_fv"]:
      matrix_name = "cardiac/heart0{0}_A_{1}".format(h, s)
      examine_matrix(matrix_name, True)


#########################################################################
if __name__ == "__main__":
  #examine_matrix("architect/mat_154652_qqp", False)
  #examine_matrix("architect/mat_154652_qqp", True)
  try_cardiac_matrices()
