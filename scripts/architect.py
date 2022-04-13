import random
import sys
from string import ascii_lowercase
import os
"""
 Architect (reference to movie Matrix) is a helper script to create toy sparse MatrixMarket matrices.
 It is created in purpose of generating toy matrices to experiment on.
"""

SEED = 2021  # fixed
RAND_STR = lambda length: "".join(random.choice(ascii_lowercase) for i in range(length))
BASE_NAME = "mat"
MAT_TYPE_REAL = "real"
MAT_TYPE_BINARY = "pattern"
MAT_TYPE_INTEGER = "integer"
PRINT = False  # If false, writes to file, if true writes to console (for testing)
FULL_CMD = ""  # this is the command input
random.seed(SEED)


def CHAR_TO_TYPE(c):
  '''Convert the cmdline input type character to MM type'''
  if c == "r":
    return MAT_TYPE_REAL
  elif c == "i":
    return MAT_TYPE_INTEGER
  elif c == "b":
    return MAT_TYPE_BINARY
  else:
    raise Exception("Unknown matrix type character: " + str(c))


def make_sampler(min: float, max: float, type: str) -> callable:
  '''Makes a sampler for random variables
  
  Args:
    min:    minimum value
    max:    maximum value
    type:   value type
      binary (returns None)
      integer
      real
  '''
  if type == MAT_TYPE_BINARY:
    return None
  elif type == MAT_TYPE_INTEGER:
    return lambda: random.randint(int(min), int(max))
  elif type == MAT_TYPE_REAL:
    return lambda: random.random() * (max - min) + min
  else:
    raise Exception("Unknown matrix type: " + str(type))


# Construct a MatrixMarket matrix
def construct(type: str,
              rows: int,
              cols: int,
              min_nz_per_row: int,
              max_nz_per_row: int,
              min_value: float,
              max_value: float,
              diagonal: bool = False,
              symmetric: bool = True,
              diagonal_dominant: bool = True):
  """Construct COO values

  Set diagonal=True if you want a non-zero diagonal.
  Set symmetric=True is you want symmetric matrix, which will only store the lower triangle here.
  Set diagonal_dominant=True if you want diagonally dominant
  """
  assert cols > max_nz_per_row
  sampler = make_sampler(min_value, max_value, type)
  nnz = 0
  coo = []
  for r in range(1, rows + 1):  # 1-indexed
    r_i = random.randint(
        min(min_nz_per_row, r - 1),
        max_nz_per_row if not symmetric else min(max_nz_per_row, r - 1),
    )  # how many values for this row

    # create columns
    c_seen = []
    while r_i > 0:
      c = random.randint(1, r - 1)
      if c not in c_seen:
        c_seen.append(c)
        nnz += 1
        r_i -= 1
    if diagonal and r not in c_seen:
      c_seen.append(r)
      nnz += 1

    c_seen.sort()  # better to have columns sorted for the row

    if sampler == None:
      coo.extend([str(r) + " " + str(c) for c in c_seen])
    else:
      if diagonal_dominant:
        # diagonal dominance can be done with cols_in_this_row * max_value, which is the upper bound for values in the row
        # or naively, just put |maxvalperrow * maxvalue| + 1
        coo.extend([
            str(r) + " " + str(c) + " " + (str(sampler()) if r != c else str(abs(max_nz_per_row * max_value) + 1))
            for c in c_seen
        ])

      else:
        coo.extend([str(r) + " " + str(c) + " " + str(sampler()) for c in c_seen])
  nnzforname = str((nnz * 2 - rows) if symmetric else nnz)
  name = BASE_NAME + "_" + str(nnzforname) + "_" + RAND_STR(3)
  matHeader = ("%%MatrixMarket matrix coordinate " + type + (" symmetric" if symmetric else " general") + "\n")
  matBanner = "%-------------------------------------------------------------------------------\n"
  matBanner += "% name: " + name + "\n"
  matBanner += "% author: architect.py\n"
  matBanner += "% date: 2021\n"
  matBanner += "% command: python " + FULL_CMD + "\n"
  matBanner += "% seed: " + str(SEED) + "\n"
  matBanner += "%-------------------------------------------------------------------------------\n"
  # print(
  #    "{5} matrix {4} created: {0} x {1} with {2} nnz (nnz/row: {3}). Values are in range ({6}, {7})".format(
  #        rows, cols, nnz, nnz / rows, name, type, min_value, max_value
  #    )
  # )
  return (
      matHeader + matBanner + str(rows) + " " + str(cols) + " " + str(nnz) + "\n" + "\n".join(coo),
      name,
  )


################## MAIN ###################
if __name__ == "__main__":
  FULL_CMD = " ".join(sys.argv[:])
  if len(sys.argv) != 10:
    print(
        "Usage:", sys.argv[0], "\n"
        "$1    Number of rows.\n"
        "$2    Number of columns.\n"
        "$3    Min. nnz per row.\n"
        "$4    Max. nnz per row.\n"
        "$5    Min. value of a nz\n"
        "$6    Max. value of a nz\n"
        "$7    Type of the matrix:\n"
        "  r     for real (doubles)\n"
        "  i     for integer (ints)\n"
        "  b     for pattern (binary)\n"
        "$8    Is matrix symmetric?\n"
        "  s     for YES\n"
        "  -     for NO\n"
        "$9    Is matrix strictly diagonally dominant?\n"
        "  d     for YES\n"
        "  -     for NO\n"
        "\nPlease provide all of these parameters. They are strictly required.")
    exit(-1)

  rows = int(sys.argv[1])
  cols = int(sys.argv[2])
  min_nz_per_row = int(sys.argv[3])
  max_nz_per_row = int(sys.argv[4])
  min_value = float(sys.argv[5])
  max_value = float(sys.argv[6])
  type = CHAR_TO_TYPE(sys.argv[7])
  isSymmetric = sys.argv[8] == "s"
  isDiagDom = sys.argv[9] == "d"
  print("Creating a {7} {8} {6} matrix: {0} x {1} with min {2} - max {3} with values in ({4}, {5})".format(
      rows,
      cols,
      min_nz_per_row,
      max_nz_per_row,
      min_value,
      max_value,
      type,
      "symmetric" if isSymmetric else "",
      "strictly diagonally dominant" if isDiagDom else "",
  ))
  mat, name = construct(
      type,
      rows,
      cols,
      min_nz_per_row,
      max_nz_per_row,
      min_value,
      max_value,
      diagonal=True,
      symmetric=isSymmetric,
      diagonal_dominant=isDiagDom,
  )

  if PRINT:
    print(mat)
  else:
    dir_path = os.path.dirname(os.path.realpath(__file__))  # dir of this file
    fullpath = dir_path + "/../res/architect/" + name + ".mtx"
    text_file = open(fullpath, "w")
    text_file.write(mat)
    text_file.close()
    print("{0} created ({1} MB)".format(fullpath, os.stat(fullpath).st_size / 1e6))
"""
<rows> <cols> <min_nzprow> <max_nzprow> <min_value> <max_value> <type: r | i | b> <symmetric: s | ?> <diagdom d | ?>


python3 scripts/architect.py 54000 54000 10 20 -100 100 r s d (1.675.252 nz) 
python3 scripts/architect.py 5000 5000 100 200 -100 100 r s d (1.476.960 nz)
python3 scripts/architect.py 32000 32000 15 25 -100 100 r s d (1.313.598 nz)
python3 scripts/architect.py 4000 4000 300 400 -2 2 r s d  (2.676.854 nz)
"""
