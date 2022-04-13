from numpy import argsort

###############################################################################
## Run types
###############################################################################
SPMV = "spmv"
JACOBI = "jacobi"
CARDIAC = "cardiac"

# Double
ANAME_DOUBLES_CPU = "FP64 (CPU)"
ANAME_DOUBLES_CUSP = "FP64 (CUSP)"
# Single
ANAME_SINGLES_SR_CUSP = "FP32-S (CUSP)"
ANAME_SINGLES_DR_CUSP = "FP32-D (CUSP)"
# Mixed
ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT = "DD (ES)"
ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE = "DD (ESBASE)"
ANAME_DATADRIVEN_MIXED_ROWWISE_DUAL = "DD (RD)"
ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE = "DD (RC)"
ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT = "DD (RS)"
ANAME_DATADRIVEN_MIXED_ROWWISE_DUAL_A_P = "DD (RD-AP)"
ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE_A_P = "DD (RC-AP)"
ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT_A_P = "DD (RS-AP)"
ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT_A_N = "DD (RS-AN)"
ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT_F_P = "DD (RS-FP)"
ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT_F_N = "DD (RS-FN)"
# Multi-step
ANAME_MULTI_SINGLE_DOUBLE = "FP32-FP64"
ANAME_MULTI_SINGLE_DD_R = "FP32-DD(RC)"
ANAME_MULTI_DD_R_DOUBLE = "DD(RC)-FP64"
ANAME_MULTI_SINGLE_DD_R_DOUBLE = "FP32-DD(RC)-FP64"

# Map names
MAPANAME_XAXIS = {
    ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT: "entry-wise\nsplit",
    ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE: "baseline",
    ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT: 'row-wise\nsplit',
    ANAME_DATADRIVEN_MIXED_ROWWISE_DUAL: 'row-wise\ndual',
    ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE: "row-wise\ncomposite",
    ANAME_DATADRIVEN_MIXED_ROWWISE_DUAL_A_P: 'row-wise\ndual',
    ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE_A_P: "row-wise\ncomposite",
    ANAME_SINGLES_DR_CUSP: "FP32\n(FP64 reduction)",
    ANAME_SINGLES_SR_CUSP: "FP32",
    ANAME_DOUBLES_CUSP: "FP64",
    ANAME_MULTI_DD_R_DOUBLE: "Row-wise Composite\n→ FP64",
    ANAME_MULTI_SINGLE_DD_R_DOUBLE: "FP32 → Row-wise\nComposite → FP64"
}
MAPANAME_LEGEND = {
    ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT: "entry-wise Split",
    ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE: "baseline",
    ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT: 'row-wise split',
    ANAME_DATADRIVEN_MIXED_ROWWISE_DUAL: 'row-wise dual',
    ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE: "row-wise composite",
    ANAME_DATADRIVEN_MIXED_ROWWISE_DUAL_A_P: 'row-wise dual (AP)',
    ANAME_SINGLES_DR_CUSP: "FP32 (FP64 reduction)",
    ANAME_SINGLES_SR_CUSP: "FP32",
    ANAME_DOUBLES_CUSP: "FP64",
    ANAME_MULTI_DD_R_DOUBLE: "2-step",  #: row-wise composite → FP64",
    ANAME_MULTI_SINGLE_DD_R_DOUBLE: "3-step"  #: FP32 → row-wise composite → FP64"
}
###############################################################################
## Colors for split types
## https://matplotlib.org/stable/gallery/color/named_colors.html
###############################################################################
_COLOR_LIGHT_RED = "#EA9999"
_COLOR_LIGHT_GREEN = "#B6D7A8"
_COLOR_LIGHT_BLUE = "#A4C2F4"
_COLOR_LIGHT_ORANGE = "#E8A710"
_COLOR_LIGHT_PINK = "#E57CF7"
COLORS = {
    # Doubles
    ANAME_DOUBLES_CPU: _COLOR_LIGHT_GREEN,
    ANAME_DOUBLES_CUSP: _COLOR_LIGHT_GREEN,
    # Singles
    ANAME_SINGLES_SR_CUSP: _COLOR_LIGHT_RED,
    ANAME_SINGLES_DR_CUSP: 'red',
    # Mixed
    ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT: "gold",
    ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE: "green",
    ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT_A_P: 'cornflowerblue',
    ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT_A_N: 'lightblue',
    ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT_F_P: 'blue',
    ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT_F_N: 'darkblue',
    ANAME_DATADRIVEN_MIXED_ROWWISE_DUAL_A_P: 'darkorchid',
    ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE_A_P: "mediumturquoise",
    ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT: 'cornflowerblue',
    ANAME_DATADRIVEN_MIXED_ROWWISE_DUAL: 'darkorchid',
    ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE: "mediumturquoise",
    # Multi
    ANAME_MULTI_SINGLE_DOUBLE: _COLOR_LIGHT_GREEN,
    ANAME_MULTI_SINGLE_DD_R: _COLOR_LIGHT_BLUE,
    ANAME_MULTI_DD_R_DOUBLE: 'brown',
    ANAME_MULTI_SINGLE_DD_R_DOUBLE: 'orange',
}
###############################################################################
## Hatches for kernel types
## https://matplotlib.org/devdocs/gallery/shapes_and_collections/hatch_style_reference.html
##
## NOTE: Hatches do not show on Ubuntu viewers sometimes,
## but you can export a PDF and look at it in web browser
###############################################################################
_HATCH_CPU = '*'  # star
_HATCH_CUSP = ''  # cirle
_HATCH_ENTRYWISE_SPLIT = '^'  # point
_HATCH_ROWWISE_SPLIT = 'o'  # lines
_HATCH_ROWWISE_DUAL = 's'  # lines both ways (checkered)
_HATCH_COMPOSITE = 'x'  # diagonal lines (diamonds)
_HATCH_TWOSTEP = '++'  # pluses
_HATCH_THREESTEP = 'xx'  # crosses
HATCHES = {
    # Doubles
    ANAME_DOUBLES_CPU: _HATCH_CPU,
    ANAME_DOUBLES_CUSP: _HATCH_CUSP,
    # Singles
    ANAME_SINGLES_SR_CUSP: _HATCH_CUSP,
    ANAME_SINGLES_DR_CUSP: _HATCH_CUSP,
    # Mixed
    ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT: _HATCH_ENTRYWISE_SPLIT,
    ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE: _HATCH_ENTRYWISE_SPLIT,
    ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT: _HATCH_ROWWISE_SPLIT,
    ANAME_DATADRIVEN_MIXED_ROWWISE_DUAL: _HATCH_ROWWISE_DUAL,
    ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE: _HATCH_COMPOSITE,
    ANAME_DATADRIVEN_MIXED_ROWWISE_DUAL_A_P: _HATCH_ROWWISE_DUAL,
    ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE_A_P: _HATCH_COMPOSITE,
    ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT_A_P: _HATCH_ROWWISE_SPLIT,
    ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT_A_N: _HATCH_ROWWISE_SPLIT,
    ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT_F_P: _HATCH_ROWWISE_SPLIT,
    ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT_F_N: _HATCH_ROWWISE_SPLIT,
    # Multi
    ANAME_MULTI_SINGLE_DOUBLE: _HATCH_TWOSTEP,
    ANAME_MULTI_SINGLE_DD_R: _HATCH_TWOSTEP,
    ANAME_MULTI_DD_R_DOUBLE: _HATCH_TWOSTEP,
    ANAME_MULTI_SINGLE_DD_R_DOUBLE: _HATCH_THREESTEP,
}

###############################################################################
## Markers for kernel types
## https://matplotlib.org/stable/api/markers_api.html
###############################################################################
_MARKER_CPU = '*'  # star
_MARKER_CUSP = 'v'  # cirle
_MARKER_ENTRYWISE_SPLIT = '^'  # point
_MARKER_ROWWISE_SPLIT = 'o'  # x
_MARKER_ROWWISE_DUAL = 's'  # x filled
_MARKER_COMPOSITE = 'x'  # diamond
_MARKER_TWOSTEP_FROM_SINGLE = 'v'  # triangle down
_MARKER_TWOSTEP_TO_DOUBLE = '^'  # triangle down
_MARKER_THREESTEP = '>'  # triangle right
MARKERS = {
    # Doubles
    ANAME_DOUBLES_CPU: _MARKER_CPU,
    ANAME_DOUBLES_CUSP: _MARKER_CUSP,
    # Singles
    ANAME_SINGLES_SR_CUSP: _MARKER_CUSP,
    ANAME_SINGLES_DR_CUSP: _MARKER_CUSP,
    # Mixed
    ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT: _MARKER_ENTRYWISE_SPLIT,
    ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE: _MARKER_ENTRYWISE_SPLIT,
    ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT: _MARKER_ROWWISE_SPLIT,
    ANAME_DATADRIVEN_MIXED_ROWWISE_DUAL: _MARKER_ROWWISE_DUAL,
    ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE: _MARKER_COMPOSITE,
    ANAME_DATADRIVEN_MIXED_ROWWISE_DUAL_A_P: _MARKER_ROWWISE_DUAL,
    ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE_A_P: _MARKER_COMPOSITE,
    ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT_A_P: _MARKER_ROWWISE_SPLIT,
    ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT_A_N: _MARKER_ROWWISE_SPLIT,
    ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT_F_P: _MARKER_ROWWISE_SPLIT,
    ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT_F_N: _MARKER_ROWWISE_SPLIT,
    # Two-step
    ANAME_MULTI_SINGLE_DOUBLE: _MARKER_TWOSTEP_FROM_SINGLE,
    ANAME_MULTI_SINGLE_DD_R: _MARKER_TWOSTEP_FROM_SINGLE,
    ANAME_MULTI_DD_R_DOUBLE: _MARKER_TWOSTEP_TO_DOUBLE,
    # Three-step
    ANAME_MULTI_SINGLE_DD_R_DOUBLE: _MARKER_THREESTEP,
}

###############################################################################
## Attribute
###############################################################################
ATTRIB_ITERATION = "iter"
ATTRIB_ERROR = "error"
ATTRIB_GFLOPS = "gflops"
ATTRIB_GBPS = "gbps"
ATTRIB_TIME = "time"
ATTRIB_SPEEDUP = "speedup"
ATTRIB_DELTA = "delta"
ATTRIB_GAMMA = "gamma"
ATTRIB_SPLIT_PERCENTAGE = "percentage"
ATTRIB_INSTANCE = "ins"  # not a member of dict, but just a point in x axis
ATTRIB_NNZ = "nnz"
ATTRIB_AVG_NNZ = "avg_nz_inrow"
ATTRIB_MAX_NNZ = "max_nz_inrow"
ATTRIB_MATRIX = 'mat'

###############################################################################
## Format type of the iterations
###############################################################################
FMT_MAX_ITER = -1  # Show iterations until the max convergence
FMT_ALL = -2  # Show all iterations
FMT_MIN_ITER = -3  # Show iterations until the min convergence

###############################################################################
## Plot selections
###############################################################################
# only focus on splits for SpMV. say that composite is similar to split, and dual is worse on each case.
INTERPET_ALGS = {
    "spmv-perf-profile": [
        ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT,
        #ANAME_DATADRIVEN_MIXED_ROWWISE_DUAL,
        ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE,
        #ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT,
        ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE,
    ],
    "spmv-scatter": [
        ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT, ANAME_SINGLES_DR_CUSP, ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE
    ],
    "jacobi": [
        ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE,
        ANAME_MULTI_DD_R_DOUBLE,
        ANAME_MULTI_SINGLE_DD_R_DOUBLE,
        ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE,
    ],
    "cardiac": [
        ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE,
        ANAME_MULTI_DD_R_DOUBLE,
        ANAME_MULTI_SINGLE_DD_R_DOUBLE,
        ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE,
    ]
}
INTERPET_MATS = {
    # selected 20/32 matrcies in their paper. the remaining 12 have no values in FP32.
    "kahmad": [
        'mark3jac100sc',
        'scagr7-2r',
        'c-73',
        'para-4',
        'rdb968',
        'fs_760_3',
        'oscil_dcop_42',
        'PR02R',
        'celegans_metabolic',
        'cavity01',
        'c-54',
        'freeFlyingRobot_3',
        'bcsstk04',
        'ms2010',
        'bcircuit',
        'oscil_dcop_12',
        'bcsstk03',
        'ASIC_320ks',
        'GD01_a',
        'oscil_dcop_56',
        #'mycielskian18',
        #'halfb',
        #'rgg_n_2_17_s0',
        #'pkustk05',
        #'pkustk02',
        #'coAuthorsDBLP',
        #'GD96_c',
        #'G19',
        #'germany_osm',
        #'nasa2910',
        #'dwt_607',
        #'lshp_265'
    ],
    "jacobi": ['language', 'circuit5M_dc', 'trans5', 'ohne2', 'torsion1', 'minsurfo', 'para-4', 'ASIC_320ks']
}


###############################################################################
###############################################################################
###############################################################################
def prepare_materials(res, type, select, ignore):
  '''Ignore or keep a selection of algorithms.'''
  # also filter out nonetype matrices (perhaps skipped)

  mats = [m for m in list(res.keys()) if res[m]['error'] == None]
  # overridden removals

  algs = list(res[mats[0]][type].keys())

  # select specific algorithms
  if len(select) > 0:
    algs = [x for x in algs if x in select]

  # ignore specific algorithms
  if len(ignore) > 0:
    algs = [x for x in algs if x not in ignore]

  return mats, algs


# filter out these guys from dict
#FILTER_OUT = ["circuit5M","dielFilterV2real","dielFilterV3real","nv2","kron_g500-logn19","kron_g500-logn20","kron_g500-logn21","as-caida","bas1lp","dbic1","dbir1","dbir2","e18","foldoc","GL7d13","GL7d14","GL7d23","GL7d24","IG5-14","IG5-15","IG5-16","IG5-17","IG5-18","kron_g500-logn18","lpl3","nsct","nsir","pltexpa","sctap1-2r","sx-askubuntu","sx-superuser","TF16","TF17","TF18","TF19","wiki-RfA","Wordnet3"]
SORT_SPECIAL_SPLITDIFF = 1
SORT_SPECIAL_SPEEDUPDIFF = 2


def filter_matrices(
    res,
    alg,
    type=SPMV,
    splitalgs=[ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT],
    #remove_selected_matrices=False,
    exclude_edge_splits=False,
    force_reals_only=False,
    edge_split=(0, 100),
    nnz_min=0,
    nnz_max=900000000,
    sort_by=None,
    selection=-1,
    runtime_min=0,
    max_error=None):
  if selection <= 0:
    selection = 9999999  # choose all
  # get all
  mats = [m for m in list(res.keys()) if res[m]['error'] == None]
  # jacobi assert
  if type == JACOBI:
    mats = [m for m in mats if res[m]['info']['rows'] == res[m]['info']['cols']]
  # remove matrices that are in the filter
  #if remove_selected_matrices:
  #  mats = [m for m in mats if m not in FILTER_OUT]
  # get real matrices only (remove integer and pattern types)
  if force_reals_only:
    mats = [m for m in mats if res[m]['info']['mattype'] == 'r']
  # filter size
  mats = [m for m in mats if res[m]['info']['nnz'] >= nnz_min and res[m]['info']['nnz'] <= nnz_max]
  # filter by runtime
  mats = [m for m in mats if res[m][type][ANAME_DOUBLES_CUSP][ATTRIB_TIME] >= runtime_min]
  # get matrix below a certain error
  if max_error != None:
    mats = [m for m in mats if res[m][type][ANAME_DOUBLES_CUSP][ATTRIB_ERROR] <= max_error]
  # ignore 0% and 100% splits
  if exclude_edge_splits:
    for a in splitalgs:
      mats = [
          m for m in mats if (res[m][type][a]["percentage"] * 100.0 > edge_split[0] and res[m][type][a]["percentage"] *
                              100.0 < edge_split[1])
      ]

  # return if no sorting
  if sort_by == None:
    return mats
  else:
    # sort by attribs
    if sort_by == ATTRIB_SPEEDUP:
      attribs = [res[m][type][alg][ATTRIB_SPEEDUP] for m in mats]
    elif sort_by in [ATTRIB_AVG_NNZ, ATTRIB_NNZ, ATTRIB_MAX_NNZ]:
      attribs = [res[m]['info'][sort_by] for m in mats]
    elif sort_by == ATTRIB_ERROR:
      attribs = [res[m][type][alg][ATTRIB_ERROR] for m in mats]
    elif sort_by == ATTRIB_SPLIT_PERCENTAGE:
      attribs = [res[m][type][alg][ATTRIB_SPLIT_PERCENTAGE] for m in mats]
    elif sort_by == SORT_SPECIAL_SPLITDIFF:
      attribs = [
          res[m][type][ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT][ATTRIB_SPLIT_PERCENTAGE] -
          res[m][type][ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT][ATTRIB_SPLIT_PERCENTAGE] for m in mats
      ]
    elif sort_by == SORT_SPECIAL_SPEEDUPDIFF:
      attribs = [
          res[m][type][ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT][ATTRIB_SPEEDUP] -
          res[m][type][ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT][ATTRIB_SPEEDUP] for m in mats
      ]

    # argsort to sort
    sorted_mats = [mats[i] for i in reversed(argsort(attribs))]

    # get the results
    if selection < 1:
      mats = sorted_mats[:int(len(sorted_mats) * selection)]
    else:
      mats = sorted_mats[:min(len(sorted_mats), int(selection))]

    return list(reversed(mats))


def print_run_info(res):
  info = res[list(res.keys())[0]]['info']
  print("Run info:\n"
        "\tSpMV iterations: {0}\n"
        "\tJacobi iterations: {1}\n"
        #\tCardiac iterations: {2}\n"
        "\tSplit Percentage: {3}\n"
        "\tSplit Shrink Factor: {4}\n".format(
            info['spmv_iter'],
            info['jacobi_iter'],
            info['cardiac_iter'] if 'cardiac_iter' in info else '---',
            info['dd_percentage'],
            info['dd_shrink'] if 'dd_shrink' in info else '---',
        ))

  # print([res[m][type][alg][attrib] for m in mats])


if __name__ == "__main__":
  pass