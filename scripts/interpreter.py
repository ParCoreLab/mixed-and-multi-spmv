import pathlib
from json import load as JSONLOAD
from utils import *
from plots import Plots
from exporter import export_results_csv
from prints import print_aggr_speedups

#_FILENAME = "paper/spmv-ellr-simula"
_FILENAME = "paper/cardiac-simula"
#_FILENAME = "paper/spmv-100k_40m-opt1-p99-2k-simula"
#_FILENAME = "paper/spmv-all-opt1-p99-2k-simula"
#_FILENAME = "paper/jacobi-100k_40m-opt1-p99-2k-simula"
_SAVE = True
if __name__ == "__main__":
  has_prefix = lambda s, pf: s[:len(pf)] == pf
  # setup path
  input_relative_path = "evaluations/in/" + _FILENAME + ".json"
  input_path = str(pathlib.Path(__file__).parent.resolve()) + "/../" + input_relative_path
  save_path = str(pathlib.Path(__file__).parent.resolve()) + "/../img"

  with open(input_path) as f:
    res = JSONLOAD(f)
    P = Plots(res, save_to_path=save_path)
    print_run_info(P.res)

    # parse filename
    filename = _FILENAME.split('/')[-1]
    print(filename)

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    if has_prefix(filename, "spmv-ellr"):
      mats = filter_matrices(
          P.res,
          ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE,
          splitalgs=[ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT],
          exclude_edge_splits=True,  # only choose when splitalg has > 0% and < 100% percentage singles
          edge_split=(10, 111),  # ignore <10% and >90% splits
          nnz_min=100000,  # matrix has at least 100k nnz
          nnz_max=40000000,  # matrix has at most 40m nnz 
          force_reals_only=True,  # only real valued matrices
          runtime_min=100,  # at least 100 milliseconds runtime
          sort_by=ATTRIB_ERROR)  # sort by error
      #mats = _KAHMAD_MATS
      print("{0} / {1} matrices used for SpMV.".format(len(mats), len(P.res)))

      print_aggr_speedups(P.res, mats, ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT, SPMV)
      print_aggr_speedups(P.res, mats, ANAME_SINGLES_DR_CUSP, SPMV)

      ## CSV EXPORT ##
      if _SAVE:
        export_results_csv(P.res, mats, SPMV, filename + '.csv')

      ## RUNTIMES ##
      P.Performance_Profile(SPMV,
                            ATTRIB_TIME,
                            forceMats=mats,
                            select=INTERPET_ALGS['spmv-perf-profile'],
                            xlim=1.35,
                            title="",
                            xlabel="SpMV Speedup relative to the best",
                            ylabel="Fraction of test instances",
                            mapaname=True,
                            fontsize=18,
                            save_as='spmv-ellr-performance.pdf' if _SAVE else None)

      ## ERRORS ##
      # can give title r"SpMV $||Ax_{64} - Ax'||_2/||Ax_{64}||_2$" for latex
      #P.Scatter(SPMV, ATTRIB_INSTANCE, ATTRIB_ERROR, forceMats=mats, select=INTERPET_ALGS['spmv-scatter'],
      #  title="", xlabel="Matrices sorted w.r.t baseline", ylabel="SpMV Relative Residual", legend=True, mapaname=True,
      #  save_as= 'spmv-ellr-errors.pdf' if _SAVE else None)

    elif has_prefix(filename, "spmv"):
      mats = filter_matrices(
          P.res,
          ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE,
          splitalgs=[ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT],
          exclude_edge_splits=True,  # only choose when splitalg has > 0% and < 100% percentage singles
          edge_split=(10, 111),  # ignore <10% and >90% splits
          nnz_min=100000,  # matrix has at least 100k nnz
          nnz_max=40000000,  # matrix has at most 40m nnz 
          force_reals_only=True,  # only real valued matrices
          runtime_min=100,  # at least 100 milliseconds runtime
          sort_by=ATTRIB_ERROR)  # sort by error
      #mats = _KAHMAD_MATS
      print("{0} / {1} matrices used for SpMV.".format(len(mats), len(P.res)))

      print_aggr_speedups(P.res, mats, ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT, SPMV)
      print_aggr_speedups(P.res, mats, ANAME_SINGLES_DR_CUSP, SPMV)

      ## CSV EXPORT ##
      if _SAVE:
        export_results_csv(P.res, mats, SPMV, filename + '.csv')

      ## RUNTIMES ##
      P.Performance_Profile(SPMV,
                            ATTRIB_TIME,
                            forceMats=mats,
                            select=INTERPET_ALGS['spmv-perf-profile'],
                            xlim=1.35,
                            title="",
                            xlabel="SpMV Speedup relative to the best",
                            ylabel="Fraction of test instances",
                            mapaname=True,
                            fontsize=18,
                            save_as='spmv-performance.pdf' if _SAVE else None)

      ## ERRORS ##
      # can give title r"SpMV $||Ax_{64} - Ax'||_2/||Ax_{64}||_2$" for latex
      P.Scatter(SPMV,
                ATTRIB_INSTANCE,
                ATTRIB_ERROR,
                forceMats=mats,
                select=INTERPET_ALGS['spmv-scatter'],
                title="",
                xlabel="Matrices sorted w.r.t baseline",
                ylabel="SpMV Relative Residual",
                legend=True,
                mapaname=True,
                fontsize=18,
                save_as='spmv-errors.pdf' if _SAVE else None)

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    elif has_prefix(filename, "cardiac"):
      mats = filter_matrices(
          P.res,
          ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE,
          type=CARDIAC,
          splitalgs=[ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT],
          exclude_edge_splits=True,  # only choose when splitalg has > 0% and < 100% percentage singles
          edge_split=(10, 111),  # ignore <10% and >90% splits
          nnz_min=100000,  # matrix has at least 100k nnz 
          force_reals_only=True,  # only real valued matrices
          runtime_min=100,  # at least 100 milliseconds runtime
          sort_by=ATTRIB_NNZ)  # sort by nnz
      print("{0} / {1} matrices used for Cardiac.".format(len(mats), len(P.res)))

      ## CSV EXPORT ##
      ## export_results_csv(P.res, mats, CARDIAC, filename+'.csv')

      print_aggr_speedups(P.res, mats, ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT, CARDIAC)

      P.Bar(CARDIAC,
            ATTRIB_SPEEDUP,
            select=INTERPET_ALGS["cardiac"],
            forceMats=mats,
            figsize=(9, 5),
            barwidth=0.16,
            fontsize=19.5,
            legend=True,
            mapaname=True,
            title="",
            xlabel="",
            ylabel='Speedup w.r.t FP64 Cardiac',
            save_as='cardiac-speedups.pdf' if _SAVE else None)

      #P.Bar(CARDIAC, ATTRIB_ERROR, select=INTERPET_ALGS["cardiac"]+[ANAME_SINGLES_DR_CUSP], forceMats=mats, figsize=(12, 7),
      #  barwidth=0.16, fontsize=8.5, legend=True, mapaname=True,
      #  title="", xlabel="", ylabel='Error',
      #  save_as='cardiac-errors.pdf' if _SAVE else None)

      P.Cardiac_Table(forceMats=mats, save_as='cardiac-table.tex' if _SAVE else None)

    ###################################################################################################################
    ###################################################################################################################
    ###################################################################################################################
    elif has_prefix(filename, "jacobi"):
      #print(res[list(res.keys())[1]])
      # selected matrices
      mats = filter_matrices(
          P.res,
          ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT,
          type=JACOBI,
          splitalgs=[ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT],
          exclude_edge_splits=True,  # only choose when splitalg has > 0% and < 100% percentage singles
          edge_split=(0, 111),  # ignore <10% and >90% splits  
          #runtime_min=100, # at least 100 milliseconds runtime
          max_error=1e-8,  # maximum allowed error in FP64 
      )
      # special jacobi mats
      mats = INTERPET_MATS["jacobi"]

      print_aggr_speedups(P.res, mats, ANAME_MULTI_SINGLE_DD_R_DOUBLE, JACOBI)

      # sort by nnz
      sorted_mats = [mats[i] for i in argsort([P.res[m]['info'][ATTRIB_NNZ] for m in mats])]
      mats = sorted_mats
      print("{0} / {1} matrices used for Jacobi (Speedups).".format(len(mats), len(P.res)))

      # export csv
      if _SAVE:
        export_results_csv(res, mats, JACOBI, filename + '.csv')

      # plots
      #P.Scatter(JACOBI, ATTRIB_MATRIX, ATTRIB_SPEEDUP, select=_jacobialgs, forceMats=mats, legend=True,
      #  title="Jacobi Speedups w.r.t FP64 Jacobi Solver", xlabel="", ylabel='Speedup', mapaname=True,
      #  save_as='jacobi-speedups-scatter.pdf' if _SAVE else None)

      P.Bar(JACOBI,
            ATTRIB_SPEEDUP,
            select=INTERPET_ALGS["jacobi"],
            forceMats=mats,
            figsize=(12, 5),
            barwidth=0.16,
            fontsize=19.5,
            legend=True,
            mapaname=True,
            title="",
            xlabel="",
            ylabel='Speedup w.r.t FP64 Jacobi',
            save_as='jacobi-speedups.pdf' if _SAVE else None)

      #P.Scatter(JACOBI, ATTRIB_MATRIX, ATTRIB_ERROR, select=_jacobialgs, forceMats=mats, legend=True,
      #  title="Jacobi Relative Residuals", xlabel="", ylabel='log(Relative Residual)', mapaname=True, usemarkers=True,
      #  save_as='jacobi-errors.pdf' if _SAVE else None)

      P.Jacobi_Table(forceMats=mats, save_as='jacobi-table.tex' if _SAVE else None)
