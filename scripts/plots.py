import json
import matplotlib.pyplot as plt
import os
from plotspecial.performance_plot import performance_profile

# utilities
from utils import *

# generic plots
from plottype.scatter import scatter
from plottype.line import line
from plottype.heatmap import heatmap
from plottype.bar import bar
from plottype.aggregatebar import aggregatebar
from plottype.boxplot import boxplot
from plottype.tuplebar import tuplebar

# special plots
from plotspecial.jacobi_steps_bar import jacobi_steps_bar
from plotspecial.split_comparison import split_comparison
from plotspecial.jacobi_table import jacobi_table
from plotspecial.cardiac_table import cardiac_table
from plotspecial.histogram_speedups import histogram_speedups


def _update_labels(ax, title, xlabel, ylabel, fontsize='x-large'):
  if title != None:
    ax.set_title(title, fontsize=fontsize)
  if xlabel != None:
    ax.set_xlabel(xlabel, fontsize=fontsize)
  if ylabel != None:
    ax.set_ylabel(ylabel, fontsize=fontsize)


class Plots:

  def __init__(self, res, save_to_path=".", save_dict_as=None):
    """Constructor.

    Args:
      save_to_path: The path where plots will be saved. Default = '.'
      save_dict_as: Save the results on file with the given filename. Default is None, which does not save anything. 
    """
    self.res = res
    self.save_to_path = save_to_path
    if save_to_path != None and not os.path.exists(save_to_path):
      os.makedirs(save_to_path)

    # save the results as dict
    if save_to_path != None and save_dict_as != None:
      with open(save_to_path + "/" + save_dict_as, "w") as f:
        json.dump(res, f)
        print("info results saved as JSON at:\n", save_to_path + "/" + save_dict_as)

  # TODO: can have a single function for all plots

  ###############################################################################
  ## Generic Plots
  ###############################################################################
  def Heatmap(self,
              type,
              attrib,
              select=[],
              ignore=[],
              save_as=None,
              legend=False,
              title=None,
              xlabel=None,
              ylabel=None):
    assert (attrib in [ATTRIB_SPLIT_PERCENTAGE, ATTRIB_ERROR, ATTRIB_DELTA, ATTRIB_ITERATION, ATTRIB_SPEEDUP])

    mats, algs = prepare_materials(self.res, type, select, ignore)
    if len(algs) == 0:
      print("No algorithms for this selection of Heatmap!")
      return
    fig, ax = plt.subplots()
    heatmap(ax, self.res, mats, algs, type, attrib, legend)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    _update_labels(ax, title, xlabel, ylabel)
    plt.tight_layout()
    plt.show()
    if save_as != None:
      fig.savefig(self.save_to_path + "/" + save_as)

  def Bar(self,
          type,
          attrib,
          select=[],
          ignore=[],
          save_as=None,
          barwidth=0.15,
          fontsize=7,
          legend=False,
          forceMats=[],
          figsize=None,
          title=None,
          xlabel=None,
          ylabel=None,
          mapaname=True):

    assert (attrib in [ATTRIB_DELTA, ATTRIB_ERROR, ATTRIB_TIME, ATTRIB_GFLOPS, ATTRIB_SPEEDUP])
    mats, algs = prepare_materials(self.res, type, select, ignore)
    if len(algs) == 0:
      print("No algorithms for this selection of Bar!")
      return
    if len(forceMats) > 0:
      mats = forceMats
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    bar(ax, self.res, mats, algs, type, attrib, barwidth, fontsize, legend, mapaname)
    _update_labels(ax, title, xlabel, ylabel, fontsize=fontsize)
    plt.tight_layout()
    plt.show()
    if save_as != None:
      fig.savefig(self.save_to_path + "/" + save_as)

  def BoxPlot(self,
              type,
              attrib,
              select=[],
              ignore=[],
              save_as=None,
              barwidth=0.15,
              fontsize=7,
              legend=False,
              forceMats=[],
              outliers=True,
              title=None,
              xlabel=None,
              ylabel=None):

    assert (attrib in [ATTRIB_DELTA, ATTRIB_ERROR, ATTRIB_TIME, ATTRIB_GFLOPS, ATTRIB_SPEEDUP, ATTRIB_GAMMA])
    mats, algs = prepare_materials(self.res, type, select, ignore)
    if len(algs) == 0:
      print("No algorithms for this selection of Bar!")
      return
    if len(forceMats) > 0:
      mats = forceMats
    fig, ax = plt.subplots()
    boxplot(ax, self.res, mats, algs, type, attrib, barwidth, fontsize, legend, outliers)
    _update_labels(ax, title, xlabel, ylabel)
    plt.tight_layout()
    plt.show()
    if save_as != None:
      fig.savefig(self.save_to_path + "/" + save_as)

  def AggregateBar(self,
                   type,
                   attrib,
                   select=[],
                   ignore=[],
                   save_as=None,
                   barwidth=0.1,
                   fontsize=9,
                   legend=False,
                   forceMats=[],
                   title=None,
                   xlabel=None,
                   ylabel=None,
                   usehatches=True,
                   mapaname=False):

    assert (attrib in [ATTRIB_DELTA, ATTRIB_ERROR, ATTRIB_TIME, ATTRIB_GFLOPS, ATTRIB_SPEEDUP])
    mats, algs = prepare_materials(self.res, type, select, ignore)
    if len(forceMats) > 0:
      mats = forceMats
    if len(algs) == 0:
      print("No algorithms for this selection of Bar!")
      return
    fig, ax = plt.subplots()
    aggregatebar(ax, self.res, mats, algs, type, attrib, barwidth, fontsize, legend, usehatches, mapaname)
    _update_labels(ax, title, xlabel, ylabel)
    plt.show()
    if save_as != None:
      fig.savefig(self.save_to_path + "/" + save_as)

  def Scatter(self,
              type,
              xattrib,
              yattrib,
              select=[],
              ignore=[],
              save_as=None,
              legend=False,
              forceMats=[],
              fontsize='large',
              title=None,
              xlabel=None,
              ylabel=None,
              usemarkers=False,
              mapaname=False):

    assert (xattrib in [
        ATTRIB_ERROR, ATTRIB_TIME, ATTRIB_GFLOPS, ATTRIB_SPEEDUP, ATTRIB_SPLIT_PERCENTAGE, ATTRIB_INSTANCE, ATTRIB_NNZ,
        ATTRIB_AVG_NNZ, ATTRIB_MATRIX, ATTRIB_GAMMA
    ])
    assert (yattrib in [
        ATTRIB_ERROR, ATTRIB_TIME, ATTRIB_GFLOPS, ATTRIB_SPEEDUP, ATTRIB_SPLIT_PERCENTAGE, ATTRIB_INSTANCE, ATTRIB_NNZ,
        ATTRIB_AVG_NNZ, ATTRIB_GAMMA
    ])

    # plots
    mats, algs = prepare_materials(self.res, type, select, ignore)
    if len(algs) == 0:
      print("No algorithms for this selection of Density!")
      return
    if len(forceMats) > 0:
      mats = forceMats
    fig, ax = plt.subplots()
    scatter(ax, self.res, mats, algs, type, xattrib, yattrib, legend, usemarkers, mapaname, fontsize)
    _update_labels(ax, title, xlabel, ylabel, fontsize=fontsize)
    plt.tight_layout()
    plt.show()

    # save
    if save_as != None:
      fig.savefig(self.save_to_path + "/" + save_as)

  def Line(self,
           type,
           xattrib,
           yattrib,
           select=[],
           ignore=[],
           save_as=None,
           legend=False,
           forceMats=[],
           title=None,
           xlabel=None,
           ylabel=None,
           usemarkers=False,
           mapaname=False):

    assert (xattrib in [
        ATTRIB_ERROR, ATTRIB_TIME, ATTRIB_GFLOPS, ATTRIB_SPEEDUP, ATTRIB_SPLIT_PERCENTAGE, ATTRIB_INSTANCE, ATTRIB_NNZ,
        ATTRIB_AVG_NNZ, ATTRIB_MATRIX, ATTRIB_GAMMA
    ])
    assert (yattrib in [
        ATTRIB_ERROR, ATTRIB_TIME, ATTRIB_GFLOPS, ATTRIB_SPEEDUP, ATTRIB_SPLIT_PERCENTAGE, ATTRIB_INSTANCE, ATTRIB_NNZ,
        ATTRIB_AVG_NNZ, ATTRIB_GAMMA
    ])

    # plots
    mats, algs = prepare_materials(self.res, type, select, ignore)
    if len(algs) == 0:
      print("No algorithms for this selection of Density!")
      return
    if len(forceMats) > 0:
      mats = forceMats
    fig, ax = plt.subplots()
    line(ax, self.res, mats, algs, type, xattrib, yattrib, legend, usemarkers, mapaname)
    _update_labels(ax, title, xlabel, ylabel)
    plt.tight_layout()
    plt.show()

    # save
    if save_as != None:
      fig.savefig(self.save_to_path + "/" + save_as)

  def Split_Comparison(self, save_as=None, forceMats=[], title=None, xlabel=None, ylabel=None):
    mats, algs = prepare_materials(self.res, SPMV, [], [])
    if len(forceMats) > 0:
      mats = forceMats
    fig = plt.figure(figsize=None)
    ax = fig.add_subplot(111)
    split_comparison(ax, self.res, mats, algs)
    _update_labels(ax, title, xlabel, ylabel)
    plt.tight_layout()
    plt.show()
    if save_as != None:
      fig.savefig(self.save_to_path + "/" + save_as)

  def Histogram_Speedups(self,
                         type,
                         select=[],
                         ignore=[],
                         save_as=None,
                         forceMats=[],
                         figsize=(15, 10),
                         title=None,
                         xlabel=None,
                         ylabel=None,
                         legend=False):
    mats, algs = prepare_materials(self.res, type, select, ignore)
    if len(forceMats) > 0:
      mats = forceMats
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    histogram_speedups(ax, self.res, mats, algs, type, legend)
    _update_labels(ax, title, xlabel, ylabel, 'x-large')
    plt.tight_layout()
    plt.grid()
    plt.show()
    if save_as != None:
      fig.savefig(self.save_to_path + "/" + save_as)

  def Performance_Profile(self,
                          type,
                          attrib,
                          select=[],
                          ignore=[],
                          save_as=None,
                          forceMats=[],
                          fontsize='large',
                          xlim=1.3,
                          title=None,
                          xlabel=None,
                          ylabel=None,
                          mapaname=True):
    mats, algs = prepare_materials(self.res, type, select, ignore)
    if len(forceMats) > 0:
      mats = forceMats
    fig = plt.figure()
    ax = fig.add_subplot(111)
    performance_profile(ax, self.res, mats, algs, type, attrib, mapaname, xlim, fontsize)
    _update_labels(ax, title, xlabel, ylabel, fontsize=fontsize)
    plt.tight_layout()
    plt.show()
    if save_as != None:
      fig.savefig(self.save_to_path + "/" + save_as)

  def TupleBar(self,
               attrib,
               save_as=None,
               barwidth=0.15,
               fontsize=8,
               legend=False,
               forceMats=[],
               algtuples=None,
               title=None,
               xlabel=None,
               ylabel=None):
    assert (attrib in [ATTRIB_SPEEDUP, ATTRIB_TIME, ATTRIB_GFLOPS, ATTRIB_ERROR])

    fig, ax = plt.subplots()
    type = SPMV
    mats, _ = prepare_materials(self.res, type, [], [])
    if len(forceMats) > 0:
      mats = forceMats
    tuplebar(ax, self.res, mats, type, attrib, barwidth, fontsize, legend, algtuples)
    _update_labels(ax, title, xlabel, ylabel)
    plt.show()

    # save
    if save_as != None:
      fig.savefig(self.save_to_path + "/" + save_as)

  ###############################################################################
  ## Special Plots
  ###############################################################################

  def Jacobi_Steps_Bar(
      self,
      select=[],
      ignore=[],
      save_as=None,
      barwidth=0.15,
      fontsize=7,
      legend=False,
      forceMats=[],
      title=None,
  ):

    # plots
    mats, algs = prepare_materials(self.res, JACOBI, select, ignore)
    if len(algs) == 0:
      print("No algorithms for this selection of Bar!")
      return
    if len(forceMats) > 0:
      mats = forceMats
    fig, ax = plt.subplots()
    print(algs)
    jacobi_steps_bar(ax, self.res, mats, algs, barwidth, fontsize, legend)
    if title != None:
      ax.set_title(title)
    plt.show()
    # save
    if save_as != None:
      fig.savefig(self.save_to_path + "/" + save_as)

  ###############################################################################
  ## Extras
  ###############################################################################
  def Jacobi_Table(self, select=[], ignore=[], forceMats=[], save_as=None):
    mats, algs = prepare_materials(self.res, JACOBI, select, ignore)
    if len(algs) == 0:
      print("No algorithms for this selection of Bar!")
      return
    if len(forceMats) > 0:
      mats = forceMats
    jacobi_table(self.res, mats, save_as, self.save_to_path)

  def Cardiac_Table(self, select=[], ignore=[], forceMats=[], save_as=None):
    mats, algs = prepare_materials(self.res, CARDIAC, select, ignore)
    if len(algs) == 0:
      print("No algorithms for this selection of Bar!")
      return
    if len(forceMats) > 0:
      mats = forceMats
    cardiac_table(self.res, mats, save_as, self.save_to_path)


if __name__ == "__main__":

  def combineDicts():
    # combine two dicts into one
    filename_in1 = "spmv-100k_1m-opt1-p99-2k"
    filename_in2 = "spmv-1m_40m-opt1-p99-2k"
    filename_output = "spmv-100k_40m-opt1-p99-2k"
    out: dict = {}
    common_path = os.path.dirname(__file__) + '/../evaluations'
    # input 1
    print("Reading input 1 from:", common_path + "/in/" + filename_in1)
    f_in1 = open(common_path + "/in/" + filename_in1 + ".json", 'r')
    in_1: dict = json.load(f_in1)
    f_in1.close()
    for k in in_1.keys():
      out[k] = in_1[k]

    # input 2
    print("Reading input 2 from:", common_path + "/in/" + filename_in2)
    f_in2 = open(common_path + "/in/" + filename_in2 + ".json", 'r')
    in_2: dict = json.load(f_in2)
    f_in2.close()
    for k in in_2.keys():
      out[k] = in_2[k]

    # output
    f_out = open(common_path + "/out/" + filename_output + ".json", 'w')
    json.dump(out, f_out)
    f_out.close()
    print("Saved output to:", common_path + "/out/" + filename_output + ".json")

  def findMatrixDifferences():
    filename_in1 = "paper/spmv-100k_40m-opt1-p99-2k-simula"
    filename_in2 = "spmv-ellr-simula"
    common_path = os.path.dirname(__file__) + '/../evaluations'

    # input 1
    print("Reading input 1 from:", common_path + "/in/" + filename_in1)
    f_in1 = open(common_path + "/in/" + filename_in1 + ".json", 'r')
    in_1: dict = json.load(f_in1)
    mats_in_1 = filter_matrices(
        in_1,
        ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE,
        splitalgs=[ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT],
        exclude_edge_splits=True,  # only choose when splitalg has > 0% and < 100% percentage singles
        edge_split=(10, 111),  # ignore <10% and >90% splits
        nnz_min=100000,  # matrix has at least 100k nnz
        nnz_max=40000000,  # matrix has at most 40m nnz 
        force_reals_only=True,  # only real valued matrices
        runtime_min=100,  # at least 100 milliseconds runtime
        sort_by=ATTRIB_ERROR)  # sort by error
    f_in1.close()

    # input 2
    print("Reading input 2 from:", common_path + "/in/" + filename_in2)
    f_in2 = open(common_path + "/in/" + filename_in2 + ".json", 'r')
    in_2: dict = json.load(f_in2)
    mats_in_2 = filter_matrices(
        in_2,
        ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE,
        splitalgs=[ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT],
        exclude_edge_splits=True,  # only choose when splitalg has > 0% and < 100% percentage singles
        edge_split=(10, 111),  # ignore <10% and >90% splits
        nnz_min=100000,  # matrix has at least 100k nnz
        nnz_max=40000000,  # matrix has at most 40m nnz 
        force_reals_only=True,  # only real valued matrices
        runtime_min=100,  # at least 100 milliseconds runtime
        sort_by=ATTRIB_ERROR)  # sort by error
    f_in2.close()

    # compare
    missings = []
    for k in mats_in_1:
      if k not in mats_in_2:
        missings.append(k)

    print("{0} missing in total.".format(len(missings)))
    for k in missings:
      print("{0:<20} missing: {1}".format(k, in_2[k]['error']))

  findMatrixDifferences()