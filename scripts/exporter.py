from utils import *
import os


def export_matrices_test(mats, save_as):
  lines = '\n'.join(mats)
  path = os.path.dirname(__file__) + '/../evaluations/' + save_as
  f = open(path, 'w')
  f.write(lines)
  f.close()


def export_results_csv(res, mats, type, save_as):
  lines = ','.join([
      'matrix',  # Matrix Name
      'nnz',  # Number of Non-Zeros
      'rows',  # Number of Rows
      'cols',  # Number of Columns
      'sym',  # Is a symmetric matrix?
      'type',  # Matrix Type (r: real, i: integer, p: pattern)
      'range',  # Range chosen dynamically
      'empty-rows',  # Number of Empty Rows
      'avg-nz-row',  # Average nnz Per Row
      'min-nz-row',  # Min nnz in a row
      'max-nz-row',  # Max nnz in a row
      'ent-base-singles',  # Entrywise baseline chosen FP32 value count
      'ent-base-doubles',  # Entrywise baseline chosen FP64 value count
      'ent-base-prc',  # Entrywise baseline chosen FP32 value percentage
      'ent-singles',  # Entrywise chosen FP32 value count
      'ent-doubles',  # Entrywise chosen FP64 value count
      'ent-prc',  # Entrywise chosen FP32 value percentage
      'row-singles',  # Rowwise chosen FP32 value count
      'row-doubles',  # Rowwise chosen FP64 value count
      'row-prc',  # Rowwise chosen FP32 value percentage
      'fp64-rt',  # FP64 SpMV runtime in (ms)
      'fp32(64)-rt',  # FP32 SpMV (with FP64 reduction) runtime in (ms) (averaged over 500)
      'ent-base-rt',  # Entrywise Split Baseline runtime in (ms) (averaged over 500)
      'ent-split-rt',  # Entrywise Split runtime in (ms) (averaged over 500)
      'row-split-rt',  # Rowwise Split runtime in (ms) (averaged over 500)
      'row-comp-rt',  # Rowwise Composite runtime in (ms) (averaged over 500)
      'row-dual-rt',  # Rowwise Dual runtime in (ms) (averaged over 500)
      'fp32(64)-spup',  # FP32 SpMV (with FP64 reduction) speedup w.r.t FP64
      'ent-base-spup',  # Entrywise Split Baseline speedup w.r.t FP64
      'ent-split-spup',  # Entrywise Split speedup w.r.t FP64
      'row-split-spup',  # Rowwise Split speedup w.r.t FP64
      'row-comp-spup',  # Rowwise Composite speedup w.r.t FP64
      'row-dual-spup',  # Rowwise Dual speedup w.r.t FP64
      'fp64-relerr',  # FP64 SpMV runtime in (ms)
      'fp32(64)-relerr',  # FP32 SpMV (with FP64 reduction) relative error 
      'ent-base-relerr',  # Entrywise Split relative error
      'ent-split-relerr',  # Entrywise Split relative error 
      'row-split-relerr',  # Rowwise Split relative error 
      'row-comp-relerr',  # Rowwise Composite relative error 
      'row-dual-relerr'  # Rowwise Dual relative error
  ])
  if type == SPMV:
    lines += ',' + ','.join([
        'fp32(32)-rt',  # FP32 SpMV (with FP32 reduction) runtime in (ms) (averaged over 500)
        'fp32(32)-spup',  # FP32 SpMV (with FP32 reduction) speedup w.r.t FP64
        'fp32(32)-relerr',  # FP32 SpMV (with FP32 reduction) relative error w.r.t FP64 (at 500th iter)
    ])
  for m in mats:
    lines += '\n' + ','.join([
        m,
        str(res[m]['info']['nnz']),
        str(res[m]['info']['rows']),
        str(res[m]['info']['cols']),
        str(1 if res[m]['info']['is_symmetric'] else 0),
        str(res[m]['info']['mattype']),
        str(res[m]['info']['dd_range']),
        str(res[m]['info']['empty_rows']),
        str(res[m]['info']['avg_nz_inrow']),
        str(res[m]['info']['min_nz_inrow']),
        str(res[m]['info']['max_nz_inrow']),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE]['singles']),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE]['doubles']),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE]['percentage'] * 100.0),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT]['singles']),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT]['doubles']),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT]['percentage'] * 100.0),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT]['singles']),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT]['doubles']),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT]['percentage'] * 100.0),
        str(res[m][type][ANAME_DOUBLES_CUSP][ATTRIB_TIME]),
        str(res[m][type][ANAME_SINGLES_DR_CUSP][ATTRIB_TIME]),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE][ATTRIB_TIME]),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT][ATTRIB_TIME]),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT][ATTRIB_TIME]),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE][ATTRIB_TIME]),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ROWWISE_DUAL][ATTRIB_TIME]),
        str(res[m][type][ANAME_SINGLES_DR_CUSP][ATTRIB_SPEEDUP]),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE][ATTRIB_SPEEDUP]),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT][ATTRIB_SPEEDUP]),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT][ATTRIB_SPEEDUP]),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE][ATTRIB_SPEEDUP]),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ROWWISE_DUAL][ATTRIB_SPEEDUP]),
        str(res[m][type][ANAME_DOUBLES_CUSP][ATTRIB_ERROR]),
        str(res[m][type][ANAME_SINGLES_DR_CUSP][ATTRIB_ERROR]),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT_BASELINE][ATTRIB_ERROR]),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ENTRYWISE_SPLIT][ATTRIB_ERROR]),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ROWWISE_SPLIT][ATTRIB_ERROR]),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ROWWISE_COMPOSITE][ATTRIB_ERROR]),
        str(res[m][type][ANAME_DATADRIVEN_MIXED_ROWWISE_DUAL][ATTRIB_ERROR])
    ])
    if type == SPMV:
      lines += ',' + ','.join([
          str(res[m][type][ANAME_SINGLES_SR_CUSP][ATTRIB_TIME]),
          str(res[m][type][ANAME_SINGLES_SR_CUSP][ATTRIB_SPEEDUP]),
          str(res[m][type][ANAME_SINGLES_SR_CUSP][ATTRIB_ERROR])
      ])

  path = os.path.dirname(__file__) + '/../evaluations/dict/' + save_as
  f = open(path, 'w')
  f.write(lines)
  f.close()
  print("Saved to", path)
