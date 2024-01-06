# -------- Backwards Compatibility ---------
from __future__ import print_function
from __future__ import division
# --------------- Utilties -----------------
from shutil import copy2
from shutil import rmtree
import os
import tarfile
import numpy as np
# ----------- Relative Imports -------------
from .utilities.mpi_utils import *
from .strength.fam_strength import famStrength
from .strength.contour import famContour
from .fortran.pnfam_run import pnfamRun
# ------------------------------------------

__version__ = u'2.0.0'
__date__    = u'2019-08-28'


#-------------------------------------------------------------------------------
def initialize_fam_contour(mgr, setts, ctr_type, beta_type, hfb_gs):

    # Instantiate and apply overrides
    ctr_main = famContour(ctr_type, setts[u'ctr'])

    # Change the INTERVAL according to hfb_gs (default pynfam behavior)
    shift = 0.0
    for suffix in [u'_prot', u'_neut']:
        try:
            shift += setts[u'fam'][u'energy_shift'+suffix]
        except KeyError:
            pass
    ctr_main.setHfbInterval(hfb_gs, beta_type, shift=shift)

    # Override the INTERVAL if supplied, otherwise this is redundant/has no effect
    ctr_main.updateSettings(setts[u'ctr'])

    if ctr_main.energy_min >= ctr_main.energy_max:
        #**** Check for Errors ****#
        msg = [u"Invalid EQRPA interval. Skipping FAM calculation.",
               u"(If using HFB auto-interval, this means Q<=0. To calculate",
               u"strength anyway, manually specify a valid interval.)"]
        pynfam_warn(msg, mgr.paths.calclabel)
        return None # FAM errors do not break the dripline
        #**************************#

    return ctr_main

#-------------------------------------------------------------------------------
def initialize_fam_calc(mgr, setts, fam_ops, ctr_main, hfb_gs):

    all_strengths, all_fam_unf, all_fam_fin = {}, [], []
    fam_unf, fam_fin = [], []
    for d in fam_ops:
        strength = famStrength(d[u'op'], d[u'k'], ctr_main, hfb_gs.nucleus)
        fam_unf, fam_fin = mgr.getFamState(strength, setts[u'fam'])

        all_strengths[d[u'name']] = strength
        all_fam_fin += fam_fin
        all_fam_unf += fam_unf

    return all_strengths, (all_fam_unf, all_fam_fin)

#-------------------------------------------------------------------------------
def run_fam_calc(comm, mgr, stdout, all_fam_unf, all_fam_fin):

    finished_pts, err_run, err_msg = runtasks_master(all_fam_unf, comm, stdout)

    #**** Check for Errors ****#
    msg = [pnfamRun.file_exenompi+u" encountered an error.",err_msg]
    if pynfam_warn(msg, mgr.paths.calclabel, err_run):
        return None # FAM errors do not break the dripline
    #**************************#

    all_fam_fin += finished_pts
    # Copy a log with EDF info to main dir
    if all_fam_fin:
        o = all_fam_fin[0]
        copy2(os.path.join(o.rundir,o.opname+u'.log'),
              os.path.join(mgr.paths.fam_m,u"fam.log"))

    return all_fam_fin

#-------------------------------------------------------------------------------
def finalize_fam_calc(mgr, all_strengths, all_fam_fin):

    for name, strength in list(all_strengths.items()):
        if strength.str_df is None:
            fam_by_op = [o for o in all_fam_fin if o.opname==name]

            # We may have lost the order of the points on gather, so sort by label
            fam_by_op.sort(key=lambda x: x.label)

            # Extract strength dataframes from lists of fam objects and write output
            strength.concatFamData(fam_by_op)
            strength.writeStrengthOut(dest=mgr.paths.fam)
            strength.writeCtrBinary(dest=mgr.paths.fam)

    # Only tar if we have untarred data, as indicated by all_fam_fin not empty
    # (tar in parallel here, since there could be many directories)
    if all_fam_fin:
        tar_itmp = 0 #newrank
        tar_tasks= [[o for o in all_fam_fin if o.opname==name] for name in all_strengths]
        while True:
            if tar_itmp > len(tar_tasks)-1: break
            tar_index = tar_itmp
            tar_itmp += 1 #newcomm_size
            tar_fam   = tar_tasks[tar_index]
            tar_fam_solns(mgr.paths, tar_fam)

# ------------------------------------------------------------------------------
def tar_fam_solns(all_paths, obj_list):
    """
    Bundle individual fam points in a tarfile for portability.

    Args:
        all_paths (pynfamPaths): Paths of the pynfam calculation.
        obj_list (list of pnfamRun): Fortran program run directories to be tarred.
    """
    if not obj_list: return
    top_dir  = os.path.dirname(obj_list[0].rundir)
    tar_name = top_dir+u'.tar'
    with tarfile.open(tar_name,u'w') as tar:
        for obj in obj_list:
            tar.add(obj.rundir, arcname=os.path.relpath(obj.rundir, all_paths.fam_m))
            rmtree(obj.rundir)
    rmtree(top_dir)



