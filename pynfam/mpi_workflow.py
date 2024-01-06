# -------- Backwards Compatibility ---------
from __future__ import print_function
from __future__ import division
from builtins   import range
from builtins   import str
# ---------------- Utilities ---------------
from shutil import copy2
import numpy as np
import os
import sys
# ----------- Relative Imports -------------
from .pynfam_manager        import pynfamManager
from .outputs.pynfam_paths  import pynfamPaths
from .outputs.ls_logger     import hfbLogger
from .outputs.ls_logger     import betaLogger
from .fortran.hfbtho_run    import hfbthoRun
from .fortran.pnfam_run     import pnfamRun
from .strength.contour      import famContour
from .strength.fam_strength import famStrength
from .strength.phase_space  import phaseSpace
from .strength.shape_factor import shapeFactor
from .utilities.mpi_utils import do_mpi
from .utilities import mpi_utils as mu
from .utilities import workflow_utils as wu
from .mpi_workflow_hfb  import *
from .mpi_workflow_fam  import *
from .mpi_workflow_beta import *
# ------------------------------------------

__version__ = u'2.0.0'
__date__    = u'2019-07-26'


#===============================================================================#
#                       Primary Pynfam Functions                                #
#===============================================================================#
def pynfam_mpi_calc(pynfam_inputs, override_settings, check=False):
    """
    A top level function to handle most of the MPI of a large scale pynfam calculation.

    Args:
        pynfam_inputs (dict)
        override_settings (dict)
        check (bool)
    """

    # Setup MPI variables
    if do_mpi:
        comm = mu.MPI.COMM_WORLD
    else:
        comm = 0
    rank, comm_size = mu.pynfam_mpi_traits(comm)

    # User input values valid? (All processes must run this b/c we format some inputs)
    inp_err_i = wu.pynfam_input_check(pynfam_inputs)
    inp_err_o = wu.pynfam_override_check(pynfam_inputs, override_settings)
    inp_err = inp_err_i + inp_err_o
    if rank == 0:
        if inp_err: mu.pynfam_abort(comm, inp_err)

    # Inputs compatible with requested mode? If so retrieve some run parameters.
    inp_err, inp_wrn, nr_calcs, do_fam, exst_data = wu.pynfam_init(pynfam_inputs, override_settings)
    if rank == 0:
        if inp_err: mu.pynfam_abort(comm, inp_err)

    # Setup MPI related parameters, split comm into master/worker.
    inp_err, mpi_wrn, newcomm, group, stdout = wu.pynfam_mpi_init(pynfam_inputs, nr_calcs, comm, check)
    if rank == 0:
        if inp_err and not check: mu.pynfam_abort(comm, inp_err)
        if mpi_wrn: mu.pynfam_warn(mpi_wrn)

    # STOP here if we just wanted to check all the inputs but not run.
    if rank==0 and check:
        if inp_wrn: mu.pynfam_warn(inp_wrn)
        check_msg = u"Input check complete. No errors were detected."
        mu.pynfam_abort(comm, check_msg)

    # Setup PyNFAM directory tree
    if rank==0:
        wu.pynfam_init_dirs(pynfam_inputs)

    # Assign calcs to masters round-robin
    if group == 0:
        kwargs = {u'pynfam_inputs': pynfam_inputs,
                  u'override_settings': override_settings,
                  u'comm': comm,
                  u'do_fam': do_fam,
                  u'stdout': stdout,
                  u'exst_data': exst_data
                  }
        rank, nr_masters = mu.pynfam_mpi_traits(newcomm)
        icalc = rank
        while True:
            if icalc > nr_calcs - 1: break
            index = icalc
            icalc += nr_masters
            kwargs.update({u'index': index})
            pynfam_dripline_calc(**kwargs)

        if do_mpi: mu.runtasks_killsignal(comm, newcomm) # Barrier
    # All else are workers. They don't know anything, just run tasks
    else:
        mu.runtasks_worker(comm, newcomm, stdout)

    if do_mpi: comm.Barrier()
    if rank==0: wu.pynfam_finalize(pynfam_inputs)

# ------------------------------------------------------------------------------
def pynfam_dripline_calc(pynfam_inputs, **kwargs):
    r"""
    Wrapper for the main pynfam_calc function which loops until the dripline
    has been reached, as indicated by the one or two nucleon separation energy
    becoming negative.

    .. math::
        Sn_x &= B(N-x,Z) - B(N,Z)\\
        Sp_x &= B(N,Z-x) - B(N,Z)

    Args:
        pynfam_inputs (dict): pynfam_inputs as in the main input file.
        **kwargs
    """

    # Initialializations
    i_drip = 0
    tEnergy_prev = None
    dripline = pynfam_inputs[u'hfb_mode'][u'dripline_mode']

    # Dripline Loop
    while True:
        kwargs.update({
            u'pynfam_inputs':pynfam_inputs,
            u'index2': i_drip
            })
        hfb_gs = pynfam_calc(**kwargs)

        if not dripline or hfb_gs is None:
            break
        if tEnergy_prev is not None:
            dE = tEnergy_prev - hfb_gs.soln_dict[u'Energy']
            if dE < 0.0:
                break
        tEnergy_prev = hfb_gs.soln_dict[u'Energy']
        i_drip += 1

# ------------------------------------------------------------------------------
def pynfam_calc(index, index2, pynfam_inputs, override_settings, comm, do_fam, stdout, exst_data):
    """
    The main pynfam calculation.

    The calculation is carried out in 3 stages:
        1) Find the HFB ground state (deformed, even or odd)
        2) Compute the FAM strength along a contour (open or closed contour)
        3) Integrate phase space weighted strength for rates

    Args:
        pynfam_inputs (dict)
        override_settings (dict)
        comm (mpi4py.MPI.intracomm, int)
        do_fam (bool)
        stdout (bool)
        exst_data (list)
    """
    #---------------------------------------------------------------------------#
    #                        Initialize PyNFAM Calc                             #
    #---------------------------------------------------------------------------#
    dirs        = pynfam_inputs[u'directories']
    rerun       = pynfam_inputs[u'rerun_mode']

    dripline    = pynfam_inputs[u'hfb_mode']['dripline_mode']
    gs_def_scan = pynfam_inputs[u'hfb_mode']['gs_def_scan']
    ignore_nc   = pynfam_inputs[u'hfb_mode']['ignore_nonconv']

    beta_type   = pynfam_inputs[u'fam_mode']['beta_type']
    fam_ops_in  = pynfam_inputs[u'fam_mode']['fam_ops']
    ctr_type    = pynfam_inputs[u'fam_mode']['fam_contour']

    fam_ops = wu.get_fam_ops(fam_ops_in, beta_type)
    setts = wu.pynfam_settings(override_settings, index)

    # Unique run label
    if rerun==1 and exst_data:
        calclabel = exst_data[index]
    else:
        calclabel = str(index).zfill(6)
    if dripline:
        calclabel_base = calclabel
        calclabel += u'_'+str(index2).zfill(4)

    # Paths (requires a calclabel, otherwise not fully defined) and manager
    all_paths = pynfamPaths(calclabel, dirs[u'outputs'], dirs[u'exes'], dirs[u'scratch'])
    mgr = pynfamManager(all_paths)
    mgr.paths.mkdirs()

    #---------------------------------------------------------------------------#
    #                               HFB Calc                                    #
    #---------------------------------------------------------------------------#
    hfb_main, hfb_gs, hfb_evens, hfb_odds = initialize_hfb_calc(mgr,
            beta_type, gs_def_scan, setts, dripline, index2)
    # In case we have hfb_gs already, but need to tar files, set fin lists
    even_fin, odd_fin = hfb_evens[1], hfb_odds[1]

    if hfb_gs is None:
        even_fin = run_hfb_even_calc(comm, mgr, ignore_nc, stdout, *hfb_evens)
        if even_fin is None: return

        odd_fin = run_hfb_odd_calc(comm, mgr, ignore_nc, stdout,
                gs_def_scan, hfb_main, even_fin, *hfb_odds)
        if odd_fin is None: return

        hfb_gs = finalize_hfb_gs(mgr, ignore_nc, hfb_main, odd_fin, even_fin)

    finalize_hfb_calc(mgr, hfb_gs, odd_fin, even_fin)
    if hfb_gs is None: return

    if not do_fam:
        msg = u"No FAM contour or operators requested. Exiting after hfb calc."
        mu.pynfam_warn(msg, calclabel)
        return hfb_gs

    #------------------------------------------------------------------------#
    #                               FAM Calc                                 #
    #------------------------------------------------------------------------#
    ctr_main = initialize_fam_contour(mgr, setts, ctr_type, beta_type, hfb_gs)
    if ctr_main is None: return hfb_gs

    all_strengths, fam_pts = initialize_fam_calc(mgr, setts, fam_ops, ctr_main, hfb_gs)

    all_fam_fin = run_fam_calc(comm, mgr, stdout, *fam_pts)
    if all_fam_fin is None: return hfb_gs

    finalize_fam_calc(mgr, all_strengths, all_fam_fin)

    #------------------------------------------------------------------------#
    #                               Beta Calc                                #
    #------------------------------------------------------------------------#
    check_qval(hfb_gs)
    psi_main = phaseSpace(beta_type)
    psi_main.updateSettings(setts[u'psi'])

    # Raw data shape factor (written to beta_meta for open ctrs)
    shapefacs = shapeFactor(list(all_strengths.values()))
    try:
        shapefacs.calcShapeFactor(psi_main, hfb_gs)
        rates_df = calc_beta_decay_results(mgr, shapefacs, ctr_main.closed)
        calc_gamow_teller_results(mgr, shapefacs, ctr_main.closed)

        # Adjusted shape factor with zeroed negative strength
        # (written to beta_soln for open ctrs). Note this uses findFirstPeak
        # (which may fail) to set all str before 1st peak to zero.
        if not ctr_main.closed:
            shapefacs = shapeFactor(list(all_strengths.values()))
            shapefacs.zeroNegStr(psi_main, hfb_gs)
            rates_df = calc_beta_decay_results(mgr, shapefacs, ctr_main.closed)
            calc_gamow_teller_results(mgr, shapefacs, ctr_main.closed)

        write_phase_space(mgr, shapefacs)
        write_beta_log(mgr, shapefacs, rates_df, hfb_gs)
    except RuntimeError as rte:
        msg = ["There was an error calculating the beta decay rates."]+[str(rte)]
        pynfam_warn(msg, mgr.paths.calclabel)

    # Return ground state for dripline loop
    return hfb_gs
