# -------- Backwards Compatibility ---------
from __future__ import print_function
from __future__ import division
# -------------- Utilities -----------------
import numpy as np
# ----------- Relative Imports -------------
from .utilities.hfb_utils import convString
from .utilities.mpi_utils import pynfam_warn
from .outputs.ls_logger import betaLogger
# ------------------------------------------

__version__ = u'2.0.0'
__date__    = u'2019-08-28'


#-------------------------------------------------------------------------------
def check_qval(hfb_gs):
    # If the Q value is negative, dont do any calcs involving phase space
    try:
        qval = hfb_gs.soln_dict[u'HFB_Qval']
        msg  = u"Cannot compute rates for Q < 0! Setting phase space to zero."
    except KeyError:
        qval = -999.9
        msg  = u"No Q value was found in the HFB solution. Phase space will be set to zero."
    if qval < 0:
        pynfam_warn(msg)

#-------------------------------------------------------------------------------
def calc_beta_decay_results(mgr, shapefacs, ctr_closed):

    # Set the filenames and dest
    if shapefacs.is_raw and not ctr_closed:
        sf_dest = mgr.paths.beta_m
        prefix = u'raw_'
    else:
        sf_dest = mgr.paths.beta
        prefix = u''

    # Store the pswsf (real part as meta data)
    title = u"# Nuclear Beta Decay Phase Space Weighted Shape Factors (Imaginary)"
    fname = prefix+u"shapefactor_im.out"
    shapefacs.writeOutput(shapefacs.sf_df_imag, title, fname, sf_dest)

    title = u"# Nuclear Beta Decay Phase Space Weighted Shape Factors (Real)"
    fname = prefix+u"shapefactor_re.out"
    shapefacs.writeOutput(shapefacs.sf_df_real, title, fname, mgr.paths.beta_m)

    # Store the rates
    title = u"# Nuclear Beta Decay Rates and Half-Lives"
    fname = prefix+u"beta.out"
    rates_df = shapefacs.calcBetaRates()
    shapefacs.writeOutput(rates_df, title, fname, sf_dest)

    # Store the cumulative strengths
    if not ctr_closed:
        title = u"# Nuclear Beta Decay Cumulative Strengths"
        fname = prefix+u"cumstr.out"
        cum_str = shapefacs.calcCumulativeStr()
        shapefacs.writeOutput(cum_str, title, fname, sf_dest)

    return rates_df

#-------------------------------------------------------------------------------
def calc_gamow_teller_results(mgr, shapefacs, ctr_closed):
    genops = shapefacs.genops
    if ctr_closed or not (u'GT_K0' in genops and u'GT_K1' in genops):
        return

    # Gamow-Teller Strength (Update E_1stpeak)
    if shapefacs.is_raw:
        sf_dest = mgr.paths.beta_m
        prefix = u'raw_'
        gt_str = shapefacs.calcTotalGT(zero_neg_str=False)
        ind = shapefacs.findFirstPeak(gt_str[u'Re(EQRPA)'].values, gt_str[u'Total-GT'].values)[0]
        shapefacs.sf_metadict[u'E_1stPeak'] = gt_str[u'Re(EQRPA)'].values[ind]
    else:
        sf_dest = mgr.paths.beta
        prefix = u''
        gt_str = shapefacs.calcTotalGT(zero_neg_str=True)

    title = u"# Nuclear Beta Decay Gamow-Teller Strength"
    fname = prefix+u"totalGT_str.out"
    shapefacs.writeOutput(gt_str, title, fname, sf_dest)

#-------------------------------------------------------------------------------
def write_phase_space(mgr, shapefacs):

    title = u"# Nuclear Beta Decay Phase Space (Imaginary)"
    fname = u"phasespace_im.out"
    shapefacs.writeOutput(shapefacs.ps_df_imag, title, fname, mgr.paths.beta_m)

    title = u"# Nuclear Beta Decay Phase Space (Real)"
    fname = u"phasespace_re.out"
    shapefacs.writeOutput(shapefacs.ps_df_real, title, fname, mgr.paths.beta_m)

#-------------------------------------------------------------------------------
def write_beta_log(mgr, shapefacs, rates_df, hfb_gs):
    betalog = betaLogger(u"logfile_mini.dat")
    beta_soln_dict = hfb_gs.soln_dict.copy()

    # Ignore the 0/0 warning for %FF
    with np.errstate(divide=u'ignore', invalid=u'ignore'):
        beta_soln_dict.update({
            u'Total-HL' :  rates_df.loc[u'Total',          u'Half-Life(s)'],
            u'%FF'      : (rates_df.loc[u'Total-Forbidden',u'Rate(s^-1)']/rates_df.loc[u'Total', u'Rate(s^-1)'])*100.0,
            u'FAM_Qval' : shapefacs.sf_metadict[u'FAM_Qval']
            })

    # Get conv and time for all operators
    conv_list = [s.meta[u'Conv'] for s in list(shapefacs.strengths.values())]
    beta_soln_dict[u'FAM_Conv'] = convString(conv_list)
    fam_time = sum([s.meta[u'Time'] for s in list(shapefacs.strengths.values())])
    beta_soln_dict[u'FAM_Time'] = fam_time

    betalog.quickWrite(beta_soln_dict, dest=mgr.paths.beta)

