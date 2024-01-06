# -------- Backwards Compatibility ---------
from __future__ import print_function
from __future__ import division
from builtins   import str
# -------------- Utilities -----------------
from collections import OrderedDict as OD
import numpy as np
import copy
import os
# ----------- Relative Imports -------------
from ..outputs.pnfam_parser import pnfamLogParser
from ..outputs.pnfam_parser import pnfamOutParser
from .fortran_utils         import fortProcess
from .hfbtho_run            import hfbthoRun
from ..config               import DEFAULTS
# ------------------------------------------

__version__ = u'2.0.0'
__date__    = u'2019-07-26'

#===============================================================================#
#                             CLASS pnfamRun                                    #
#===============================================================================#
class pnfamRun(fortProcess):
    """
    A pnfamRun is a fortProcess subclass that contains the methods to run the
    serial pnfam fortran program for a single energy point and handle the output.

    All of the higher level beta decay calculations are handled by python. The
    pnfamRun class is effectively one iterative solution of the pnfam equations
    for a single energy point and operator, analogous to one run of HFBTHO.

    Args:
        paths (pynfamPaths): The paths for the pynfam calculation.
        operator (str): The operator name, as in the pnfam namelist.
        kval (int): The operator K-value, as in the pnfam namelist.

    Attributes:
        op (str): The operator name, as in the pnfam namelist.
        k (int): The operator K-value, as in the pnfam namelist.
        soln_dict (dict): Contains key outputs.
        soln_file (file): The pnfam output file name OP.out.
        _soln (pnfamOutParser): Internal initialization for soln attribute.
        _log (pnfamLogParser): Internal initialization for log attribute.
        ctr_params (dict): Contour parameters for the EQRPA point.
    """

    file_exenompi=u"pnfam_nompi.x"
    """ file_exenompi (str): The executable filename.
    """
    # file_nml = instance.opname + ".in"
    # file_txt = instance.opname + ".out"
    # file_log = instance.opname + ".log"

    def __init__(self, paths, operator, kval):
        # Operator characteristics
        self.op = operator
        self.k = kval

        # Inheret/override attributes for fortran object
        exe = pnfamRun.file_exenompi
        default_nml = self._getDefaultNml(DEFAULTS[u'fam'], paths)
        fortProcess.__init__(self, paths, exe, self.opname+u'.in', default_nml)

        # Strength result
        self.output = None
        self.log_output = None
        self.soln_err = False
        self.soln_err_msg = u''
        self.soln_dict = {}

        # Contour parameters (Note EQRPA point comes from namelist-see property omega)
        self.ctr_params = None

    @property
    def opname(self):
        """ opname (str): Bare_operator + beta_type + kval.
        """
        return u"{:}K{:1d}".format(self.op, self.k)
    @property
    def bareop(self):
        """ bareop (str): Bare operator name without beta type.
        """
        return self.op[:-1]
    @property
    def beta(self):
        """ beta (str): Beta decay type.
        """
        return self.op[-1]
    @property
    def genopname(self):
        """ genopname (str): Bare_operator + _ + kval.
        """
        return u"{:}_K{:1d}".format(self.bareop, self.k)
    @property
    def omega(self):
        """ omega (complex): The complex energy point.
        """
        return complex(self.nml[u'STR_PARAMETERS'][u'energy_start'],
                       self.nml[u'STR_PARAMETERS'][u'half_width'])
    @property
    def interaction(self):
        """ interaction (str): The interaction name as in the pnfam namelist.
        """
        return self.nml[u'INTERACTION'][u'interaction_name']

    #-----------------------------------------------------------------------
    def updateLogOutput(self, output, get_soln=True):
        self.log_output = output
        if get_soln:
            sd, err, emsg = self.getSolnLog(output)
            self.soln_dict.update(sd)
            self.soln_err = err
            self.soln_err_msg = emsg

    #-----------------------------------------------------------------------
    def updateOutput(self, output, get_soln=True):
        self.output = output
        if get_soln:
            sd, err, emsg = self.getSolnOut(output)
            self.soln_dict.update(sd)
            self.soln_err = err
            self.soln_err_msg = emsg

    #-----------------------------------------------------------------------
    def setCtrPoint(self, ctr_pt):
        """
        Translate a complex energy into the appropriate namelist inputs.

        Args:
            ctr_pt (complex): The complex energy point.
        """

        setpoint = {
                u'STR_PARAMETERS':
                    {u'energy_start': np.real(ctr_pt),
                     u'energy_step' : 0.0,
                     u'nr_points'   : 1,
                     u'half_width'  : np.imag(ctr_pt)},
                u'GENERAL':
                    {u'fam_mode': u'STR'}
                }
        self.adjustNml(setpoint)

    #-----------------------------------------------------------------------
    def runExe(self, stdout_bool=False, debug=0):
        """
        Extend the runExe method to handle the pnfam output.

        Args:
            stdout_bool (bool): Print to stdout in real time.
            debug (int): If !=0 don't actually run the executable.

        Returns:
            str, None : Stderr or None if empty.
        """

        # Returns stdout, stderr.
        stmt = [os.path.join(self.paths.exe, self.exe), self.fname_nml]
        out, err = fortProcess.runExe(self, stdout_bool=stdout_bool,
            statement=stmt, debug=debug)

        # Require stdout for op.log. It has EDF info, time, and is used as fam.log
        if not out and err is None:
            err = u"The log data from pnfam stdout is missing."
        else:
            logfile = os.path.join(self.rundir,self.opname+u'.log')
            with open(logfile,u'w') as bl:
                for line in out:
                    bl.write(line+u'\n')

        # Use out since we already have it, rather than reading from file.
        self.updateLogOutput(out, get_soln=True)

        # Require the op.out file for parsing strength
        outfile = os.path.join(self.rundir, self.opname+u'.out')
        if not os.path.exists(outfile) and err is None:
            err = u"Missing pnFAM output file."

        self.updateOutput(outfile, get_soln=True)

        # Collect errors (output errors are printed over log errors)
        if self.soln_err and err is None:
            err = self.soln_err_msg

        return err

    #-----------------------------------------------------------------------
    def getSolnLog(self, output=None):
        """ Parse log file and return soln_dict.
        """

        if output is None: output = self.output

        soln = pnfamLogParser(output)
        sd = {}

        sd[u'Time'] = soln.getTime()
        sd[u'Conv'] = soln.getConv()

        return sd, soln.err, soln.err_msg

    #-----------------------------------------------------------------------
    def getSolnOut(self, output=None):
        """ Parse output file and return soln_dict.
        """

        if output is None: output = self.output

        soln = pnfamOutParser(output)
        sd = {}

        sd[u'Version']  = soln.getVersion()
        sd[u'Strength'] = soln.getStrength()
        sd[u'Xterms']   = soln.getXterms()

        return sd, soln.err, soln.err_msg

    #-----------------------------------------------------------------------
    def _getDefaultNml(self, DEFAULTS_FAM, paths):
        """
        Combine the default fam settings with the rest of the parameters
        needed to run pnfam but are handled internally.

        Args:
            DEFAULTS_FAM (OrderedDict): The default values from config.py.

        Returns:
            OrderedDict: The full namelist for pnfam.
        """

        default_nml = copy.deepcopy(DEFAULTS_FAM)

        # Some namelist parameters are not available to the user -- some are fixed
        # while others are handled by other objects, e.g. contour.
        # Note: hfb_input_filename has dummy vals for opdir/eqrpadir just to get the
        # number of ../ correct in the rel path.
        fam_default_nml = OD([
            (u'GENERAL',            OD([
                    (u'fam_mode'           , u'STR'),
                    (u'hfb_input_filename' ,
                     os.path.relpath(os.path.join(paths.hfb, hfbthoRun.file_fam),
                            os.path.join(paths.fam_m, u'opdir',u'eqrpadir'))),
                    (u'fam_output_filename', self.opname+u'.out'),
                    (u'log_filename'       , self.opname+u'.log')
                    ])),
            (u'EXT_FIELD',          OD([
            	    (u'operator_name' , self.op),
                    (u'K' , self.k)
            	    ])),
            (u'STR_PARAMETERS',     OD([
                    (u'nr_points'   , 1),
                    (u'energy_start', 0.0),
                    (u'energy_step' , 0.0),
                    (u'half_width'  , 0.1)
                    ]))
        ])
        fam_default_nml[u'EXT_FIELD'][u'compute_crossterms'] = default_nml[u'EXT_FIELD'][u'compute_crossterms']
        default_nml.pop(u'EXT_FIELD')
        fam_default_nml.update(default_nml)

        return fam_default_nml
