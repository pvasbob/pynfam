# -------- Backwards Compatibility ---------
from __future__ import print_function
from __future__ import division
from builtins   import zip
# -------------- Utilities -----------------
import pandas as pd
import numpy as np
# ----------- Relative Imports -------------
from .parser import parser
# ------------------------------------------

__version__ = u'2.0.0'
__date__    = u'2019-07-26'

#===============================================================================#
#                             CLASS pnfamLogParser                              #
#===============================================================================#
class pnfamLogParser(parser):
    """
    A pnfamLogParser is a subclass of the parser class that contains a
    pnfam log and the methods to parse it for values.

    Args:
        pnfam_output (str, list): A str is assumed to be path to output file, which
            is then read in and converted to a list of strings. Newline symbols
            are stripped.

    Attributes:
        keys (dict): A collection of strings to search for in the output.
    """
    def __init__(self, pnfam_output):
        parser.__init__(self, pnfam_output)
        self.keys = {u'time': u"Time (s)",
                     u'conv': u"FAM was interrupted at iteration limit",
                     u'header': u"Energy [MeV]"}

    #-----------------------------------------------------------------------
    def getTime(self):
        """
        Parse for the time.

        Returns:
            float
        """
        try:
            l_t = self.getLineIndices(self.keys[u'time'])[0]
            time = self.getNumbers(self.output[l_t+2], -2, err_val=self.float_err)
        except Exception:
            time = self.float_err
            self.flagError(u"Error parsing time from FAM log.")
        return time

    #-----------------------------------------------------------------------
    def getConv(self):
        """
        Parse for the convergence.

        Returns:
            str
        """
        conv_lines = self.getLineIndices(self.keys[u'conv'])
        header = self.getLineIndices(self.keys[u'header'])
        # hacky way to avoid false positives since pnfam doesn't print convergence
        # is to check the file contents seem legit by looking for the header.
        if conv_lines or not header:
            return u'No'
        else:
            return u'Yes'


#===============================================================================#
#                             CLASS pnfamOutParser                              #
#===============================================================================#
class pnfamOutParser(parser):
    """
    A pnfamOutParser is a subclass of the parser class that contains a
    pnfam output file (OP.out) and the methods to parse it for values.

    Args:
        pnfam_output (str, list): A str is assumed to be path to output file, which
            is then read in and converted to a list of strings. Newline symbols
            are stripped.

    Attributes:
        keys (dict): A collection of strings to search for in the output.
    """
    def __init__(self, pnfam_output):
        parser.__init__(self, pnfam_output)
        self.keys = {u"version"    : u"pnFAM code version",
                     u"interaction": u"Residual interaction",
                     u"operator"   : u"Operator",
                     u"half_width" : u"Gamma",
                     u"strength"   : u"Re(EQRPA)"}

    #-----------------------------------------------------------------------
    def getVersion(self):
        """
        Parse for the pnfam version.

        Returns:
            str
        """
        try:
            l_v = self.getLineIndices(self.keys[u'version'])[0]
            version = self.output[l_v].split(u'version')[-1].strip()
        except Exception:
            version = self.str_err
            self.flagError(u"Error parsing version from FAM output.")
        return version

    #-----------------------------------------------------------------------
    def getStrength(self):
        """
        Parse for the strength value.

        Returns:
            list of dict: Of form {'label: str, 'val': float} for each column.
        """
        try:
            l_s = self.getLineIndices(self.keys[u'strength'])[0]
            labels = self.output[l_s].split()[2:]
            str_vals = self.output[l_s+1].split()[1:]

            # Store as list of dicts, with Re and Im parts separate. Using this
            # more or less as a list of glorified tuples with names rather than 0,1
            strength = [{u'label':h,u'val':float(d)} for h, d in zip(labels,str_vals)]
        except Exception:
            strength = 2*[{u'label':self.str_err, u'val':self.float_err}]
            self.flagError(u"Error parsing strength from FAM output.")
        return strength

    #-----------------------------------------------------------------------
    def getXterms(self):
        """
        Parse for the cross term labels.

        Returns:
            list of str
        """
        try:
            l_s = self.getLineIndices(self.keys[u'strength'])[0]
            # Skip [#, EQRPA, Re(Strength), Im(Strength)]
            labels = self.output[l_s].split()[4:]
            xterms = [x[3:-1] for x in labels if u'Re' in x]
        except Exception:
            xterms = []
            self.flagError(u"Error parsing cross terms from FAM output.")
        return xterms

#===============================================================================#
#                             CLASS strengthOutParser                           #
#===============================================================================#
class strengthOutParser(parser):
    """
    A strengthOutParser is a subclass of the parser class that contains a pynfam
    compiled strength output file (OP.out) and the methods to parse it for values.

    Args:
        pnfam_output (str, list): A str is assumed to be path to output file, which
            is then read in and converted to a list of strings. Newline symbols
            are stripped.

    Attributes:
        keys (dict): A collection of strings to search for in the output.
    """
    def __init__(self, pnfam_output):
        parser.__init__(self, pnfam_output)
        self.keys = {u"version": u"pnFAM code version",
                     u"time"   : u"Total run time",
                     u"conv"   : u"All points converged"}

    #-----------------------------------------------------------------------
    def getVersion(self):
        """
        Parse for the pnfam version.

        Returns:
            str
        """
        try:
            l_v = self.getLineIndices(self.keys[u'version'])[0]
            version = self.output[l_v].split(u'version')[-1].strip()
        except Exception:
            version = self.str_err
            self.flagError(u"Error parsing version from strength output.")
        return version

    #-----------------------------------------------------------------------
    def getTime(self):
        """
        Parse for the total fam time.

        Returns:
            float
        """
        try:
            l_t = self.getLineIndices(self.keys[u'time'])[0]
            time = self.getNumbers(self.output[l_t], -2, err_val=self.float_err)
        except Exception:
            time = self.float_err
            self.flagError(u"Error parsing total time from strength output.")
        return time

    #-----------------------------------------------------------------------
    def getConv(self):
        """
        Parse for the total fam convergence.

        Returns:
            str
        """
        try:
            l_c = self.getLineIndices(self.keys[u'conv'])[0]
            conv = self.output[l_c].split()[-1].strip()
        except Exception:
            conv = self.str_err
            self.flagError(u"Error parsing total convergence from strength output.")
        return conv

    #-----------------------------------------------------------------------
    def getAllConv(self):
        """
        Parse for the entire convergence column.

        Returns:
            ndarray
        """
        try:
            io = pd.compat.StringIO(u'\n'.join(self.output))
            df = pd.read_csv(io, delim_whitespace=True, header=0, comment=u'#')
            conv_list = df[u'Conv'].dropna().values
        except Exception:
            conv_list = np.array([self.str_err])
            self.flagError(u"Error parsing convergence list from strength output.")
        return conv_list

    #-----------------------------------------------------------------------
    def getAllTime(self):
        """
        Parse for the entire time column.

        Returns:
            ndarray
        """
        try:
            io = pd.compat.StringIO(u'\n'.join(self.output))
            df = pd.read_csv(io, delim_whitespace=True, header=0, comment=u'#')
            time_list = df[u'Time'].dropna().values
        except Exception:
            time_list = np.array([self.float_err])
            self.flagError(u"Error parsing time list from strength output.")
        return time_list
