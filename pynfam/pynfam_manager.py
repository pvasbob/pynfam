# -------- Backwards Compatibility ---------
from __future__ import print_function
from __future__ import division
from builtins   import object
from builtins   import str
# -------------- Utilities -----------------
from shutil import copy2, rmtree
from copy import deepcopy
import tarfile
import numpy  as np
import pandas as pd
import os
# ----------- Relative Imports -------------
from .outputs.pynfam_paths  import pynfamPaths
from .outputs.ls_logger     import hfbLogger
from .outputs.ls_logger     import lsLogger
from .fortran.hfbtho_run    import hfbthoRun
from .fortran.pnfam_run     import pnfamRun
from .strength.contour      import famContour
from .strength.fam_strength import famStrength
from .strength.phase_space  import phaseSpace
from .strength.shape_factor import shapeFactor
from .utilities.hfb_utils   import hfbEvenDefList
from .utilities.hfb_utils   import hfbOddList
# ------------------------------------------

__version__ = u'2.0.0'
__date__    = u'2019-07-26'

#===============================================================================#
#                             CLASS pynfamManager                               #
#===============================================================================#
class pynfamManager(object):
    """
    A manager for paths, fortran objects, and output data related to a pynfam calculation.

    An instance of the class pynfamManager contains the relevant paths for a pynfam
    calculation, methods manipulate fortran objects, and methods to convert
    exisiting output files into fortran objects. An instance can be instantiated
    with a pynfamPaths object, or a string indicating the pynfam level directory
    (e.g. 'top_level_dir/000000'), which is then converted into a pynfamPaths object.

    Args:
        pynfam_paths (pynfamPaths, str): The paths of the pynfam calculation.

    Attributes:
        paths (pynfamPaths): The paths of the pynfam calculation.

    Notes:
        * Methods (HFB Objects):
              - hfbEvenDefList
              - hfbRerunList
              - hfbOddList
              - hfbConvIgnoreNC
              - hfbConvCheck
              - hfbGroundState
        * Methods (Object-OutputFile Interface):
              - getNonConv
              - getAllHfbObjs
              - getHfbthoRun
              - getPnfamRun
              - getShapeFactor
              - getHfbState
              - getFamState
        * Methods (Output Data Compiling):
              - gatherHfbLogs
              - gatherHfbMasterLog
              - gatherFamMasterLog
              - gatherBetaMasterLog
    """

    def __init__(self, pynfam_paths):
        if isinstance(pynfam_paths,str):
            topdir = os.path.dirname(pynfam_paths)
            label = os.path.basename(os.path.normpath(pynfam_paths))
            pynfam_paths = pynfamPaths(label, topdir)
        self.paths = pynfam_paths


    #==============================================================#
    #             Interface objects with existing data             #
    #==============================================================#
    def getNonConv(self, obj=False):
        """
        Generate a list of non-converged hfb solutions from existing
        output logfiles.

        Args:
            obj (bool): Return type is hfbthoRun if True (default, False)

        Returns:
            DataFrame, list: if obj=True, returns list of hfbthoRun objects
        """

        hfb_df = self.gatherHfbLogs(skip=True)
        nc_df = hfb_df.loc[hfb_df[u'Conv'] != u'Yes']

        if not obj:
            return nc_df.reset_index(drop=True)
        else:
            hfb_list = []
            for (index, row) in nc_df.iterrows():
                hfb = self.getHfbthoRun(row[u'Label'])
                hfb_list.append(hfb)
            return hfb_list

    #--------------------------------------------------------------
    def getAllHfbObjs(self, beta_type):
        """
        Generate a list of all hfbthoRun objects from existing ouputs.

        Args:
            beta_type (str): Type of beta decay.

        Returns:
            list of hfbthoRun
        """

        p = self.paths
        objs = []
        labels = p.getLabels(p.hfb_m, tier2=True, prefix=p.out)
        for l in labels:
            try:
                objs.append(self.getHfbthoRun(l, beta_type))
            except IOError:
                continue

        return objs

    #--------------------------------------------------------------
    def getHfbthoRun(self, label, beta_type=None, paths_obj=None):
        """
        Create an hfbthoRun object from existing outputs.

        If beta_type is None, solution values which depend on this
        will not be populated (e.g. Q_value, EQRPA_max,...).

        Args:
            label (str): hfbtho run label
            beta_type (str): beta decay type (default, None)
            paths_obj (pynfamPaths): (default, None --> self.paths)

        Returns:
            hfbthoRun

        Raises:
            IOError
        """

        if paths_obj is None: paths_obj = self.paths
        # instantiate and set label
        hfb = hfbthoRun(paths_obj, beta_type)
        hfb.label = label
        # set namelist
        hfb.readNml(hfb.rundir)
        # read solution data
        outfile = os.path.join(hfb.rundir, hfbthoRun.file_txt)
        hfb.updateOutput(outfile, get_soln=True)

        if hfb.soln_err: raise IOError(u"Problem parsing HFBTHO output.")
        return hfb

    #--------------------------------------------------------------
    def getPnfamRun(self, operator, kval, label, ctr_params=None, paths_obj=None):
        """
        Create a pnfamRun object from existing outputs.

        Args:
            operator (str): The operator as in the pnfam namelist.
            kval (int): The k value as in the pnfam namelist.
            label (str) : The pnfam run label.
            ctr_params (dict) : Contour parameters for this point (default, None).
            paths_obj (pynfamPaths): (default, None --> self.paths)

        Returns:
            pnfamRun

        Raises:
            IOError
        """

        if paths_obj is None:
            paths_obj = self.paths
        # instantiate and set label
        fam = pnfamRun(paths_obj, operator, kval)
        fam.label = label
        # set namelist
        fam.readNml(fam.rundir)
        # read solution data
        logfile = os.path.join(fam.rundir,fam.opname+u'.log')
        outfile = os.path.join(fam.rundir,fam.opname+u'.out')
        fam.updateLogOutput(logfile, get_soln=True)
        fam.updateOutput(outfile, get_soln=True)
        if fam.soln_err: raise IOError(u"Problem parsing pnfam output.")
        # Contour info
        if ctr_params is not None:
            fam.ctr_params = ctr_params
        #  { 'dzdt'  : self.contour.ctr_dzdt[i],
        #    'theta' : self.contour.theta[i],
        #    'glwt'  : self.contour.glwts[i]
        #    }
        return fam

    #--------------------------------------------------------------
    def getShapeFactor(self, contour_type, beta_type='-', raw=True, ps_settings=None, new_dfs=None):
        """
        Create a shapeFactor object from existing outputs, with modifications if desired.

        The shapeFactor can be altered e.g. by supplying different phaseSpace settings,
        or manually supplying the complex str_df for a given operator to override the
        binary file data.

        Args:
            contour_type (str): Type of contour object.
            beta_type (str): Type of beta decay.
            raw (bool): if False, negative strength will be zeroed
            ps_settings (dict): same form as in run_pynfam.py script
            new_dfs (dict): form {"opname": dataframe} where dataframe has the same
                form as famStrength.str_df

        Returns:
            shapeFactor

        Raises:
            IOError
        """

        ctr   = famContour(contour_type)
        pso   = phaseSpace(beta_type)
        if ps_settings is not None: pso.updateSettings(ps_settings)
        label = self.paths.rp(self.paths.hfb)
        hfb_gs= self.getHfbthoRun(label, beta_type=beta_type)
        strengths = []
        for f in [f for f in os.listdir(self.paths.fam) if f.endswith(u'.ctr')]:
            op = f.split(u'K')[0]
            k = int(f.split(u'K')[1][0])
            s = famStrength(op, k, ctr)
            s.readCtrBinary(self.paths.fam, f)
            if new_dfs is not None and s.opname in new_dfs:
                df = new_dfs[s.opname]
                # str_df contains only strength and xterms, split by Re and Im
                bare = ["Strength"]+s.xterms;  pref = ["Re","Im"]
                scols= ["{:}({:})".format(ir,c) for c in bare for ir in pref]
                drop = [c for c in df.columns if c not in scols]
                if not all(sc in list(df.columns) for sc in scols):
                    raise ValueError("New data frame does not contain necessary str_df columns.")
                s.str_df = df.drop(columns=drop)
            strengths.append(s)
        found_ops  = sorted([s.genopname for s in strengths])
        unique_ops = sorted(list(set([s.genopname for s in strengths])))
        if found_ops != unique_ops:
            raise IOError(u"Multiple files for the same operator found")
        pswsf = shapeFactor(strengths)
        if not raw:
            pswsf.zeroNegStr(pso,hfb_gs)
        else:
            pswsf.calcShapeFactor(pso, hfb_gs)
        return pswsf


    #==========================================================================#
    #                   Determine the State of Existing Outputs                #
    #==========================================================================#
    def _getTotalHfbTime(self):
        """
        Parse logs to get the total hfb time for a given pynfam calculation.

        This method is meant for use inside the getHfbState method.
        If data is missing, try reconstructing the result, but if reconstructing
        the result requires re-running HFBTHO, just return an error value.
        Note that solution.hfb is deleted for all but ground state solution,
        which might trigger re-running HFBTHO if enough data is missing (and
        would subsequently be skipped and return error values).

        Returns:
            float, np.nan, None: float for result, np.nan for error value,
                None for if we can reconstruct by entering rerun mode.
        """

        p = self.paths
        ftar = os.path.join(p.hfb_m, hfbthoRun.file_tar)
        # Try to get from existing ground state mini log
        try:
            log = hfbLogger(hfbthoRun.file_log).readLog(p.hfb)
            return log.loc[0,u'HFB_Time']
        except (IOError, KeyError):
            pass
        # Fall back to summing the times in hfb_meta log
        try:
            hfb_meta_df = hfbLogger().readLog(p.hfb_m)
            return np.sum(hfb_meta_df[u'Time'].values)
        except IOError:
            # no hfb meta - keep error val, not worth rerun
            if not os.path.exists(p.hfb_m):
                return np.nan
            # hfb meta, but it's empty - keep error val, not worth rerun
            elif not os.listdir(p.hfb_m):
                return np.nan
            # hfb meta, it's not empty, but no tarfile - get objects and remake
            elif not os.path.exists(ftar):
                return None
            # hfb meta, with tarfile - untar and treat as untarred
            else:
                with tarfile.open(ftar) as tf:
                    tf.extractall(path=p.hfb_m)
                copy2(ftar, os.path.join(p.hfb_m, hfbthoRun.file_tar))
                os.remove(ftar)
                return None

    #--------------------------------------------------------------
    def getRemainingHfbTasks(self, tasks, fsln, beta_type):
        """
        Given a list of tasks, find which are already completed and populate the
        instance objects.

        Args:
            tasks (list of hfbthoRun): The list of tasks for the calculation, based on the input file.
            fsln (str): A task is determined completed based on the existence of this filename.
            beta_type (str): The type of beta decay.

        Returns:
            list of hfbthoRun: remaining, unfinished tasks.
            list of hfbthoRun: completed tasks.
        """

        fin, unf = [], []

        fin_f = [t for t in tasks if os.path.exists(os.path.join(t.rundir,fsln))]
        lfin_f = [t.label for t in fin_f]

        # Unfinished if solution not converged (or namelist and thoout dont exist)
        for l in lfin_f:
            try:
                obj = self.getHfbthoRun(l, beta_type)
                if obj.soln_dict[u'Conv']==u'Yes':
                    fin.append(obj)
                else:
                    # Allow for more iterations on NC solns
                    obj.setNmlParam({u'lambda_active':8*[0]})
                    obj.setNmlParam({u'expectation_values':8*[0.0]})
                    obj.setRestartFile(True)
                    unf.append(obj)
            except IOError:
                continue
        lfin = [t.label for t in fin+unf]

        # Unfinished labels. Unf+Fin must = tasks according to input file
        unf += [t for t in tasks if t.label not in lfin]
        assert(len(unf)+len(fin) == len(tasks))

        return unf, fin

    #--------------------------------------------------------------
    def getHfbState(self, hfb_main, gs_def_scan, beta_type):
        """
        Determine the state of existing hfb outputs.

        States considered:

            * GS solution.hfb and tarfile - skips to FAM
            * GS solution.hfb and untarred data - skips to HFB Finalize
            * No GS and tarfile - untar and treat as below
            * No GS and untarred data - unfinished and finished objects

                - non-converged count as unfinished
                - blocking core with wel file counts as finished
                  (regardless if solution.hfb exists)

        Args:
            hfb_main (hfbthoRun): template object from inputs.
            gs_def_scan (tuple): parameter from input script.
            beta_type (str): Type of beta decay.

        Returns:
            hfbthoRun, None: ground state, if available
            list of hfbthoRun: finished even hfbthoRun objects (populated)
            list of hfbthoRun: unfinished even hfbthoRun objects (tasks)
            list of hfbthoRun: finished odd hfbthoRun objects (populated)
            list of hfbthoRun: unfinished odd hfbthoRun objects (tasks)
        """

        p = hfb_main.paths
        fsln = hfbthoRun.file_fam
        fwel = hfbthoRun.file_bin
        ftar = os.path.join(p.hfb_m, hfbthoRun.file_tar)

        # Initialize outputs
        hfb_gs = None
        even_fin, even_unf = [], []
        odd_fin, odd_unf = [], []

        # Ground state solution.hfb exists?
        recalc = (not os.path.exists(os.path.join(p.hfb, fsln)))
        if not recalc:
            try:
                hfb_gs = self.getHfbthoRun(p.rp(p.hfb), beta_type)
                ttime = self._getTotalHfbTime()
                if ttime is not None:
                    hfb_gs.soln_dict[u'Total_Time'] = ttime

                # Untarred data? (store even+odd as dummy in even_fin)
                if not os.path.exists(ftar):
                    for hfb_label in p.getLabels(p.hfb_m, tier2=True, prefix=p.out):
                        even_fin.append(self.getHfbthoRun(hfb_label, beta_type))
            except IOError:
                # Any problems, default to recalc
                hfb_gs = None
                even_fin = []
                recalc = True

        if recalc:
            if os.path.exists(ftar):
                with tarfile.open(ftar) as tf:
                    tf.extractall(path=p.hfb_m)
                # Avoid overwriting original data
                copy2(ftar, os.path.join(p.hfb_m, ftar.split('.tar')[0]+u'_og.tar'))

            # Even tasks according to input file
            etasks = hfbEvenDefList(hfb_main, *gs_def_scan)

            # Finished if solution file exists in rundir (solution file = wel if blocking)
            fsln_even = fsln
            if hfb_main.blocking[0][0] or hfb_main.blocking[1][0]:
                fsln_even = fwel

            even_unf, even_fin = self.getRemainingHfbTasks(etasks, fsln_even, beta_type)

            # Only proceed to populate odds if all evens are finished
            if (hfb_main.blocking[0][0] or hfb_main.blocking[1][0]) and not even_unf:
                # Odd tasks
                otasks = hfbOddList(even_fin, hfb_main, gs_def_scan[0])

                odd_unf, odd_fin = self.getRemainingHfbTasks(otasks, fsln, beta_type)

        return hfb_gs, (even_unf, even_fin), (odd_unf, odd_fin)

    #--------------------------------------------------------------
    def _getFamMeta(self, strength):
        """
        Parse outputs to get pnfam meta data for a given pynfam calculation.

        This method is meant for use inside the getFamState method.
        If data is missing, we can enter rerun mode by raising IOError.
        If reconstructing the values requires re-running pnfam, just return
        None and keep error values populated by famStrength.getMeta.

        Args:
            strength (famStrength): The strength object.

        Returns:
            populates attributes with famStrength.getMeta

        Raises:
            IOError
        """

        p = self.paths
        p_op = os.path.join(p.fam_m, strength.opname)
        fout = os.path.join(p.fam, strength.opname+u'.out')
        ftar = os.path.join(p.fam_m, strength.opname+u'.tar')
        # Try to get version, convs, times, from existing op.out
        try:
            strength.getMeta(fout)
        except IOError:
            # no fam_meta - not worth rerunning, keep error vals
            if not os.path.exists(p.fam_m):
                return
            # fam meta, and tarfile - raise error and enter rerun
            elif os.path.exists(ftar):
                raise IOError
            # fam meta, no tar, and no op directory - not worth
            elif not os.path.exists(p_op):
                return
            # fam meta, no tar, op dir, but it's empty - not worth
            elif not os.listdir(p_op):
                return
            # fam meta, no tar, op dir, and it's not empty - enter rerun
            else:
                raise IOError

    #--------------------------------------------------------------
    def getFamState(self, strength, fam_params):
        """
        Determine state of existing fam outputs for a single operator.

        Note:
            populates strength attributes.

        Args:
            strength (famStrength): template famStrength object
            fam_params (dict): fam parameters from input script

        Returns:
            list of pnfamRun: unifinished pnfamRun objects (tasks)
            list of pnfamRun: finished pnfamRun objects (populated)

        """

        p = self.paths
        p_op = os.path.join(p.fam_m, strength.opname)
        fout = os.path.join(p.fam, strength.opname+u'.out')
        fctr = os.path.join(p.fam, strength.opname+u'.out.ctr')
        ftar = os.path.join(p.fam_m, strength.opname+u'.tar')
        ffin = strength.opname+u'.log'
        unf_tasks, fin_tasks = [], []

        # OP.out and OP.ctr files exist?
        recalc = (not (os.path.exists(fout) and os.path.exists(fctr)))
        if not recalc:
            try:
                strength.readCtrBinary(p.fam, fctr)
                self._getFamMeta(strength)
                # Store existing solns for tarring later (need an extra check that
                # fam_meta exists for getLabels(fam_meta/OP) to not throw error)
                labels = []
                if os.path.exists(p_op):
                    labels = p.getLabels(p_op, prefix=p.out)
                if labels and not os.path.exists(ftar):
                    for fam_label in labels:
                        fin_tasks.append(self.getPnfamRun(strength.op, strength.k, fam_label))
            except IOError:
                # Any problems then default to recalc
                strength.str_df = None
                fin_tasks = []
                recalc = True

        if recalc:
            # If we have a tarfile, extract and treat as usual. We need to be
            # careful about MPI here, but it's not a common enough use case to
            # warrant including a barrier/bcast for typical runs.
            if os.path.exists(ftar):
                with tarfile.open(ftar) as tf:
                    tf.extractall(path=p.fam_m)
                # Avoid overwriting original data by renaming the original tar
                copy2(ftar, os.path.join(p.fam_m, strength.opname+u'_og.tar'))

            # Tasks based on input file
            tasks = strength.getFamList(p, fam_params)
            # Unfinished if rundir does not exist
            unf_tasks = [t for t in tasks if not os.path.exists(t.rundir)]
            # Unfinished if missing files needed to rescontruct fam object
            for t in tasks:
                if t in unf_tasks: continue
                try:
                    ft = self.getPnfamRun(t.op, t.k, t.label, t.ctr_params)
                    fin_tasks.append(ft)
                except IOError:
                    unf_tasks.append(t)

        return unf_tasks, fin_tasks


    #==============================================================#
    #             Compile data from many output files              #
    #==============================================================#
    def remakeHfbLog(self, label):
        """
        Attempt to remake a hfbtho mini logfile from existing outputs.

        Args:
            label (str): hfbtho run label
        """

        print(u"Warning: Missing logfile at "+str(label)+". Attemping"+\
              " to make a new one...")
        try:
            hfb_obj = self.getHfbthoRun(label, beta_type='-')
            hfblog = hfbLogger(hfbthoRun.file_log)
            hfblog.quickWrite(hfb_obj.soln_dict, hfb_obj.rundir)
            print(u"         Success!")
        except Exception:
            print(u"         Failed.")

    #--------------------------------------------------------------
    def gatherHfbLogs(self, pynfam_dir=None, skip=False):
        """
        Construct dataframe for all hfbtho solutions of a given pynfam run
        from individual hfbtho logfiles.

        Args:
            pynfam_dir (str): pynfam run directory name
                (default, self.paths.calc)
            skip (bool): Behavior when there's an error read log
                True = try to remake log, False = skip this solution
                (default, False)

        Returns:
            DataFrame

        Raises:
            IOError
        """

        if pynfam_dir is None: pynfam_dir = self.paths.calc
        top = os.path.dirname(pynfam_dir)
        logdata = []
        # (getHfbLabels excludes even core for blocking)
        for runlabel in self.paths.getLabels(self.paths.hfb_m,
                tier2=True, prefix=top):
            rundir = os.path.join(top, runlabel)
            runlog = hfbLogger(hfbthoRun.file_log)
            # Read the log file
            try:
                log_df = runlog.readLog(rundir)
            except IOError:
                if skip: continue
                self.remakeHfbLog(runlabel)
                log_df = runlog.readLog(rundir)

            # Check the contents for missing or multiple lines
            err_mult = None; err_miss = None
            for col in log_df:
                if len(log_df[col].values) > 1:
                    err_mult = 1
                elif len(log_df[col].values) == 0:
                    err_miss = 1
            if err_miss:
                print(u"Warning: Missing data in mini-logfile at "+\
                        str(runlabel)+u". Excluding this file.")
                continue
            elif err_mult:
                print(u"Warning: Multiple lines detected in mini-logfile at "+\
                        str(runlabel)+u". Using last line.")
                log_df = log_df.tail(1)

            # Store the logfile dataframe
            logdata.append(log_df)

        # Combine all the dataframes sorted on nuclei
        if logdata:
            master_df = pd.concat(logdata, sort=False).sort_values(\
                    by=[u'Z', u'Z-Blk', u'N', u'N-Blk'])
            return master_df.reset_index(drop=True)
        else:
            raise IOError(u"Error in gatherHfbLogs. No log data found.")

    #--------------------------------------------------------------
    def gatherMasterLog(self, path=None):
        """
        Gather logfiles for all pynfam solutions and write to master logfile
        in meta data directory.

        This will be hfb_soln logfiles, or beta_soln logfiles if available.

        Args:
            path (str, None): Top level directory name (default, self.paths.out).
        """

        log_name = u"logfile_master.dat"
        ml = lsLogger()

        logs=[]
        for d in self.paths.getLabels(top=path, prefix=True):
            pb = os.path.join(d, self.paths.subdir_beta)
            ph = os.path.join(d, self.paths.subdir_hfb)
            ml.filename=hfbthoRun.file_log
            try:
                # If we ran fam, tack on the rates
                if os.path.exists(os.path.join(pb,ml.filename)):
                    log_b=ml.readLog(pb)

                    ml.filename=u"beta.out"
                    log_r=ml.readLog(pb)
                    log_r=log_r.transpose()
                    log_r.drop(log_r.tail(1).index,inplace=True) # drop half lives
                    log_r.reset_index(drop=True, inplace=True)
                    # Hacky way to keep scientific notation for rates, but apply
                    # float format to everything else, by changing to strings
                    log_r = log_r.applymap(u'{:.6e}'.format)

                    log=pd.concat([log_b, log_r], axis=1, sort=False)
                else:
                    # If we only ran HFB
                    #try:
                    log=ml.readLog(ph)
                    #except IOError:
                    #    label = self.paths.rp(os.path.join(d, self.paths.subdir_hfb))
                    #    self.remakeHfbLog(label)
                    #    log=ml.readLog(ph)
            except IOError:
                # Keep label but fill with NaNs if error
                label = self.paths.rp(os.path.join(d, self.paths.subdir_hfb))
                log=pd.DataFrame({u'Label':[label]})
            logs.append(log)
        master = pd.concat(logs, sort=False)
        master.reset_index(drop=True, inplace=True)

        # Don't mess with the order (I think?)
        ml.filename = log_name
        ml.quickWrite(master, dest=self.paths.meta, fmt=False)

#    #--------------------------------------------------------------
#    def makeHfbFamFile(self, hfb_obj):
#        """
#        Use an existing hfbtho_output.hel file to remake a solution.hfb file
#        (RUN IT IN ITS OWN TMP DIRECTORY TO AVOID OVERWRITING EXISTING
#        NML AND HFBTHO OUTPUT FILES)
#        """
#        # Original labels
#        og_dir = hfb_obj.rundir
#        og_lab = hfb_obj.label
#
#        # Temporary copy, no constraints
#        tmp = deepcopy(hfb_obj)
#        tmp.setRestartFile(True)
#        tmp.setNmlParam({u'number_iterations':0}) # Only works for HFBTHOv3
#        tmp.setNmlParam({u'lambda_active':8*[0]})
#        tmp.setNmlParam({u'expectation_values':8*[0.0]})
#
#        # Temporary rundir with original wel file
#        tmp.label = os.path.join(og_lab,u'tmp')
#        if not os.path.exists(tmp.rundir):
#            os.mkdir(tmp.rundir)
#        copy2(os.path.join(og_dir, hfbthoRun.file_bin), tmp.rundir)
#
#        # Run, copy back to original dir and delete tmp dir
#        err = tmp.runExe(stdout_bool=True)
#        copy2(os.path.join(tmp.rundir, hfbthoRun.file_fam),
#              os.path.join(og_dir, hfbthoRun.file_fam))
#        rmtree(tmp.rundir)
#
#        return err
