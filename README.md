# Overview

This repository contains the python package *pynfam* which includes the functions,
classes, and methods for running large scale beta decay caluclations with the fortran
codes hfbtho and pnfam. It contains an executable script *run_pynfam.py* through which
the user can supply all necessary inputs for one or more beta decay calculations,
which are parallelized with mpi4py.

Currently, runs of fortran codes are handled by the *fortProcess* parent class in the
*fortran* sub-package. A run of an executable (i.e. the *runExe* method) consists of
creating a uniquely named directory, generating all necessary input files, then executing
the fortran program from within this directory using python's subprocess.Popen() method.
The fortran executables themselves are the only extra files needed to use this package.
(Future versions may replace this functionality with python bindings.)

### Submodules

A hard copy of the HFBTHO source code is included for pynfam users. However, in addtion a
git submodule linking to the original HFBTHO repositories also exists, allowing simultaneous
development of both projects. Users need access to the HFBTHO repository to use this feature.

After cloning the repository the submodule directory will still be empty. To populate the
directory users must run:
```git submodule init; git submodule update```
At this stage the the submodule will not be on any working branch (detached HEAD state), so
no changes will be tracked. To work on the project from the pynfam repository, be sure to
checkout a branch first.

For a short tutorial on using git submodules, see:
https://git-scm.com/book/en/v2/Git-Tools-Submodules

---

# Installation

Installation of the pynfam package can be performed with pip by calling install on the
directory which constains the setup.py file:

``` pip install /path/to/pynfam/ ```

If the user is not working in a virtual environemnt (see below) and does not have root
access to the python intsallation, doing ```pip install --user``` instead will install
the package locally, typically under ```~/.local/lib/pythonX.X/site-packages```.

### Dependencies

The *pynfam* package uses the following modules outside the standard library (all
can be found in the PYPI repository):

* numpy, pandas, scipy, mpi4py, f90nml, future

The package also depends on the HBFTHO and pnFAM executables. Compiling HFBTHO and
pnFAM requires the following libraries and headers (see source for more details):

* HFBTHO:
    * LAPACK, BLAS

* pnFAM (SERIAL VERSION ONLY!):
    * LAPACK, BLAS
    * GNU scientific library (gsl - this dependency will soon be deprecated and not required.)

The openmp versions of the programs should be compatible with the *pynfam* package
(though this has not been tested yet), but the versions compiled with MPI active should
NOT be used. All mpi parallelization is handled by python via mpi4py.

### mpi4py

A few errors encountered with mpi4py are worth noting here.

1. mpi4py must be installed and linked to the same mpi implementation currently active. This can
   cause issue e.g. on clusters where mpi4py was installed with one implementation but the user
   has a different one loaded in their modules. Sometimes the user must check their environment
   paths to ensure the correct MPI implementation will be found, and also mpi4py can be found by
   python (e.g. in the PYTHONPATH variable).
2. This package calls fork() in an mpi environment via pythons subprocess package, which can be
   dangerous. Supposedly the versions of subprocess imported in this package are fork safe, and
   the fortran programs being executed do not themselves contain calls to MPI, so all should work
   fine. However, we have enountered and never were able to resolve some errors related to this
   using OpenMPI 3.0.0. Our recommendation is to use the latest version of mpi4py with mvapich2.

---

# Workflow

### Parallelization

The workflow for a single pynfam calculation involves at most four serial stages
that each generate a list of "fortran objects" as tasks to be executed in parallel.

1. HFBTHO runs for even-even HFB solutions with several deformations
2. * Feature: HFBTHO re-runs for non-converged even-even HFB solutions
3. HFBTHO runs for all blocking candidates of all even solutions
4. pnFAM runs for each EQRPA point per operator

The MPI workflow is designed for load balancing, and offers the best performance if the
number of MPI processes is much larger than the number of beta decay calculations. This is
particularly useful for large scale calculations involving many beta decay calculations run
from a single input script (see override_settings below). MPI processes are split into two
communicators, one with a small number "master" processes (<= number of beta decay calculations),
and another with a large number of "worker" processes. Each master process will execute pynfam
code for a single beta decay calculation, which will be computed in parallel if multiple master
processes are requested. As each master process generates the task lists above, the tasks are
distributed to the pool of worker processes, which is shared by all master processes. In this
way the work load is evenly distributed among the workers, even if the number of tasks
for different beta decay calculations differs significantly.

This workflow requries one worker to be reserved as a "leader" to handle details of the
communications, therefore the resources are organized as:

* comm_size = nr_masters + nr_workers + 1_lead_worker

The code can also be run serially with a single process, but the MPI execution therefore requires
at least 3 processes. Some flexibility is provided for the user to determine the size of the
worker pool and the number of master processes via the input parameter nr_parallel_calcs
(see pynfam_inputs below).

### Runtime Behavior

There are several non-fatal error checks in the pynfam which allow a large scale calculation
to continue while skipping an individual calculation which encountered the error. Examples
include a non-converged HFB solution, a segfault or other error encountered by the fortran
executable, etc.

---

# Inputs

Inputs for a given calculation are specified in the executable *run_pynfam.py* script. A copy
of it is stored in the output directories containing the final results for easy reference.
The run_pynfam.py input file is split up into two main input sections, specified as the
dictionaries pynfam_inputs and override_settings which are discussed below.

### pynfam_inputs

* **directories (Str)**

    * 'outputs': Name of the directory for pynfam outputs
    * 'exes'   : Path to a single directory containing the HFBTHO and pnFAM executables
    * 'scratch': Path to location for pynfam outputs directory

* **nr_parallel_calcs (Int or None)**

     In the case of multiple pynfam calculations (see override_settings for details on
     how to specify multiple calculations), each one may be computed in parallel. This
     parameter determines the number of mpi processes that will be dedicated to running
     the set of pynfam calculations. If less than the total number of calculations, the
     calculations will be executed by these processes in a round-robin fashion. The number
     of mpi worker processes dedicated to running tasks is comm_size - nr_parallel_calcs - 1.
     If nr_parallel_calcs is 0 or None, it defaults to the number of pynfam calculations.

* **rerun_mode (Int)**

    This parameter only takes effect if the output directory already exists.

    * 0 - Full Restart: Uses the input file to determine the state of pre-existing data
          and pickup where the calculation left off if the data is incomplete. Non-converged
          HFB solutions count as incomplete, and will be re-run (unconstrained) to allow
          more iterations. Sub-directories must have original naming conventions.
    * 1 - FAM Restart: Determines the number of calculations based on number of existing directories.
          All existing directories must contain a solution.hfb file, and the calculation begins in
          the FAM stage, ignoring any HFB inputs. Restart behavior for pre-existing FAM data is
          identical to Full Restart. Sub-directory names do not matter. Tuple override settings must
          have length equal to number of existing directories.

* **gs_def_scan (Tuple)**

    This setting activates our built in method for finding axially deformed ground states.
    The input has the form (kickoff, (b2_1,b2_2,...)). For each value of deformation beta2
    (b2), hfbtho is run with the following inputs:

     * basis_deformation = b2
     * beta2_deformation = b2
     * lambda_active     = [0, kickoff, 0, 0, 0, 0, 0, 0]
     * lambda_active     = [0, Q2(A,b2), 0, 0, 0, 0, 0, 0]
     * Q2(A,b2)          = sqrt(5/pi)/100 * b2 * A^(5/3)

    The kickoff parameter maintains its original behavior as in hfbtho:

    * kickoff =  0 (or b2 iterable is empty): gs_def_scan feature is not used
    * kickoff = -1: Applies constraint for first few iterations only
    * kickoff = +1: Applies constraint for entire calculation

    Additionally it has options for blocking calculations of odd nuclei:

    * kickoff = +/-1: even calculations are constrained, odd are unconstrained
    * kickoff = -2: kickoff=-1 behavior for even and odd calculations
    * kickoff = +2: kickoff=+1 behavior for even and odd calculations

    Note: if active, these values will supercede inputs from 'override_settings'.

* **dripline_mode (Int)**

    Serially calculate out to the neutron dripline from the initial nucleus.

    * 0    - off
    * +/-1 - increase N by 1 until 1Sn = E(N+1)-E(N) < 0.
    * +/-2 - increase N by 2 until 2Sn = E(N+2)-E(N) < 0.

    The sign indicates direction to block. Blocking in the initial nucleus is
    independent of this sign.

* **ignore_nonconv (Int)**

    Define how to handle non-converged (nc) hfb solutions.

    * 0 - stop for any nc soln
    * 1 - discard odd nc solns (stops if ANY even or ALL odd are nc)
    * 2 - discard all nc solns (stops if ALL even or ALL odd are nc)
    * 3 - use nc solns

* **fam_contour (Str)**

    Specify the the type of energy contour to compute strength along. These come in 2 flavors:

    * Open contours: Used for computed strength functions. The imaginary part of the energy determines
      the half-width of the Lorentzian smeared strength peaks. Different open contours allow for a
      non-constant smearing, which gives greater flexibility in specifying the density of the energy
      grid and can save computational resources. Note that if the imaginary part is large,
      1/(x+iy) = P(1/x) + i /pi \delta(x) is no longer valid and the QRPA sum rule breaks.

    * Closed contours: Used for computing rates, but do not provide details of the strength function.
      This is done via a complex contour integration of the phase space weighted strength (or shape
      factor, which combines phase space weighted strength from allowed + first forbidden operators).

    All contours are defined on an energy interval along the real axis, while other settings specific
    to the type of contour define the behavior in the complex plane, spacing, number of points, etc.
    A contour's defining settings are listed in the config.py file, for details see the
    override_settings section below.

    A list of the available contours is given below. Contours with * adjust number of points until
    the desired profile fills the energy interval. Those without have a specified number of points.
    Where applicable, the spacing is defined in terms of the half-width via the de_hw_ratio setting.
    If de_hw_ratio > ~1, the contour grid is too coarse, and may miss peaks in the strength function.

    * None      : Skip the fam calculation, only compute the hfb ground state.
    * 'CIRCLE'  : (Closed) Circular contour
    * 'CONSTl'  : (Open)   Constant defined by nr_points (linspace)
    * 'CONSTr'  : (Open*)  Constant defined by spacing  (range)
    * 'EXP'     : (Open*)  Exponential profile
    * 'MONOMIAL': (Open*)  Monomial profile
    * 'FERMIs'  : (Open*)  Static Fermi function profile
    * 'FERMIa'  : (Open)   Adaptive Fermi function profile. Changes the contour itself to fill an
       energy interval with a specified number of points, prioritizing small width at small energy.
       This occurs in stages, between two limiting cases.

        1. Max points limiting case: CONSTl at a min half-width.
        2. Raise right side of contour to form a fermi function type profile.
        3. If right side reaches max half-width, raise left side of contour until max half-width.
        4. Min points limiting case: CONSTl at max half-width
        5. If the interval is still too large after stage 4, the number of points is increased until
           CONSTl at max half-width fills the interval.

* **beta_type (Str)**

    A string '-' or '+' for beta minus or beta plus operators

* **fam_ops (Str or Tuple)**

    pnfam calculates the hfb linear response to allowed and first-forbidden beta decay operators
    (external fields). The operators are separated by how they change K=(-J,...,J), thus each
    (Operator, K) combination will have it's own strength function. For even-even parents, and odd
    parents computed in the equal filling approximation, only K>=0 contributions need to be computed.

    The following operators are implemented in pnfam:

    * 'F'   : (J=0) Fermi operator (identity)
    * 'GT'  : (J=1) Gamow-Teller operator (pauli matrix sigma)
    * 'R'   : (J=1) position vector (r)
    * 'P'   : (J=1) momentum over i (p/i)
    * 'RS0' : (J=0) r and sigma coupled to J=0
    * 'RS1' : (J=1) r and sigma coupled to J=1
    * 'RS2' : (J=2) r and sigma coupled to J=2
    * 'PS0' : (J=0) p/i dotted with sigma (note: this is not equal to p/i and sigma coupled to J=0)

    This package wraps sets of operators for convenient beta decay calculations. Available calculations
    include total, total allowed, total first-forbidden, or transition by J-pi. Calculation names,
    number of strength functions computed (n), and operators considered, are listed below.
    (note only K>=0 operators are computed):

    * 'All'       : n=14, all operators
    * 'Allowed'   : n= 3, [F,GT]
    * 'Forbidden' : n=11, [R,P,RS0,RS1,RS2,PS0]
    * '0+'        : n= 1, [F]
    * '1+'        : n= 2, [GT]
    * '0-'        : n= 2, [RS0,PS0]
    * '1-'        : n= 6, [R,P,RS1]
    * '2-'        : n= 3, [RS2]
    * (Operator,K): n= 1, manually specify 1 strength function with a tuple of form (str, int)
      where str is one of the 8 operators listed above and int is the K value.

### override_settings

Default calculation settings are defined in the config.py file in the pynfam package, and
can be overridden here. To run multiple calculations in parallel, a tuple of values be
specified for any parameter. This will run len(tuple) pynfam calculations, with the i'th
calculation using the i'th tuple element as the setting. Standard (Non-tuple) parameters are
applied to every calculation. All tuple inputs must have the same length. The len(tuple) in
conjunction with the 'min_mpi_tasks' value determines the number of sub-communicators.

* **hfb (Dict)**

    HFBTHO namelist parameters to override.

* **ctr (Dict)**

    Parameters defining the fam_contour. All contours have common settings `energy_min`,
    `energy_max`, and `hfb_emin_buff`. The latter setting defines a buffer to increase the default
    interval computed from properties of the HFB ground state by reducing the default value of
    energy_min (see below). If energy_min is specified, this value will supercede the above behavior.

    The default energy interval computed from the HFB ground state is:

    * energy_min = min(0, E_gs - hfb_emin_buff)
    * energy_max = (E_gs + Q) = MnH - lambda_p + lambda_n

        - E_gs = min( min(E_p+E_n), min(E_p - E_n_blocked), min(E_n-E_p_blocked) ) where applicable
        - MnH = M_neutron - M_Hydrogen
        - lambda = HFB approximation to the Fermi energy

    If an imaginary value is supplied as the `half_width` setting for applicable contours, strength
    is on  [energy_min*1j, energy_max*1j] a distance Imag(half_width) from the imaginary axis.

* **fam (Dict)**

    pnFAM namelist parameters to override. The default values include EDF coupling overrides
    for values fit by Mustonen 2016 (Table 2 set 1A from PRC 93 014304). To use the pnfam code's
    default values (computed from the interaction) the user can override these values by
    supplying the empty string ''. This leaves the namelist entry for the parameter blank.

* **psi (Dict)**

    Parameters defining how phase space integrals are computed. There are two main methods for
    approximating the phase space integrals in the complex plain: rational function
    interpolation (RATINT) and polynomial least squares fit (POLYFIT). The integrals are
    calculated with gauss legendre integration with 'psi_glpts' number of quadrature points.

---

# Outputs

The output directory tree is structured as detailed below, where parenthesis indicate files.
pynfam generates logfiles along the way with key outputs. Key hfb+fam+beta outputs for every
pynfam calculation are compiled into one master logfile in the meta data directory.

        outputs
          |- meta
          |    |- (copied_inputs_and_slurm_files+master_logfile)
          |
          |- calc_label
               |- hfb_soln
               |    |- (copied_ground_state_output_files+mini_logfile)
               |    |- hfb_meta
               |         |- (logfile_for_all_hfb_runs)
               |         |- (tarfile_of_all_hfb_run_dirs)
               |              |- hfb_label_even
               |              |    |- (raw_output_files)
               |              |- hfb_label_odd
               |                   |- core
               |                       |- (raw_output_files)
               |                   |- candidate_label
               |                       |- (raw_output_files)
               |
               |- fam_soln
               |    |- (strength_output_text_and_binary_files_per_operator)
               |    |- fam_meta
               |         |- (representative_logfile_for_pnfam_runs)
               |         |- (tarfiles_of_all_pnfam_runs_per_operator)
               |              |- fam_point_label
               |                   |- (raw_output_files)
               |
               |- beta_soln
                    |- (beta_decay_output_files+mini_logfile)
                    |- beta_meta
                         |- (non_essential_output_files)
