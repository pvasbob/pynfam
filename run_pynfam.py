#!/usr/bin/env python
from __future__ import unicode_literals
from pynfam import pynfam_mpi_calc

pynfam_inputs = {
 'directories': {
     'outputs' : 'outputs',
     'exes'    : './exes',
     'scratch' : './'
     },

 'nr_parallel_calcs': None,

 'rerun_mode': 0,

 'hfb_mode': {
     'gs_def_scan'    : (-2, (-0.2,0.0,0.2)),
     'dripline_mode'  : 0,
     'ignore_nonconv' : 0
     },

 'fam_mode': {
     'fam_contour': 'CIRCLE',
     'beta_type'  : '-',
     'fam_ops'    : 'All'

     }
}

override_settings = {
 'hfb' : {'proton_number'    : 22,
          'neutron_number'   : 30
          },

 'ctr' : {'nr_points' : 50},

 'fam' : {},

 'psi' : {}
}


if __name__ == '__main__':
    pynfam_mpi_calc(pynfam_inputs, override_settings, check=False)
