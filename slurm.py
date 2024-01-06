#!/usr/bin/env python
from __future__ import unicode_literals
from pynfam import submit_batch_script

# This script requires the same exact inputs as writing a batch script by hand,
# it just checks that the inputs are correct and the job will run, before submitting
# the job to the scheduler. (It actually runs the 1st part of the mpi workflow).

machine       = 'dogwood'
jobname       = 'pynfam'
queue         = '528_queue'
time          = [1,0,0]
nodes         = 6
tasks_per_node= 44
bank          = '' # Quartz: nucfiss
disk          = '' # Quartz: lustre2
mpi_cmd       = 'mpirun'
python_exe    = './run_pynfam.py'
batch_name    = 'slurm_'+machine+'.sh'
openmp        = False
threads       = 1

if __name__ == '__main__':
    submit_batch_script(machine       ,
                        jobname       ,
                        queue         ,
                        time          ,
                        nodes         ,
                        tasks_per_node,
                        bank          ,
                        disk          ,
                        mpi_cmd       ,
                        python_exe    ,
                        batch_name    ,
                        openmp        ,
                        threads       )
