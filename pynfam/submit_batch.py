# -------- Backwards Compatibility ---------
from __future__ import print_function
from __future__ import division
from builtins   import str
# -------------- Utilities -----------------
from shutil import copy2
import os
import subprocess
# ------------------------------------------

__version__ = u'2.0.0'
__date__    = u'2019-07-26'

#===============================================================================#
#                        Function submit_batch_script                           #
#===============================================================================#
def submit_batch_script(machine       ,
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
                        threads       ):
    """
    Perform checks on inputs, generate a slurm script, and submit a batch job.
    """


    # Machine settings
    listed_machines = [u'dogwood',u'quartz']

    cpus_per_node = {u'dogwood'  : 44,
                     u'quartz'   : 36
                    }
    queues        = {u'dogwood' : [u'debug_queue',u'528_queue', u'2112_queue'],
                     u'quartz'  : [u'pdebug', u'pbatch']
                    }
    time_limits   = {u'dogwood' : {u'debug_queue': [0,4,0],
                                  u'528_queue'  : [3,0,0],
                                  u'2112_queue' : [2,0,0]},
                     u'quartz'  : {u'pdebug'     : [0,0,30],
                                  u'pbatch'     : [1,0,0]}
                    }
    cpu_limits    = {u'dogwood' : {u'debug_queue': 88,
                                  u'528_queue'  : 528,
                                  u'2112_queue' : 2112},
                     u'quartz'  : {u'pdebug'     : 999999,
                                  u'pbatch'     : 999999}
                    }

    # Slurm checks (slurm does these for us I think but can't hurt)
    if machine not in listed_machines:
        print(u"Machine settings not recorded. Skipping slurm input checks.")
    else:
        if queue not in queues[machine]:
            raise ValueError(u"Invalid queue for machine")
        if tasks_per_node > cpus_per_node[machine]:
            raise ValueError(u"tasks per node > than cpus per node")
        if time[0] > time_limits[machine][queue][0] or \
           time[0] == time_limits[machine][queue][0] and \
               time[1] > time_limits[machine][queue][1] or\
           time[0] == time_limits[machine][queue][0] and \
               time[1] == time_limits[machine][queue][1] and\
               time[2] > time_limits[machine][queue][2]:
            raise ValueError(u"Time limit > queue time limit")
        if tasks_per_node*nodes > cpu_limits[machine][queue]:
            raise ValueError(u"CPU limit exceeded for requested queue")

    # File checks
    if not os.path.exists(python_exe):
        raise ValueError(u"python executable does not exist")


    # Batch header lines
    header_lines = []
    header_lines.append(u'#!/bin/bash\n')
    if bank:
        header_lines.append(u'#SBATCH -A '+bank+u'\n')
    if disk:
        header_lines.append(u'#SBATCH --license='+disk+u'\n')
    header_lines.append(u'#SBATCH -p '+queue+u'\n')
    header_lines.append(u'#SBATCH -J '+jobname+u'\n')
    header_lines.append(u'#SBATCH -o '+u'slurm_%A.out'+u'\n')
    header_lines.append(u'#SBATCH -t '+str(time[0])+u'-'+str(time[1]).zfill(2)+u':'+str(time[2]).zfill(2)+u'\n')
    header_lines.append(u'#SBATCH -N '+str(nodes)+u'\n')
    header_lines.append(u'#SBATCH --ntasks-per-node='+str(tasks_per_node)+u'\n')
    header_lines.append(u'\n')

    # Batch extra settings
    precmd_lines = []
    if openmp:
        precmd_lines.append(u'export OMP_NUM_THREADS='+str(threads)+u'\n')
        precmd_lines.append(u'export KMP_STACKSIZE=64M\n')
    else:
        precmd_lines.append(u'unset OMP_NUM_THREADS\n')
    precmd_lines.append(u'ulimit -s unlimited\n')
    precmd_lines.append(u'\n')

    # Batch execute lines
    cmd_lines = []
    cmd_lines.append(mpi_cmd+u' '+python_exe)
    cmd_lines.append(u'\n')

    # Write the batch script
    contents = header_lines + precmd_lines + cmd_lines
    with open(batch_name, u'w') as job:
        for lines in contents:
            job.write(lines)

    # Copy the python executable with check only mode as input
    exe_copy = python_exe+u'_copy'
    copy2(python_exe, exe_copy)
    with open(exe_copy, u'r') as py_file:
        py_data = py_file.readlines()

    key = u'min_mpi_tasks'
    line = [i for i, x in enumerate(py_data) if x.find(key) > -1 ][0]
    min_mpi_tasks = int(py_data[line].split(u',')[0].split(u':')[-1])

    key = u'pynfam_mpi_calc'
    lastline = [i for i, x in enumerate(py_data) if x.find(key) > -1 ][-1]
    py_data[lastline] = py_data[lastline].split(u'False')[0] + u'True)\n'
    with open(exe_copy, u'w') as py_file:
        py_file.writelines(py_data)

    # Run the script with check only mode
    proc = subprocess.Popen([exe_copy], stdin=None,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    stdout, stderr = proc.communicate()
    print(stdout, stderr)

    os.remove(exe_copy)
    if u'Input check complete. No errors were detected.' not in stdout:
        return
    elif tasks_per_node*nodes < min_mpi_tasks:
        print(u"To few mpi processes reserved. Total tasks must be >="+\
              " min tasks per calc.")
        return

    # Submit batch script
    try:
        proc = subprocess.call(u'sbatch '+batch_name, shell=True)
    except OSError:
        print(u"Error submitting job. Is sbatch a valid command on this machine?")
        return
