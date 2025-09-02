#!/bin/bash

#flux: -N 2
#flux: -n 2
#flux: --exclusive  
#flux: -t 10m
#flux: -q pdebug

echo "Job started on $(date)"
echo "Testing basic connectivity:"

# Test basic commands first
flux run -N 2 -n 2 /bin/bash -c "echo 'Node:' \$(hostname) 'Python:' \$(which python || echo 'not found')"

echo "Job completed on $(date)"
