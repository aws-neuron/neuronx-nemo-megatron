#!/usr/bin/env python3
# Helper script to create a list of worker pods participating in
#   a distributed training job based on current worker's hostname
#   and the OMPI world size
import re
import os
##os.environ['PMIX_HOSTNAME'] = "test-mpi-dumpenv-worker-1"
##os.environ['OMPI_COMM_WORLD_SIZE'] = "2"
this_host = os.environ.get("PMIX_HOSTNAME", "")
s = re.search(r"^(.*-worker)-\d+", this_host)
if not s:
    raise Exception("Error: This script should be run via mpirun on EKS")
host_prefix = s.group(1)
world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0"))

hosts = []
for x in range(world_size):
    hosts.append(f"{host_prefix}-{x}.{host_prefix}.default.svc")

print(" ".join(hosts))
