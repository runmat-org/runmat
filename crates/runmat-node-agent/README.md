# RunMat node agent

`runmat-node-agent` is the independently deployed native service for an
enrolled RunMat execution node. It maintains node credentials, reports signed
coarse inventory, validates allocation fencing and local capability policy,
launches contained RunMat driver/worker processes, drains, and cleans up every
owned process on shutdown.

The agent does not schedule MATLAB tasks and never grants itself work. The
Server offers coarse allocation leases; the portable execution driver remains
the only fine-grained scheduler.
