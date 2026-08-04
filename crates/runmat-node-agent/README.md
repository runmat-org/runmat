# RunMat node agent

`runmat-node-agent` is the independently deployed native service for an
enrolled RunMat execution node. It maintains node credentials, reports signed
coarse inventory, validates allocation fencing and local capability policy,
launches contained RunMat driver/worker processes, drains, and cleans up every
owned process on shutdown.

The agent does not schedule MATLAB tasks and never grants itself work. The
Server offers coarse allocation leases; the portable execution driver remains
the only fine-grained scheduler.

## Enrollment and foreground operation

Create a single-use enrollment token with `runmat cluster enroll`, then enroll the node with the exact Server and RunMat executable:

```bash
runmat-node-agent \
  --server https://api.runmat.com \
  --runmat /usr/local/bin/runmat \
  enroll --token "$RUNMAT_NODE_ENROLLMENT_TOKEN"

runmat-node-agent \
  --server https://api.runmat.com \
  --runmat /usr/local/bin/runmat \
  run
```

The enrollment token is consumed once and is never written to the service definition. The resulting rotating node credential and endpoint identity stay in the private state directory. `inventory` prints the bounded, content-free capabilities the agent will advertise, and `rotate-credential` rotates the enrolled control-plane credential.

## Operating-system service

`service install` validates the complete configuration, persists a canonical JSON file, installs the native boot service, and starts it. It uses systemd on Linux, a system LaunchDaemon on macOS, and a LocalSystem Windows service. Run it with root or Administrator privileges:

```bash
sudo runmat-node-agent \
  --server https://api.runmat.com \
  --runmat /usr/local/bin/runmat \
  service install

runmat-node-agent \
  --server https://api.runmat.com \
  --runmat /usr/local/bin/runmat \
  service install --dry-run
```

The dry-run and `service print` forms emit the exact machine-readable file/command plan without changing the host. The installed service invokes `runmat-node-agent --config <platform-config> run`; the JSON config contains no enrollment token, node credential, workload key, or registry credential. CLI flags can override a loaded config for foreground diagnosis.

`service uninstall` stops and removes the boot service and its non-secret configuration. It deliberately preserves the private state directory and enrolled credential so an operator can reinstall without silently changing node identity. Revoke the node in RunMat before deleting that directory when retiring a host.

The service definitions use restart-on-failure behavior. Linux additionally enables systemd filesystem/process hardening and grants write access only to the node-agent state root. Windows removes inherited ACLs from the state directory and grants access only to SYSTEM and Administrators.
