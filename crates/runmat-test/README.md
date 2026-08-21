# runmat-test

`runmat-test` is RunMat's portable test-domain authority. It defines stable identities, immutable plans, lifecycle and teardown semantics, qualification records, deterministic events and replay, canonical results and attempts, bounded execution context commands, executor ports, and the versioned worker protocol.

The crate intentionally has no dependency on RunMat Core, the VM, runtime builtins, the CLI, operating-system process APIs, filesystems, browser APIs, JavaScript, Desktop, or RunMat Server. Host executors and coordinators implement its narrow ports.
