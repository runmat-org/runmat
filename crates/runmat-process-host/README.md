# runmat-process-host

`runmat-process-host` contains native, domain-neutral child-process and local IPC
mechanics shared by RunMat composition roots and worker adapters.

The crate owns command/environment policy, child process-tree lifecycle, bounded
stderr capture, bounded length-prefixed frames, stdio/local endpoint descriptors,
generic handshake constraints, hidden host-mode recognition, and portable shared
memory descriptors. It does not own MATLAB, testing, scheduling, package, job, or
Server semantics.

Local process hosting deliberately has no TCP endpoint. Network transports belong
to the execution transport layer and require their own authenticated composition.
