# runmat-execution

`runmat-execution` is the dependency-light portable contract authority for RunMat execution. It owns canonical program identity, strongly typed execution IDs, user-visible handle descriptors, resource and lifecycle vocabulary, inert cross-boundary value schemas, protocol envelopes, limits, and validation.

It deliberately owns no scheduler, runtime `Value`, MATLAB behavior, threads, processes, filesystem, network transport, package solver, authentication, tenancy, or billing. Live value conversion belongs to `runmat-runtime`; scheduling belongs to `runmat-execution-runner`; native and browser mechanisms belong to host adapters.
