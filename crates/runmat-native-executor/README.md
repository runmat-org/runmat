# runmat-native-executor

`runmat-native-executor` owns the reusable host for verified RunMat Native IR. It binds one exact Native IR function set to process-linked or dynamically owned machine-code entrypoints, then owns invocation state, semantic callbacks, GC roots, workspace publication, cancellation, suspension, deoptimization, and on-stack replacement for the lifetime of each call.

The crate does not compile machine code, choose when compilation happens, publish session versions, or own frontend and bytecode fallback policy. `runmat-jit` supplies dynamically owned entrypoints and adaptive profiles; `runmat-aot-runtime` supplies linked entrypoints retained by the process image. Both paths use the same executor and runtime-owned native ABI.

`NativeExecutable` makes code ownership explicit. A dynamically allocated executable retains an opaque `Send` owner alongside its entrypoint table and measured code bytes. A linked executable records no reclaimable code-memory owner. `NativeExecutor::bind` verifies the Native IR and requires the executable's function identities to match it exactly before any entrypoint can run.

Invocation construction uses `NativeInvocationRequest`, while executor construction uses `NativeExecutorOptions`. These cohesive contracts keep capture, workspace, deoptimization, OSR, representation-profile, resume-point, and timing state explicit without parallel argument lists or lint exceptions.
