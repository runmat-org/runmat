# Python Interoperability

RunMat can call CPython modules, functions, classes, and objects from MATLAB-syntax source on native hosts. Python objects keep their interpreter identity and belong to the RunMat session that created them.

## Configure Python

RunMat discovers CPython from `executable`, a requested `version`, and the host `PATH`. The supported range is checked before the interpreter starts:

```toml
[runtime.foreign.python]
version = "3.12"
minimum_version = "3.9"
maximum_version = "3.13"
execution_mode = "in_process"
```

`version`, `minimum_version`, and `maximum_version` use `major.minor` form. `executable` selects a specific interpreter. The default `in_process` mode gives Python and NumPy direct access to compatible RunMat host buffers. `out_of_process` runs CPython through a child mode of the same `runmat` executable; it contains interpreter crashes and supports `terminate`, but values cross a process boundary.

In-process sessions share CPython's process-wide interpreter, including its import cache and module search path. Separate RunMat sessions do not isolate Python module globals, native extensions, or Python-created threads. Use this mode for trusted code; choose `out_of_process` when you need a separate interpreter process. Process isolation is not an operating-system sandbox.

`pyenv` reports the selected executable, library, version, status, and execution mode without starting the persistent Python session. Discovery can launch a short-lived interpreter probe. Its `Version` and `ExecutionMode` options can change an environment only before startup. An in-process CPython instance cannot be unloaded safely; `terminate(environment)` is therefore available only for an out-of-process environment.

## Call Python code

Use the `py` namespace with ordinary dotted names:

```matlab
root = py.math.sqrt(81);
items = py.list({int32(10), int32(20)});
first = items(1);
items(2) = int32(42);
closeEnough = py.math.isclose(1, 1.000001, pyargs("rel_tol", 0.00001));
```

RunMat resolves Python modules, classes, constructors, functions, methods, attributes, indexing, assignment, and iteration through the same session-owned foreign-object boundary. `pyargs` supplies keyword arguments after positional arguments.

`pyrun` evaluates source in a persistent Python workspace and can exchange named inputs and outputs:

```matlab
answer = pyrun("answer = input_value + 2", "answer", "input_value", int32(40));
```

`pyrunfile` executes a Python file with the same named input and output model. File execution has its own script workspace rather than adding definitions to the persistent `pyrun` workspace.

## Values, arrays, and callbacks

RunMat converts logical and numeric scalars, all eight fixed-width integer classes, complex values, strings and characters, cells, structures, Python lists, tuples and dictionaries, and Python `None`. Values without a lossless RunMat representation remain session-owned Python objects.

Compatible dense host arrays become read-only NumPy views over RunMat's copy-on-write storage during an in-process call. The view retains the host allocation for its lifetime. A writable request, unsupported layout or dtype, device-resident value, or out-of-process call uses an explicit copy. NumPy arrays returned to RunMat preserve supported dtypes and column-major shapes; unsupported or non-native layouts are normalized at the boundary.

A RunMat function handle can be passed to Python and called synchronously. The callback returns to its originating session with the active cancellation state. Python exceptions retain their Python type, message, explicit cause, and traceback frames in the resulting RunMat error.

Python waits for the callback result while RunMat can continue its suspended execution. A callback can call back into the same Python session while it is active. Cancelling an in-process operation stops RunMat from awaiting it, but work already admitted to the interpreter may continue; project replacement requires verified retirement. Cancelling an out-of-process call may retire its child and invalidate its Python objects.

Scalar `datetime` values map to naïve Python `datetime.datetime` values and scalar `duration` values map to `datetime.timedelta`. Arrays use NumPy `datetime64[us]` and `timedelta64[us]`. RunMat's current datetime storage may differ by a few microseconds after a round trip; Python timedeltas returning to RunMat are truncated to milliseconds. A timezone-aware Python datetime stays an opaque Python object because RunMat datetime values do not currently carry a timezone field.

## Package Python wheels

Declare a wheel that must travel with the project in `runmat.toml`:

```toml
[python-artifacts.analysis]
path = "python/analysis-1.0-py3-none-any.whl"
module = "analysis"
```

RunMat freezes the wheel bytes with the resolved package graph, verifies the digest again at every handoff, and records the exact wheel and CPython environment identities in compiled and distributed products. Standalone executables embed the canonical wheel bundle. At startup, RunMat validates and materializes the wheel into a private import root before interoperability admission. Standard wheel `purelib` and `platlib` layouts are supported.

An in-process interpreter can retain wheel code through native extensions, module globals, or Python-created threads after a RunMat session ends. Exposed wheel files therefore remain process-owned; retiring a session does not reclaim them. Managed imports are removed only while their original module identity still occupies the name. If cleanup cannot verify ownership, replacement may remain blocked rather than clearing unrelated modules. Out-of-process wheel bytes remain protected through verified child retirement and are reclaimable after their other product owners release them.

Artifact names must be unique across the resolved package graph. RunMat does not retarget native wheel contents: extension modules must match the exact CPython ABI and platform recorded for the product. Use target-specific package dependencies or artifacts for different native targets. Wheel code has the permissions of its Python host: use `out_of_process` for crash containment, and apply operating-system sandboxing when untrusted code needs a stronger security boundary.

## Native compilation, distribution, and browsers

Python calls are part of executable reachability. Native compilation embeds the required environment and wheel identities together with the wheel contents. Remote submissions carry the same content-addressed artifacts, and a worker must satisfy the declared Python adapter contract before execution.

Explicit local module paths do not grant a standalone or remote host access to those directories. Declare project wheels for code that must travel with the program. RunMat rebases certified installed-wheel paths to the receiving host's materialization; additional local paths without authenticated materialization are rejected before installation, rather than silently removed.

Browser and WebAssembly sessions do not embed CPython. Calls through `py.*`, `pyenv`, `pyrun`, `pyrunfile`, and `terminate` require a native Python host. RunMat does not silently substitute another Python implementation. `pyargs` only constructs a local keyword-argument bundle, so it works without a Python host, including in WebAssembly; using that bundle to call Python still requires the native host. A future browser host bridge must be declared explicitly by both the product and host.
