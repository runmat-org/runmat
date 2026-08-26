# Native Library Interfaces

RunMat can call functions in native shared libraries through the legacy `loadlibrary` and `calllib` APIs or the modern `clib.*` namespace. Both surfaces use the same normalized prototype metadata, value conversions, callback machinery, pointer lifetime tracking, and session-owned loader.

## Preparing An Interface

Prepare an interface from a C header and one shared library:

```matlab
clibgen.buildInterface("include/filters.h", ...
    "Libraries", "native/libfilters.dylib", ...
    "InterfaceName", "filters");
```

RunMat writes a canonical `.runmat.json` manifest beside the library. The manifest records the normalized declarations, target triple, exact library digest and length, and a content-derived interface identity. Moving the manifest and matching library together does not change that identity; changing either file does.

Header preparation uses a compiler frontend so structures, aliases, enumerations, pointers, arrays, calling conventions, and platform layout come from the target ABI. RunMat does not infer an interface from function calls in source code.

## Declaring Prepared Interfaces

Declare each prepared interface in the package that owns it:

```toml
[native-interfaces.filters]
manifest = "native/libfilters.dylib.runmat.json"
library = "native/libfilters.dylib"
```

Both paths are relative to that package's manifest and cannot escape the package root. The declaration name must match the interface name in the prepared manifest. Dependencies can declare their own interfaces; project resolution collects the complete set from the frozen package graph and rejects duplicate public interface names.

The interface manifest and library bytes participate in the package tree identity. `--locked` and `--frozen` therefore detect changes to either artifact just as they detect source changes. Publish native packages with `runmat package publish --allow-native`, and include target-specific libraries only for targets the package supports.

## Calling Functions

The modern namespace uses the declared interface name:

```matlab
output = clib.filters.apply(input, int32(3));
```

The legacy surface loads an interface under a session alias and calls through the same runtime adapter:

```matlab
loadlibrary("native/libfilters.dylib", "include/filters.h", "alias", "filters");
output = calllib("filters", "apply", input, int32(3));
```

Pointers returned with borrowed ownership remain tied to the loaded library and session. Owned pointers require a declared release contract. Nullable returns remain explicitly nullable. RunMat rejects calls when an argument cannot be represented by the declared native type without violating its range or ownership contract.

## Compilation And Remote Execution

`runmat compile` embeds the canonical prepared-interface manifest and exact library bytes in the standalone program. At startup, the linked runtime validates the bundle, writes it into a private temporary directory, seals the files read-only, installs only the requested interfaces, and admits the executable's interop requirements before running its entrypoint. The resulting executable does not depend on the project checkout or the original library path.

Remote jobs carry the same two exact objects in the execution bundle. A worker verifies their descriptors, privately materializes them, confirms the target and digest, installs only identities requested by the executable, and rejects missing, changed, ambiguous, orphaned, or unrequested native libraries before program execution.

Native libraries are unavailable inside the browser's WebAssembly runtime. The portable executable still carries the interop requirement, allowing browser and WASM hosts to reject it before executing program instructions with a native-capability diagnostic. RunMat does not silently substitute a browser implementation for a declared native interface.

## Trust

A shared library loaded in process has the permissions of the RunMat process and can corrupt or terminate it. Treat in-process interfaces as trusted native code. Exact digests provide identity and tamper detection; they do not make library code safe.
