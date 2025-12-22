# runmat-ffi

Foreign Function Interface for calling native C libraries from RunMat.

## Overview

This crate provides builtin functions for calling native shared libraries from RunMat code:

- `ffi_load("libname")` - Load a native library
- `ffi_call("libname", "funcname", args...)` - Call a function in a loaded library
- `ffi_unload("libname")` - Unload a library

## Example (RunMat)

```matlab
% Call a function in a native library
result = ffi_call("mymath", "add", 1.0, 2.0);  % Returns 3.0

% Explicit load/unload
ffi_load("mymath");
y = ffi_call("mymath", "square", 4.0);  % Returns 16.0
ffi_unload("mymath");
```

## Supported Function Signatures

Currently supports these native function types:

### Scalar Functions

```c
double func();                           // nullary
double func(double x);                   // unary
double func(double x, double y);         // binary
double func(double a, double b, double c, ...); // up to 5 args
```

### Array Functions (Coming Soon)

```c
int process_array(
    const double* input, size_t rows, size_t cols,
    double* output, size_t* out_rows, size_t* out_cols
);
```

## Writing Native Libraries

Create a C library with exported functions:

```c
// mymath.c
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT double add(double x, double y) {
    return x + y;
}

EXPORT double square(double x) {
    return x * x;
}
```

Compile:
- Windows: `cl /LD mymath.c` → `mymath.dll`
- Linux: `gcc -shared -fPIC -o libmymath.so mymath.c`
- macOS: `clang -shared -o libmymath.dylib mymath.c`

## Library Search

Libraries are searched in this order:
1. Current directory
2. System library paths (PATH on Windows, LD_LIBRARY_PATH on Linux, DYLD_LIBRARY_PATH on macOS)

## Future Work

- `.ffi` signature language for type-safe calls
- Array function support with automatic memory management
- Complex number support
- Integer type support
