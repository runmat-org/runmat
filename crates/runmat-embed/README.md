# runmat-embed

C ABI for embedding RunMat in C/C++ applications.

## Overview

This crate provides a stable C ABI for:
- Creating and managing RunMat execution contexts
- Evaluating RunMat code from C/C++
- Exchanging numeric arrays between C and RunMat

## Building

```bash
cargo build -p runmat-embed --release
```

This produces:
- `librunmat.dll` / `librunmat.so` / `librunmat.dylib` (dynamic library)
- `librunmat.a` / `runmat.lib` (static library)

## C Header

The C header is located at `include/runmat.h`.

## Example (C)

```c
#include "runmat.h"
#include <stdio.h>

int main() {
    rm_context* ctx = rm_context_new();
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    rm_value** out = NULL;
    size_t nout = 0;
    rm_error err = {0};

    // Evaluate some code
    if (rm_eval(ctx, "A = [1 2; 3 4]; A * A", &out, &nout, &err) == RM_STATUS_OK) {
        printf("Evaluation succeeded, %zu results\n", nout);

        if (nout > 0) {
            RmArrayF64 array;
            if (rm_value_to_array_f64(out[0], &array, &err) == RM_STATUS_OK) {
                printf("Result is %zux%zu matrix:\n", array.rows, array.cols);
                for (size_t i = 0; i < array.rows; i++) {
                    for (size_t j = 0; j < array.cols; j++) {
                        printf("  %g", array.data[i + j * array.rows]);
                    }
                    printf("\n");
                }
            }
        }

        rm_values_free(ctx, out, nout);
    } else {
        fprintf(stderr, "Error: %s\n", err.message);
    }

    rm_context_free(ctx);
    return 0;
}
```

## API Reference

### Context Lifecycle

- `rm_context_new()` - Create a new execution context
- `rm_context_free(ctx)` - Free context and all owned values

### Code Evaluation

- `rm_eval(ctx, code, &out, &nout, &err)` - Evaluate null-terminated code
- `rm_eval_utf8(ctx, code, len, &out, &nout, &err)` - Evaluate code with explicit length

### Value Access

- `rm_value_to_f64(v, &out, &err)` - Extract scalar as double
- `rm_value_from_f64(ctx, x, &out, &err)` - Create scalar from double
- `rm_value_to_array_f64(v, &out, &err)` - Get array view (zero-copy)
- `rm_value_from_array_f64(ctx, array, &out, &err)` - Create array (copies data)

### Memory Management

- `rm_values_free(ctx, values, count)` - Free value array
- `rm_error_clear(&err)` - Clear error state

### Version Info

- `rm_get_abi_version()` - Get ABI version number
- `rm_get_version_string()` - Get version string

## Memory Ownership

- **Inputs from C**: When passing arrays to RunMat, the data is copied. C retains ownership of its buffers.
- **Outputs from RunMat**: Value pointers are owned by the context. Use `rm_values_free()` to release them.
- **Array views**: Pointers from `rm_value_to_array_f64()` are valid until the value or context is freed.
