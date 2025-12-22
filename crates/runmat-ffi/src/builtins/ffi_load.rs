//! `ffi_load` builtin - Load a native library.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::registry::{load_library, load_library_with_signatures};

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "ffi_load"
category: "ffi"
keywords: ["ffi", "load", "library", "native", "dll", "so", "signatures"]
summary: "Load a native shared library for FFI calls."
---

# ffi_load

Load a native shared library (.dll on Windows, .so on Linux, .dylib on macOS).

## Syntax

```matlab
ffi_load("libname")
ffi_load("libname", "libname.ffi")
```

## Description

`ffi_load("libname")` loads the specified native library into memory. The library
can then be used with `ffi_call` to invoke functions.

`ffi_load("libname", "libname.ffi")` also loads a signature file that defines
function types for type-safe FFI calls.

### Signature File Format

```
# Comments start with #
add: (f64, f64) -> f64
square: (f64) -> f64
get_pi: () -> f64
scale_array: (ptr<f64>, usize, usize, f64, ptr_mut<f64>) -> i32
```

## Examples

```matlab
% Load without signatures (types inferred from arguments)
ffi_load("mymath");
result = ffi_call("mymath", "add", 1.0, 2.0);

% Load with signatures (type-safe calls)
ffi_load("mymath", "mymath.ffi");
result = ffi_call("mymath", "add", 1.0, 2.0);
```
"#;

/// Load a native library for FFI calls.
///
/// Syntax:
///   ffi_load("libname")
///   ffi_load("libname", "libname.ffi")
#[runtime_builtin(
    name = "ffi_load",
    category = "ffi",
    summary = "Load a native shared library for FFI calls.",
    keywords = "ffi,load,library,native,dll,signatures"
)]
pub fn ffi_load_builtin(lib_name: Value, rest: Vec<Value>) -> Result<Value, String> {
    let name = extract_string(&lib_name, "library name")?;

    if rest.is_empty() {
        // No signature file
        load_library(&name)?;
    } else {
        // Signature file provided
        let sig_path = extract_string(&rest[0], "signature file path")?;
        load_library_with_signatures(&name, &sig_path)?;
    }

    Ok(Value::Bool(true))
}

fn extract_string(value: &Value, context: &str) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) => Ok(ca.data.iter().collect()),
        _ => Err(format!("ffi_load: {} must be a string", context)),
    }
}
