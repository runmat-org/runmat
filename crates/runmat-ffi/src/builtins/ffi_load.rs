//! `ffi_load` builtin - Load a native library.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::registry::load_library;

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "ffi_load"
category: "ffi"
keywords: ["ffi", "load", "library", "native", "dll", "so"]
summary: "Load a native shared library for FFI calls."
---

# ffi_load

Load a native shared library (.dll on Windows, .so on Linux, .dylib on macOS).

## Syntax

```matlab
ffi_load("libname")
```

## Description

`ffi_load("libname")` loads the specified native library into memory. The library
can then be used with `ffi_call` to invoke functions.

## Examples

```matlab
ffi_load("mymath");
result = ffi_call("mymath", "add", 1.0, 2.0);
```
"#;

/// Load a native library for FFI calls.
#[runtime_builtin(
    name = "ffi_load",
    category = "ffi",
    summary = "Load a native shared library for FFI calls.",
    keywords = "ffi,load,library,native,dll"
)]
pub fn ffi_load_builtin(lib_name: Value) -> Result<Value, String> {
    let name = extract_string(&lib_name)?;
    load_library(&name)?;
    Ok(Value::Bool(true))
}

fn extract_string(value: &Value) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) => Ok(ca.data.iter().collect()),
        _ => Err("ffi_load: library name must be a string".to_string()),
    }
}
