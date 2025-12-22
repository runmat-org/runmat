//! `ffi_unload` builtin - Unload a native library.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::registry::global_registry;

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "ffi_unload"
category: "ffi"
keywords: ["ffi", "unload", "library", "native"]
summary: "Unload a previously loaded native library."
---

# ffi_unload

Unload a native shared library that was loaded with `ffi_load`.

## Syntax

```matlab
ffi_unload("libname")
```
"#;

/// Unload a native library.
#[runtime_builtin(
    name = "ffi_unload",
    category = "ffi",
    summary = "Unload a previously loaded native library.",
    keywords = "ffi,unload,library"
)]
pub fn ffi_unload_builtin(lib_name: Value) -> Result<Value, String> {
    let name = extract_string(&lib_name)?;

    let mut registry = global_registry()
        .lock()
        .map_err(|_| "Failed to acquire library registry lock")?;

    if registry.unload(&name) {
        Ok(Value::Bool(true))
    } else {
        Ok(Value::Bool(false))
    }
}

fn extract_string(value: &Value) -> Result<String, String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) => Ok(ca.data.iter().collect()),
        _ => Err("ffi_unload: library name must be a string".to_string()),
    }
}
