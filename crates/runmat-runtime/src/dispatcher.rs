use std::collections::HashMap;

use runmat_builtins::{builtin_functions, Tensor, Value};

use crate::{make_cell, new_object_builtin};

/// Return `true` when the passed value is a GPU-resident tensor handle.
pub fn is_gpu_value(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(_))
}

/// Returns true when the value (or nested elements) contains any GPU-resident tensors.
pub fn value_contains_gpu(value: &Value) -> bool {
    match value {
        Value::GpuTensor(_) => true,
        Value::Cell(ca) => ca.data.iter().any(|ptr| value_contains_gpu(&**ptr)),
        Value::Struct(sv) => sv.fields.values().any(value_contains_gpu),
        Value::Object(obj) => obj.properties.values().any(value_contains_gpu),
        _ => false,
    }
}

/// Convert GPU-resident values to host tensors when an acceleration provider exists.
/// Non-GPU inputs are passed through unchanged.
pub fn gather_if_needed(value: &Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => {
            let provider = runmat_accelerate_api::provider()
                .ok_or_else(|| "gather: no acceleration provider registered".to_string())?;
            let host = provider.download(handle).map_err(|e| e.to_string())?;
            let tensor = Tensor::new(host.data, host.shape).map_err(|e| e.to_string())?;
            Ok(Value::Tensor(tensor))
        }
        Value::Cell(ca) => {
            let mut gathered = Vec::with_capacity(ca.data.len());
            for ptr in &ca.data {
                gathered.push(gather_if_needed(&**ptr)?);
            }
            make_cell(gathered, ca.rows, ca.cols)
        }
        Value::Struct(sv) => {
            let mut fields = HashMap::with_capacity(sv.fields.len());
            for (key, val) in &sv.fields {
                fields.insert(key.clone(), gather_if_needed(val)?);
            }
            Ok(Value::Struct(runmat_builtins::StructValue { fields }))
        }
        Value::Object(obj) => {
            let mut cloned = obj.clone();
            for value in cloned.properties.values_mut() {
                *value = gather_if_needed(value)?;
            }
            Ok(Value::Object(cloned))
        }
        other => Ok(other.clone()),
    }
}

/// Call a registered language builtin by name.
/// Supports function overloading by trying different argument patterns.
/// Returns an error if no builtin with that name and compatible arguments is found.
pub fn call_builtin(name: &str, args: &[Value]) -> Result<Value, String> {
    let mut matching_builtins = Vec::new();

    // Collect all builtins with the matching name
    for b in builtin_functions() {
        if b.name == name {
            matching_builtins.push(b);
        }
    }

    if matching_builtins.is_empty() {
        // Fallback: treat as class constructor if class is registered
        if let Some(cls) = runmat_builtins::get_class(name) {
            // Prefer explicit constructor method with the same name as class (static)
            if let Some(ctor) = cls.methods.get(name) {
                // Dispatch to constructor builtin; pass args through
                return call_builtin(&ctor.function_name, args);
            }
            // Otherwise default-construct object
            return new_object_builtin(name.to_string());
        }
        return Err(format!(
            "{}: Undefined function: {name}",
            "MATLAB:UndefinedFunction"
        ));
    }

    // Try each builtin until one succeeds
    let mut last_error = String::new();
    for builtin in matching_builtins {
        let f = builtin.implementation;
        match (f)(args) {
            Ok(result) => return Ok(result),
            Err(err) => {
                if should_retry_with_gpu_gather(&err, args) {
                    match gather_args_for_retry(args) {
                        Ok(Some(gathered_args)) => match (f)(&gathered_args) {
                            Ok(result) => return Ok(result),
                            Err(retry_err) => last_error = retry_err,
                        },
                        Ok(None) => last_error = err,
                        Err(gather_err) => last_error = gather_err,
                    }
                } else {
                    last_error = err;
                }
            }
        }
    }

    // If none succeeded, return the last error
    Err(format!(
        "No matching overload for `{}` with {} args: {}",
        name,
        args.len(),
        last_error
    ))
}

fn should_retry_with_gpu_gather(err: &str, args: &[Value]) -> bool {
    if !args.iter().any(value_contains_gpu) {
        return false;
    }
    let lowered = err.to_ascii_lowercase();
    lowered.contains("gpu")
}

fn gather_args_for_retry(args: &[Value]) -> Result<Option<Vec<Value>>, String> {
    let mut gathered_any = false;
    let mut gathered_args = Vec::with_capacity(args.len());
    for arg in args {
        if value_contains_gpu(arg) {
            gathered_args.push(gather_if_needed(arg)?);
            gathered_any = true;
        } else {
            gathered_args.push(arg.clone());
        }
    }
    if gathered_any {
        Ok(Some(gathered_args))
    } else {
        Ok(None)
    }
}
