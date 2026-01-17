use runmat_builtins::{builtin_functions, LogicalArray, NumericDType, Tensor, Value};
use runmat_async::{PendingInteraction, SuspendMarker};

use crate::{make_cell_with_shape, new_object_builtin, build_runtime_error, RuntimeControlFlow, RuntimeError};

/// Return `true` when the passed value is a GPU-resident tensor handle.
pub fn is_gpu_value(value: &Value) -> bool {
    matches!(value, Value::GpuTensor(_))
}

/// Returns true when the value (or nested elements) contains any GPU-resident tensors.
pub fn value_contains_gpu(value: &Value) -> bool {
    match value {
        Value::GpuTensor(_) => true,
        Value::Cell(ca) => ca.data.iter().any(|ptr| value_contains_gpu(ptr)),
        Value::Struct(sv) => sv.fields.values().any(value_contains_gpu),
        Value::Object(obj) => obj.properties.values().any(value_contains_gpu),
        _ => false,
    }
}

/// Convert GPU-resident values to host tensors when an acceleration provider exists.
/// Non-GPU inputs are passed through unchanged.
pub fn gather_if_needed(value: &Value) -> Result<Value, RuntimeControlFlow> {
    match value {
        Value::GpuTensor(handle) => {
            // In parallel test runs, ensure the WGPU provider is reasserted for WGPU handles.
            #[cfg(all(test, feature = "wgpu"))]
            {
                if handle.device_id != 0 {
                    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                    );
                }
            }
            let provider = runmat_accelerate_api::provider_for_handle(handle).ok_or_else(|| {
                RuntimeControlFlow::Error(build_runtime_error("gather: no acceleration provider registered").build())
            })?;
            let is_logical = runmat_accelerate_api::handle_is_logical(handle);
            let host = match provider.download(handle) {
                Ok(host) => host,
                Err(err) => {
                    if let Some(marker) = err.downcast_ref::<SuspendMarker>() {
                        return Err(RuntimeControlFlow::Suspend(PendingInteraction {
                            prompt: marker.prompt.clone(),
                            kind: marker.kind,
                        }));
                    }
                    return Err(RuntimeControlFlow::Error(
                        build_runtime_error(format!("gather: {err}")).build(),
                    ));
                }
            };
            runmat_accelerate_api::clear_residency(handle);
            let runmat_accelerate_api::HostTensorOwned { data, shape } = host;
            if is_logical {
                let bits: Vec<u8> = data.iter().map(|&v| if v != 0.0 { 1 } else { 0 }).collect();
                let logical = LogicalArray::new(bits, shape).map_err(|e| {
                    RuntimeControlFlow::Error(build_runtime_error(format!("gather: {e}")).build())
                })?;
                Ok(Value::LogicalArray(logical))
            } else {
                let mut data = data;
                let precision = runmat_accelerate_api::handle_precision(handle)
                    .unwrap_or_else(|| provider.precision());
                if matches!(precision, runmat_accelerate_api::ProviderPrecision::F32) {
                    for value in &mut data {
                        *value = (*value as f32) as f64;
                    }
                }
                let dtype = match precision {
                    runmat_accelerate_api::ProviderPrecision::F32 => NumericDType::F32,
                    runmat_accelerate_api::ProviderPrecision::F64 => NumericDType::F64,
                };
                let tensor = Tensor::new_with_dtype(data, shape, dtype).map_err(|e| {
                    RuntimeControlFlow::Error(build_runtime_error(format!("gather: {e}")).build())
                })?;
                Ok(Value::Tensor(tensor))
            }
        }
        Value::Cell(ca) => {
            let mut gathered = Vec::with_capacity(ca.data.len());
            for ptr in &ca.data {
                gathered.push(gather_if_needed(ptr)?);
            }
            make_cell_with_shape(gathered, ca.shape.clone()).map_err(|err| {
                RuntimeControlFlow::Error(build_runtime_error(format!("gather: {err}")).build())
            })
        }
        Value::Struct(sv) => {
            let mut gathered = sv.clone();
            for value in gathered.fields.values_mut() {
                let updated = gather_if_needed(value)?;
                *value = updated;
            }
            Ok(Value::Struct(gathered))
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
pub fn call_builtin(name: &str, args: &[Value]) -> Result<Value, RuntimeControlFlow> {
    call_builtin_sync(name, args)
}

pub async fn call_builtin_async(name: &str, args: &[Value]) -> Result<Value, RuntimeControlFlow> {
    call_builtin_sync(name, args)
}

fn call_builtin_sync(name: &str, args: &[Value]) -> Result<Value, RuntimeControlFlow> {
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
        return Err(RuntimeControlFlow::Error(build_runtime_error(format!(
            "{}: Undefined function: {name}",
            "MATLAB:UndefinedFunction"
        )).build()));
    }

    // Partition into no-category (tests/legacy shims) and categorized (library) builtins.
    let mut no_category: Vec<&runmat_builtins::BuiltinFunction> = Vec::new();
    let mut categorized: Vec<&runmat_builtins::BuiltinFunction> = Vec::new();
    for b in matching_builtins {
        if b.category.is_empty() {
            no_category.push(b);
        } else {
            categorized.push(b);
        }
    }

    // Try each builtin until one succeeds. Within each group, prefer later-registered
    // implementations to allow overrides when names collide.
    let mut last_error = RuntimeError::new("unknown error");
    for builtin in no_category
        .into_iter()
        .rev()
        .chain(categorized.into_iter().rev())
    {
        let f = builtin.implementation;
        match (f)(args) {
            Ok(mut result) => {
                // Normalize certain logical scalar results to numeric 0/1 for
                // compatibility with legacy expectations in dispatcher tests
                // and VM shims.
                if matches!(name, "eq" | "ne" | "gt" | "ge" | "lt" | "le") {
                    if let Value::Bool(flag) = result {
                        result = Value::Num(if flag { 1.0 } else { 0.0 });
                    }
                }
                return Ok(result);
            }
            Err(flow) => match flow {
                RuntimeControlFlow::Suspend(pending) => return Err(RuntimeControlFlow::Suspend(pending)),
                RuntimeControlFlow::Error(err) => {
                    if should_retry_with_gpu_gather(&err, args) {
                        match gather_args_for_retry(args) {
                        Ok(Some(gathered_args)) => match (f)(&gathered_args) {
                            Ok(result) => return Ok(result),
                            Err(RuntimeControlFlow::Suspend(pending)) => {
                                return Err(RuntimeControlFlow::Suspend(pending))
                            }
                            Err(RuntimeControlFlow::Error(retry_err)) => last_error = retry_err,
                        },
                        Ok(None) => last_error = err,
                        Err(RuntimeControlFlow::Suspend(pending)) => {
                            return Err(RuntimeControlFlow::Suspend(pending))
                        }
                        Err(RuntimeControlFlow::Error(gather_err)) => last_error = gather_err,
                    }
                    } else {
                        last_error = err;
                    }
                }
            },
        }
    }

    // If none succeeded, return the last error
    Err(build_runtime_error(format!(
        "No matching overload for `{}` with {} args: {}",
        name,
        args.len(),
        last_error.message()
    ))
    .build()
    .into())
}

fn should_retry_with_gpu_gather(err: &RuntimeError, args: &[Value]) -> bool {
    if !args.iter().any(value_contains_gpu) {
        return false;
    }
    let lowered = err.message().to_ascii_lowercase();
    lowered.contains("gpu")
}

fn gather_args_for_retry(args: &[Value]) -> Result<Option<Vec<Value>>, RuntimeControlFlow> {
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
