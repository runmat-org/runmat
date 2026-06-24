use crate::{build_runtime_error, create_class_object, make_cell_with_shape, RuntimeError};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorOwned};
use runmat_builtins::{
    builtin_functions, ComplexTensor, LogicalArray, NumericDType, Tensor, Value,
};
use std::cell::RefCell;

thread_local! {
    static CLASS_ACCESS_CONTEXT: RefCell<Option<String>> = const { RefCell::new(None) };
}

#[cfg(target_arch = "wasm32")]
fn ensure_wasm_builtins_registered() {
    crate::builtins::wasm_registry::register_all();
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure_wasm_builtins_registered() {}

pub struct ClassAccessContextGuard {
    previous: Option<String>,
}

impl Drop for ClassAccessContextGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        CLASS_ACCESS_CONTEXT.with(|slot| {
            *slot.borrow_mut() = previous;
        });
    }
}

pub fn push_class_access_context(class_name: Option<String>) -> ClassAccessContextGuard {
    let previous =
        CLASS_ACCESS_CONTEXT.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), class_name));
    ClassAccessContextGuard { previous }
}

fn current_class_access_context() -> Option<String> {
    CLASS_ACCESS_CONTEXT.with(|slot| slot.borrow().clone())
}

pub fn class_access_context() -> Option<String> {
    current_class_access_context()
}

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
        Value::Closure(closure) => closure.captures.iter().any(value_contains_gpu),
        Value::OutputList(values) => values.iter().any(value_contains_gpu),
        _ => false,
    }
}

/// Convert GPU-resident values to host tensors when an acceleration provider exists.
/// Non-GPU inputs are passed through unchanged.
pub async fn gather_if_needed_async(value: &Value) -> Result<Value, RuntimeError> {
    gather_if_needed_async_impl(value).await
}

pub async fn download_handle_async(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> anyhow::Result<HostTensorOwned> {
    provider.download(handle).await
}

fn gather_if_needed_async_impl<'a>(
    value: &'a Value,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value, RuntimeError>> + 'a>> {
    Box::pin(async move {
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
                let provider =
                    runmat_accelerate_api::provider_for_handle(handle).ok_or_else(|| {
                        build_runtime_error("gather: no acceleration provider registered")
                            .with_identifier("RunMat:gather:ProviderUnavailable")
                            .build()
                    })?;
                let is_logical = runmat_accelerate_api::handle_is_logical(handle);
                let host = download_handle_async(provider, handle)
                    .await
                    .map_err(|err| {
                        build_runtime_error(format!("gather: {err}"))
                            .with_identifier("RunMat:gather:DownloadFailed")
                            .build()
                    })?;
                runmat_accelerate_api::clear_residency(handle);
                let runmat_accelerate_api::HostTensorOwned {
                    data,
                    shape,
                    storage,
                } = host;
                if is_logical {
                    let bits: Vec<u8> =
                        data.iter().map(|&v| if v != 0.0 { 1 } else { 0 }).collect();
                    let logical = LogicalArray::new(bits, shape).map_err(|e| {
                        build_runtime_error(format!("gather: {e}"))
                            .with_identifier("RunMat:gather:LogicalShapeError")
                            .build()
                    })?;
                    Ok(Value::LogicalArray(logical))
                } else if storage == GpuTensorStorage::ComplexInterleaved {
                    let mut data = data;
                    let precision = runmat_accelerate_api::handle_precision(handle)
                        .unwrap_or_else(|| provider.precision());
                    if matches!(precision, runmat_accelerate_api::ProviderPrecision::F32) {
                        for value in &mut data {
                            *value = (*value as f32) as f64;
                        }
                    }
                    let mut complex = Vec::with_capacity(data.len() / 2);
                    for chunk in data.chunks_exact(2) {
                        complex.push((chunk[0], chunk[1]));
                    }
                    let tensor = ComplexTensor::new(complex, shape).map_err(|e| {
                        build_runtime_error(format!("gather: {e}"))
                            .with_identifier("RunMat:gather:TensorShapeError")
                            .build()
                    })?;
                    Ok(Value::ComplexTensor(tensor))
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
                        build_runtime_error(format!("gather: {e}"))
                            .with_identifier("RunMat:gather:TensorShapeError")
                            .build()
                    })?;
                    Ok(Value::Tensor(tensor))
                }
            }
            Value::Cell(ca) => {
                let mut gathered = Vec::with_capacity(ca.data.len());
                for ptr in &ca.data {
                    gathered.push(gather_if_needed_async_impl(ptr).await?);
                }
                make_cell_with_shape(gathered, ca.shape.clone()).map_err(|err| {
                    build_runtime_error(format!("gather: {err}"))
                        .with_identifier("RunMat:gather:CellShapeError")
                        .build()
                })
            }
            Value::Struct(sv) => {
                let mut gathered = sv.clone();
                for value in gathered.fields.values_mut() {
                    let updated = gather_if_needed_async_impl(value).await?;
                    *value = updated;
                }
                Ok(Value::Struct(gathered))
            }
            Value::Object(obj) => {
                let mut cloned = obj.clone();
                for value in cloned.properties.values_mut() {
                    *value = gather_if_needed_async_impl(value).await?;
                }
                Ok(Value::Object(cloned))
            }
            Value::Closure(closure) => {
                let mut cloned = closure.clone();
                for value in &mut cloned.captures {
                    *value = gather_if_needed_async_impl(value).await?;
                }
                Ok(Value::Closure(cloned))
            }
            Value::OutputList(values) => {
                let mut gathered = Vec::with_capacity(values.len());
                for value in values {
                    gathered.push(gather_if_needed_async_impl(value).await?);
                }
                Ok(Value::OutputList(gathered))
            }
            other => Ok(other.clone()),
        }
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn gather_if_needed(value: &Value) -> Result<Value, RuntimeError> {
    futures::executor::block_on(gather_if_needed_async(value))
}

#[cfg(target_arch = "wasm32")]
pub fn gather_if_needed(_value: &Value) -> Result<Value, RuntimeError> {
    Err(
        build_runtime_error("gather: synchronous gather is unavailable on wasm")
            .with_identifier("RunMat:gather:UnavailableOnWasm")
            .build(),
    )
}

/// Call a registered language builtin by name.
/// Supports function overloading by trying different argument patterns.
/// Returns an error if no builtin with that name and compatible arguments is found.
pub fn call_builtin(name: &str, args: &[Value]) -> Result<Value, RuntimeError> {
    futures::executor::block_on(call_builtin_async(name, args))
}

#[async_recursion::async_recursion(?Send)]
async fn call_builtin_async_impl(
    name: &str,
    args: &[Value],
    output_count: Option<usize>,
) -> Result<Value, RuntimeError> {
    ensure_wasm_builtins_registered();

    let _output_guard = crate::output_count::push_output_count(output_count);
    let mut matching_builtins = Vec::new();

    // Collect all builtins with the matching name
    for b in builtin_functions() {
        if b.name == name {
            matching_builtins.push(b);
        }
    }

    if matching_builtins.is_empty() {
        if let Some(result) = try_call_registered_instance_method(name, args, output_count).await? {
            return Ok(result);
        }
        if let Some(result) = try_call_registered_static_method(name, args, output_count).await? {
            return Ok(result);
        }
        // Fallback: treat as class constructor if class is registered.
        if runmat_builtins::get_class(name).is_some() {
            return call_registered_class_constructor(name, args, output_count).await;
        }
        return Err(build_runtime_error(format!("Undefined function: {name}"))
            .with_identifier("RunMat:UndefinedFunction")
            .build());
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
    let matching_count = no_category.len() + categorized.len();

    // Try each builtin until one succeeds. Within each group, prefer later-registered
    // implementations to allow overrides when names collide.
    let mut last_error = RuntimeError::new("unknown error");
    for builtin in no_category
        .into_iter()
        .rev()
        .chain(categorized.into_iter().rev())
    {
        let f = builtin.implementation;
        match (f)(args).await {
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
            Err(err) => {
                if should_retry_with_gpu_gather(&err, args) {
                    match gather_args_for_retry_async(args).await {
                        Ok(Some(gathered_args)) => match (f)(&gathered_args).await {
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

    // A single implementation already knows whether its inputs are invalid or
    // whether execution failed. Preserve that error verbatim instead of
    // presenting it as overload resolution noise.
    if matching_count == 1 || last_error.identifier().is_some() {
        return Err(last_error);
    }

    // If none succeeded, return the last error
    let identifier = last_error
        .identifier()
        .unwrap_or("RunMat:NoMatchingOverload")
        .to_string();
    let mut builder = build_runtime_error(format!(
        "No matching overload for `{}` with {} args: {}",
        name,
        args.len(),
        last_error.message()
    ))
    .with_source(last_error);
    builder = builder.with_identifier(identifier);
    Err(builder.build())
}

async fn try_call_registered_instance_method(
    method_name: &str,
    args: &[Value],
    output_count: Option<usize>,
) -> Result<Option<Value>, RuntimeError> {
    let Some(receiver) = args.first() else {
        return Ok(None);
    };
    let class_name = match receiver {
        Value::Object(obj) => obj.class_name.as_str(),
        Value::HandleObject(handle) => handle.class_name.as_str(),
        _ => return Ok(None),
    };
    let Some((method, owner)) = runmat_builtins::lookup_method(class_name, method_name) else {
        return Ok(None);
    };
    if method.is_static {
        return Ok(None);
    }
    let caller_class = current_class_access_context();
    let access_allowed = match method.access {
        runmat_builtins::Access::Public => true,
        runmat_builtins::Access::Private => caller_class.as_deref() == Some(owner.as_str()),
        runmat_builtins::Access::Protected => caller_class
            .as_deref()
            .is_some_and(|caller| runmat_builtins::is_class_or_subclass(caller, &owner)),
    };
    if !access_allowed {
        return Err(build_runtime_error(format!(
            "Method '{}' is not accessible from current context.",
            method_name
        ))
        .with_identifier("RunMat:MethodPrivate")
        .build());
    }
    if let Some(result) = crate::user_functions::try_call_semantic_function_by_name(
        &method.function_name,
        args,
        output_count.unwrap_or(1),
    )
    .await
    {
        return result.map(Some);
    }
    if runmat_builtins::builtin_function_by_name(&method.function_name).is_some()
        && method.function_name != method_name
    {
        return call_builtin_async_impl(&method.function_name, args, output_count)
            .await
            .map(Some);
    }
    let owner_qualified = format!("{owner}.{method_name}");
    if owner_qualified != method.function_name {
        if let Some(result) = crate::user_functions::try_call_semantic_function_by_name(
            &owner_qualified,
            args,
            output_count.unwrap_or(1),
        )
        .await
        {
            return result.map(Some);
        }
        if runmat_builtins::builtin_function_by_name(&owner_qualified).is_some()
            && owner_qualified != method_name
        {
            return call_builtin_async_impl(&owner_qualified, args, output_count)
                .await
                .map(Some);
        }
    }
    Ok(None)
}

async fn try_call_registered_static_method(
    qualified_name: &str,
    args: &[Value],
    output_count: Option<usize>,
) -> Result<Option<Value>, RuntimeError> {
    let Some((class_name, method_name)) = qualified_name.rsplit_once('.') else {
        return Ok(None);
    };
    if class_name.trim().is_empty() || method_name.trim().is_empty() {
        return Ok(None);
    }
    if runmat_builtins::get_class(class_name).is_none() {
        return Ok(None);
    }
    let Some((method, owner)) = runmat_builtins::lookup_method(class_name, method_name) else {
        return Ok(None);
    };
    if !method.is_static || method.access != runmat_builtins::Access::Public {
        return Ok(None);
    }
    if let Some(result) = crate::user_functions::try_call_semantic_function_by_name(
        &method.function_name,
        args,
        output_count.unwrap_or(1),
    )
    .await
    {
        return result.map(Some);
    }
    if runmat_builtins::builtin_function_by_name(&method.function_name).is_some()
        && method.function_name != qualified_name
    {
        return call_builtin_async_impl(&method.function_name, args, output_count)
            .await
            .map(Some);
    }
    let owner_qualified = format!("{owner}.{method_name}");
    if owner_qualified != method.function_name {
        if let Some(result) = crate::user_functions::try_call_semantic_function_by_name(
            &owner_qualified,
            args,
            output_count.unwrap_or(1),
        )
        .await
        {
            return result.map(Some);
        }
        if runmat_builtins::builtin_function_by_name(&owner_qualified).is_some()
            && owner_qualified != qualified_name
        {
            return call_builtin_async_impl(&owner_qualified, args, output_count)
                .await
                .map(Some);
        }
    }
    Ok(None)
}

async fn call_registered_class_constructor(
    class_name: &str,
    args: &[Value],
    output_count: Option<usize>,
) -> Result<Value, RuntimeError> {
    let requested_outputs = output_count.unwrap_or(1);
    let default_object = create_class_object(class_name.to_string()).await?;
    let constructor_method_name = class_name.rsplit('.').next().unwrap_or(class_name);
    let Some((ctor, owner)) = runmat_builtins::lookup_method(class_name, constructor_method_name)
        .or_else(|| runmat_builtins::lookup_method(class_name, class_name))
    else {
        return Ok(default_object);
    };
    let owner_qualified = format!("{owner}.{constructor_method_name}");
    let caller_class = current_class_access_context();
    let ctor_access_allowed = match ctor.access {
        runmat_builtins::Access::Public => true,
        runmat_builtins::Access::Private => caller_class.as_deref() == Some(owner.as_str()),
        runmat_builtins::Access::Protected => caller_class
            .as_deref()
            .is_some_and(|caller| runmat_builtins::is_class_or_subclass(caller, &owner)),
    };
    if !ctor_access_allowed {
        return Err(build_runtime_error(format!(
            "Constructor '{}' is not accessible from current context.",
            class_name
        ))
        .with_identifier("RunMat:MethodPrivate")
        .build());
    }
    let constructor_result = crate::with_constructor_receiver(default_object.clone(), async {
        if let Some(result) = crate::user_functions::try_call_semantic_function_by_name(
            &ctor.function_name,
            args,
            requested_outputs,
        )
        .await
        {
            return Ok::<Option<Value>, RuntimeError>(Some(result?));
        }
        if runmat_builtins::builtin_function_by_name(&ctor.function_name).is_some()
            && ctor.function_name != class_name
        {
            let result = call_builtin_async_impl(&ctor.function_name, args, output_count).await?;
            return Ok::<Option<Value>, RuntimeError>(Some(result));
        }
        if let Some(result) = crate::user_functions::try_call_semantic_function_by_name(
            &owner_qualified,
            args,
            requested_outputs,
        )
        .await
        {
            return Ok::<Option<Value>, RuntimeError>(Some(result?));
        }
        if runmat_builtins::builtin_function_by_name(&owner_qualified).is_some()
            && owner_qualified != class_name
        {
            let result = call_builtin_async_impl(&owner_qualified, args, output_count).await?;
            return Ok::<Option<Value>, RuntimeError>(Some(result));
        }
        Ok::<Option<Value>, RuntimeError>(None)
    })
    .await?;
    let Some(result) = constructor_result else {
        return Ok(default_object);
    };
    normalize_constructor_result(default_object, result, requested_outputs)
}

fn normalize_constructor_result(
    default_object: Value,
    result: Value,
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
    if requested_outputs != 1 {
        return Ok(result);
    }
    match result {
        Value::Struct(struct_value) => match default_object {
            Value::Object(mut object) => {
                for (field, value) in struct_value.fields {
                    object.properties.insert(field, value);
                }
                Ok(Value::Object(object))
            }
            Value::HandleObject(handle) => {
                enum ConstructorMergeStatus {
                    Merged,
                    InvalidHandle,
                    NonObject,
                }

                let merged = runmat_gc::gc_with_value_mut(&handle.target, |target| {
                    if let Value::Object(object) = target {
                        if !crate::object_handle_flag_valid(object) {
                            return ConstructorMergeStatus::InvalidHandle;
                        }
                        for (field, value) in struct_value.fields {
                            runmat_gc::gc_record_handle_write(&handle.target, &value);
                            object.properties.insert(field, value);
                        }
                        ConstructorMergeStatus::Merged
                    } else {
                        ConstructorMergeStatus::NonObject
                    }
                })
                .map_err(|e| {
                    build_runtime_error(format!("constructor result handle target invalid: {e}"))
                        .build()
                })?;
                match merged {
                    ConstructorMergeStatus::Merged => {}
                    ConstructorMergeStatus::InvalidHandle => {
                        return Err(build_runtime_error(
                            "constructor result handle target is invalid",
                        )
                        .build());
                    }
                    ConstructorMergeStatus::NonObject => {
                        return Err(build_runtime_error(
                            "constructor result handle target is not an object",
                        )
                        .build());
                    }
                }
                Ok(Value::HandleObject(handle))
            }
            _ => Ok(Value::Struct(struct_value)),
        },
        Value::Object(_) | Value::HandleObject(_) => Ok(result),
        _ => Ok(default_object),
    }
}

pub async fn call_builtin_async(name: &str, args: &[Value]) -> Result<Value, RuntimeError> {
    call_builtin_async_impl(name, args, None).await
}

pub async fn call_builtin_async_with_outputs(
    name: &str,
    args: &[Value],
    output_count: usize,
) -> Result<Value, RuntimeError> {
    call_builtin_async_impl(name, args, Some(output_count)).await
}

fn should_retry_with_gpu_gather(err: &RuntimeError, args: &[Value]) -> bool {
    if !args.iter().any(value_contains_gpu) {
        return false;
    }
    let lowered = err.message().to_ascii_lowercase();
    lowered.contains("gpu")
}

async fn gather_args_for_retry_async(args: &[Value]) -> Result<Option<Vec<Value>>, RuntimeError> {
    let mut gathered_any = false;
    let mut gathered_args = Vec::with_capacity(args.len());
    for arg in args {
        if value_contains_gpu(arg) {
            gathered_args.push(gather_if_needed_async(arg).await?);
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

#[cfg(test)]
mod tests {
    use super::{call_builtin, gather_if_needed_async, value_contains_gpu};
    use runmat_accelerate_api::{GpuTensorHandle, ThreadProviderGuard};
    use runmat_builtins::{
        register_class, Access, ClassDef, Closure, MethodDef, StructValue, Value,
    };
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_CLASS_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_class_name(prefix: &str) -> String {
        let id = TEST_CLASS_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("{}_{}", prefix, id)
    }

    #[test]
    fn value_contains_gpu_detects_nested_closure_captures() {
        let value = Value::Closure(Closure {
            function_name: "worker".to_string(),
            bound_function: None,
            captures: vec![Value::GpuTensor(GpuTensorHandle {
                shape: vec![1],
                device_id: 999,
                buffer_id: 42,
            })],
        });
        assert!(value_contains_gpu(&value));
    }

    #[test]
    fn value_contains_gpu_detects_output_list_entries() {
        let value = Value::OutputList(vec![
            Value::Num(1.0),
            Value::GpuTensor(GpuTensorHandle {
                shape: vec![1],
                device_id: 998,
                buffer_id: 43,
            }),
        ]);
        assert!(value_contains_gpu(&value));
    }

    #[test]
    fn gather_if_needed_reports_provider_unavailable_for_nested_output_list_gpu() {
        runmat_accelerate_api::clear_provider();
        let _provider_guard = ThreadProviderGuard::set(None);
        let value = Value::OutputList(vec![Value::GpuTensor(GpuTensorHandle {
            shape: vec![1],
            // Keep device id at zero so test-only WGPU re-registration hooks are not triggered.
            device_id: 0,
            buffer_id: 44,
        })]);
        let err = futures::executor::block_on(gather_if_needed_async(&value))
            .expect_err("missing provider should fail nested output-list gather");
        assert_eq!(err.identifier(), Some("RunMat:gather:ProviderUnavailable"));
    }

    #[test]
    fn gather_if_needed_reports_provider_unavailable_for_closure_capture_gpu() {
        runmat_accelerate_api::clear_provider();
        let _provider_guard = ThreadProviderGuard::set(None);
        let value = Value::Closure(Closure {
            function_name: "worker".to_string(),
            bound_function: None,
            captures: vec![Value::GpuTensor(GpuTensorHandle {
                shape: vec![1],
                // Keep device id at zero so test-only WGPU re-registration hooks are not triggered.
                device_id: 0,
                buffer_id: 45,
            })],
        });
        let err = futures::executor::block_on(gather_if_needed_async(&value))
            .expect_err("missing provider should fail closure-captured gather");
        assert_eq!(err.identifier(), Some("RunMat:gather:ProviderUnavailable"));
    }

    #[test]
    fn constructor_fallback_uses_inherited_constructor_metadata_with_semantic_invoker() {
        let parent_name = unique_class_name("runtime_ctor_parent");
        let child_name = unique_class_name("runtime_ctor_child");
        let ctor_fn_name = unique_class_name("runtime_ctor_fn");
        let ctor_fn_name_for_resolver = ctor_fn_name.clone();
        let ctor_fn_name_for_invoker = ctor_fn_name.clone();
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(Some(
            std::sync::Arc::new(move |name| (name == ctor_fn_name_for_resolver).then_some(10101)),
        ));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            std::sync::Arc::new(move |function, _args, requested_outputs| {
                assert_eq!(function, 10101);
                assert_eq!(requested_outputs, 1);
                let mut sv = StructValue::new();
                sv.fields.insert("x".to_string(), Value::Num(12.0));
                Box::pin(async move { Ok(Value::Struct(sv)) })
            }),
        ));

        let mut parent_methods = HashMap::new();
        parent_methods.insert(
            child_name.clone(),
            MethodDef {
                name: child_name.clone(),
                is_static: true,
                is_abstract: false,
                is_sealed: false,
                access: Access::Public,
                function_name: ctor_fn_name_for_invoker,
                implicit_class_argument: None,
            },
        );
        register_class(ClassDef {
            name: parent_name.clone(),
            parent: None,
            properties: HashMap::new(),
            methods: parent_methods,
        });
        register_class(ClassDef {
            name: child_name.clone(),
            parent: Some(parent_name),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });

        let out =
            call_builtin(&child_name, &[]).expect("inherited static constructor should dispatch");
        let Value::Object(obj) = out else {
            panic!("expected object from constructor dispatch");
        };
        assert_eq!(obj.class_name, child_name);
        assert_eq!(obj.properties.get("x"), Some(&Value::Num(12.0)));
    }

    #[test]
    fn constructor_fallback_defaults_when_constructor_is_private_or_unavailable() {
        let private_class_name = unique_class_name("runtime_ctor_private");
        let mut private_methods = HashMap::new();
        private_methods.insert(
            private_class_name.clone(),
            MethodDef {
                name: private_class_name.clone(),
                is_static: true,
                is_abstract: false,
                is_sealed: false,
                access: Access::Private,
                function_name: "Point.origin".to_string(),
                implicit_class_argument: None,
            },
        );
        register_class(ClassDef {
            name: private_class_name.clone(),
            parent: None,
            properties: HashMap::new(),
            methods: private_methods,
        });
        let err = call_builtin(&private_class_name, &[])
            .expect_err("private constructor should enforce access before default fallback");
        assert_eq!(err.identifier(), Some("RunMat:MethodPrivate"));

        let public_class_name = unique_class_name("runtime_ctor_public_no_semantic");
        let mut public_methods = HashMap::new();
        public_methods.insert(
            public_class_name.clone(),
            MethodDef {
                name: public_class_name.clone(),
                is_static: true,
                is_abstract: false,
                is_sealed: false,
                access: Access::Public,
                function_name: unique_class_name("runtime_ctor_missing_body"),
                implicit_class_argument: None,
            },
        );
        register_class(ClassDef {
            name: public_class_name.clone(),
            parent: None,
            properties: HashMap::new(),
            methods: public_methods,
        });

        let out = call_builtin(&public_class_name, &[])
            .expect("public ctor metadata without semantic body should default-construct");
        let Value::Object(obj) = out else {
            panic!("expected object result");
        };
        assert_eq!(obj.class_name, public_class_name);
    }

    #[test]
    fn dotted_static_method_name_dispatches_to_registered_class_method() {
        let class_name = unique_class_name("runtime_static_dispatch");
        let fn_name = unique_class_name("runtime_static_fn");
        register_class(ClassDef {
            name: class_name.clone(),
            parent: None,
            properties: HashMap::new(),
            methods: {
                let mut methods = HashMap::new();
                methods.insert(
                    "zero".to_string(),
                    MethodDef {
                        name: "zero".to_string(),
                        is_static: true,
                        is_abstract: false,
                        is_sealed: false,
                        access: Access::Public,
                        function_name: fn_name.clone(),
                        implicit_class_argument: None,
                    },
                );
                methods
            },
        });

        let fn_name_for_resolver = fn_name.clone();
        let _resolver_guard = crate::user_functions::install_semantic_function_resolver(Some(
            std::sync::Arc::new(move |name| (name == fn_name_for_resolver).then_some(20202)),
        ));
        let _invoker_guard = crate::user_functions::install_semantic_function_invoker(Some(
            std::sync::Arc::new(move |function, _args, requested_outputs| {
                assert_eq!(function, 20202);
                assert_eq!(requested_outputs, 1);
                Box::pin(async { Ok(Value::Num(77.0)) })
            }),
        ));

        let out = call_builtin(&format!("{class_name}.zero"), &[])
            .expect("dotted static class method call should dispatch");
        assert_eq!(out, Value::Num(77.0));
    }
}
