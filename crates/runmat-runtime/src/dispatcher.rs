use crate::{build_runtime_error, create_class_object, make_cell_with_shape, RuntimeError};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::builtin_functions;

use runmat_value::Value;
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
    context: Option<std::rc::Rc<crate::context::RuntimeContextState>>,
}

impl Drop for ClassAccessContextGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        if let Some(context) = &self.context {
            context.call.borrow_mut().class_access = previous;
        } else {
            CLASS_ACCESS_CONTEXT.with(|slot| {
                *slot.borrow_mut() = previous;
            });
        }
    }
}

pub fn push_class_access_context(class_name: Option<String>) -> ClassAccessContextGuard {
    let context =
        crate::context::legacy::active().map(|context| std::rc::Rc::clone(context.state()));
    let previous = if let Some(context) = &context {
        std::mem::replace(&mut context.call.borrow_mut().class_access, class_name)
    } else {
        CLASS_ACCESS_CONTEXT.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), class_name))
    };
    ClassAccessContextGuard { previous, context }
}

fn current_class_access_context() -> Option<String> {
    if let Some(context) = crate::context::legacy::active() {
        return context.state().call.borrow().class_access.clone();
    }
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

fn gather_if_needed_async_impl<'a>(
    value: &'a Value,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value, RuntimeError>> + 'a>> {
    Box::pin(async move {
        match value {
            Value::GpuTensor(handle) => {
                // In parallel test runs, ensure the WGPU provider is reasserted for WGPU handles.
                #[cfg(all(test, feature = "wgpu"))]
                {
                    let active_owner = runmat_accelerate_api::provider()
                        .is_some_and(|provider| provider.device_id() == handle.device_id);
                    if handle.device_id != 0 && !active_owner {
                        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                    );
                    }
                }
                let provider = runmat_accelerate_api::provider_for_handle(handle)
                    .filter(|provider| provider.device_id() == handle.device_id)
                    .ok_or_else(|| {
                        build_runtime_error("gather: no acceleration provider registered")
                            .with_identifier("RunMat:gather:ProviderUnavailable")
                            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                            .build()
                    })?;
                let expected_element =
                    crate::builtins::common::gpu_helpers::expected_handle_numeric_element_type(
                        handle,
                    )
                    .map_err(|error| {
                        build_runtime_error(format!("gather: {error}"))
                            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
                            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                            .build()
                    })?;
                let host = provider.download_numeric(handle).await.map_err(|err| {
                    build_runtime_error(format!("gather: {err}"))
                        .with_identifier("RunMat:gather:DownloadFailed")
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build()
                })?;
                let expected_storage = runmat_accelerate_api::handle_storage(handle);
                if host.shape != handle.shape
                    || host.storage != expected_storage
                    || host.data.element_type() != expected_element
                {
                    return Err(provider_payload_mismatch(
                        handle,
                        &host.shape,
                        format!(
                            "{:?} {:?}, expected {:?} {:?}",
                            host.data.element_type(),
                            host.storage,
                            expected_element,
                            expected_storage
                        ),
                    ));
                }
                crate::builtins::common::gpu_helpers::value_from_numeric_download(
                    host,
                    runmat_accelerate_api::handle_is_logical(handle),
                )
                .map_err(|error| {
                    build_runtime_error(format!("gather: {error}"))
                        .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .build()
                })
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

fn provider_payload_mismatch(
    handle: &GpuTensorHandle,
    actual_shape: &[usize],
    detail: String,
) -> RuntimeError {
    build_runtime_error(format!(
        "gather: provider payload mismatch ({detail}; shape {actual_shape:?}, expected {:?})",
        handle.shape
    ))
    .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .build()
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
    let scoped_builtin_service = crate::context::legacy::active()
        .and_then(|context| context.service_ports().builtin().cloned());
    let matching_bindings = scoped_builtin_service.as_ref().map_or_else(
        || crate::builtin::runtime_builtin_bindings_by_name(name),
        |service| service.bindings_by_name(name),
    );
    let mut matching_builtins = Vec::new();

    // Collect all builtins with the matching name
    if scoped_builtin_service.is_none() {
        for b in builtin_functions() {
            if b.name == name {
                matching_builtins.push(b);
            }
        }
    }

    if !matching_bindings.is_empty() && !matching_builtins.is_empty() {
        return Err(build_runtime_error(format!(
            "builtin `{name}` has both canonical and legacy runtime bindings"
        ))
        .with_identifier("RunMat:Catalog:DuplicateBindingAuthority")
        .build());
    }

    if matching_bindings.is_empty() && matching_builtins.is_empty() {
        if let Some(result) = try_call_registered_instance_method(name, args, output_count).await? {
            return compatibility_checked_builtin_result(name, args, result);
        }
        if let Some(result) = try_call_registered_static_method(name, args, output_count).await? {
            return compatibility_checked_builtin_result(name, args, result);
        }
        // Fallback: treat as class constructor if class is registered.
        if crate::class_registry::get_class(name).is_some() {
            let result = call_registered_class_constructor(name, args, output_count).await?;
            return compatibility_checked_builtin_result(name, args, result);
        }
        return Err(build_runtime_error(format!("Undefined function: {name}"))
            .with_identifier("RunMat:UndefinedFunction")
            .build());
    }

    if let Some(result) = try_call_registered_instance_method(name, args, output_count).await? {
        return compatibility_checked_builtin_result(name, args, result);
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
    let matching_count = matching_bindings.len() + no_category.len() + categorized.len();
    let implementations = matching_bindings
        .into_iter()
        .rev()
        .map(|binding| binding.implementation)
        .chain(
            no_category
                .into_iter()
                .rev()
                .chain(categorized.into_iter().rev())
                .map(|builtin| builtin.implementation),
        );

    // Try each builtin until one succeeds. Within each group, prefer later-registered
    // implementations to allow overrides when names collide.
    let mut last_error = RuntimeError::new("unknown error");
    for implementation in implementations {
        let f = implementation;
        match (f)(args).await {
            Ok(result) => return compatibility_checked_builtin_result(name, args, result),
            Err(err) => {
                if should_retry_with_gpu_gather(&err, args) {
                    match gather_args_for_retry_async(args).await {
                        Ok(Some(gathered_args)) => match (f)(&gathered_args).await {
                            Ok(result) => {
                                return compatibility_checked_builtin_result(name, args, result);
                            }
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

fn compatibility_checked_builtin_result(
    name: &str,
    args: &[Value],
    mut result: Value,
) -> Result<Value, RuntimeError> {
    crate::compatibility::ensure_value_compatible(&result, name)?;
    propagate_gpu_provenance(name, args, &mut result);
    Ok(result)
}

fn propagate_gpu_provenance(name: &str, args: &[Value], result: &mut Value) {
    let mut saw_gpu = false;
    let mut explicit = false;
    for arg in args {
        visit_gpu_handles(arg, &mut |handle| {
            saw_gpu = true;
            explicit |= runmat_accelerate_api::handle_is_explicit(handle);
        });
    }
    if !saw_gpu {
        let explicit_constructor = matches!(
            name,
            "zeros"
                | "ones"
                | "inf"
                | "nan"
                | "rand"
                | "randn"
                | "randi"
                | "eye"
                | "true"
                | "false"
        ) && args.iter().any(|arg| {
            crate::builtins::common::tensor::value_to_string(arg)
                .is_some_and(|text| text.eq_ignore_ascii_case("gpuarray"))
        });
        visit_gpu_handles_mut(result, &mut |handle| {
            if explicit_constructor {
                handle.descriptor.provenance =
                    Some(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            } else if runmat_accelerate_api::handle_provenance(handle).is_none() {
                handle.descriptor.provenance =
                    Some(runmat_accelerate_api::GpuHandleProvenance::Automatic);
            }
        });
        return;
    }
    let provenance = if explicit {
        runmat_accelerate_api::GpuHandleProvenance::Explicit
    } else {
        runmat_accelerate_api::GpuHandleProvenance::Automatic
    };
    visit_gpu_handles_mut(result, &mut |handle| {
        handle.descriptor.provenance = Some(provenance);
    });
}

fn visit_gpu_handles(value: &Value, visitor: &mut impl FnMut(&GpuTensorHandle)) {
    match value {
        Value::GpuTensor(handle) => visitor(handle),
        Value::Cell(cell) => cell
            .data
            .iter()
            .for_each(|value| visit_gpu_handles(value, visitor)),
        Value::Struct(value) => value
            .fields
            .values()
            .for_each(|value| visit_gpu_handles(value, visitor)),
        Value::Object(value) => value
            .properties
            .values()
            .for_each(|value| visit_gpu_handles(value, visitor)),
        Value::Closure(value) => value
            .captures
            .iter()
            .for_each(|value| visit_gpu_handles(value, visitor)),
        Value::OutputList(values) => values
            .iter()
            .for_each(|value| visit_gpu_handles(value, visitor)),
        _ => {}
    }
}

fn visit_gpu_handles_mut(value: &mut Value, visitor: &mut impl FnMut(&mut GpuTensorHandle)) {
    match value {
        Value::GpuTensor(handle) => visitor(handle),
        Value::Cell(cell) => cell
            .data
            .iter_mut()
            .for_each(|value| visit_gpu_handles_mut(value, visitor)),
        Value::Struct(value) => value
            .fields
            .values_mut()
            .for_each(|value| visit_gpu_handles_mut(value, visitor)),
        Value::Object(value) => value
            .properties
            .values_mut()
            .for_each(|value| visit_gpu_handles_mut(value, visitor)),
        Value::Closure(value) => value
            .captures
            .iter_mut()
            .for_each(|value| visit_gpu_handles_mut(value, visitor)),
        Value::OutputList(values) => values
            .iter_mut()
            .for_each(|value| visit_gpu_handles_mut(value, visitor)),
        _ => {}
    }
}

pub(crate) async fn try_call_registered_instance_method(
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
    let Some((method, owner)) = crate::class_registry::lookup_method(class_name, method_name)
    else {
        return Ok(None);
    };
    if method.is_static {
        return Ok(None);
    }
    let caller_class = current_class_access_context();
    let access_allowed = match method.access {
        runmat_types::MemberAccess::Public => true,
        runmat_types::MemberAccess::Private => caller_class.as_deref() == Some(owner.as_str()),
        runmat_types::MemberAccess::Protected => caller_class
            .as_deref()
            .is_some_and(|caller| crate::class_registry::is_class_or_subclass(caller, &owner)),
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
        return finalize_instance_method_result(method_name, receiver, result).map(Some);
    }
    if runmat_builtins::builtin_name_is_known(&method.function_name)
        && method.function_name != method_name
    {
        let result = call_builtin_async_impl(&method.function_name, args, output_count).await;
        return finalize_instance_method_result(method_name, receiver, result).map(Some);
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
            return finalize_instance_method_result(method_name, receiver, result).map(Some);
        }
        if runmat_builtins::builtin_name_is_known(&owner_qualified)
            && owner_qualified != method_name
        {
            let result = call_builtin_async_impl(&owner_qualified, args, output_count).await;
            return finalize_instance_method_result(method_name, receiver, result).map(Some);
        }
    }
    Ok(None)
}

fn finalize_instance_method_result(
    method_name: &str,
    receiver: &Value,
    result: Result<Value, RuntimeError>,
) -> Result<Value, RuntimeError> {
    let result = result?;
    if method_name == "delete" {
        if let Value::HandleObject(handle) = receiver {
            if !crate::set_handle_valid(handle, false) {
                return Err(build_runtime_error(format!(
                    "delete: failed to invalidate handle object '{}' after its destructor completed",
                    handle.class_name
                ))
                .with_identifier("RunMat:delete:InvalidHandle")
                .build());
            }
        }
    }
    Ok(result)
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
    if crate::class_registry::get_class(class_name).is_none() {
        return Ok(None);
    }
    let Some((method, owner)) = crate::class_registry::lookup_method(class_name, method_name)
    else {
        return Ok(None);
    };
    if !method.is_static || method.access != runmat_types::MemberAccess::Public {
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
    if runmat_builtins::builtin_name_is_known(&method.function_name)
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
        if runmat_builtins::builtin_name_is_known(&owner_qualified)
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
    let Some((ctor, owner)) =
        crate::class_registry::lookup_method(class_name, constructor_method_name)
            .or_else(|| crate::class_registry::lookup_method(class_name, class_name))
    else {
        return Ok(default_object);
    };
    let owner_qualified = format!("{owner}.{constructor_method_name}");
    let caller_class = current_class_access_context();
    let ctor_access_allowed = match ctor.access {
        runmat_types::MemberAccess::Public => true,
        runmat_types::MemberAccess::Private => caller_class.as_deref() == Some(owner.as_str()),
        runmat_types::MemberAccess::Protected => caller_class
            .as_deref()
            .is_some_and(|caller| crate::class_registry::is_class_or_subclass(caller, &owner)),
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
        if runmat_builtins::builtin_name_is_known(&ctor.function_name)
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
        if runmat_builtins::builtin_name_is_known(&owner_qualified) && owner_qualified != class_name
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
    if error_chain_has_gpu_gather_retry(err, crate::GpuGatherRetry::Never) {
        return false;
    }
    if args.iter().any(value_contains_explicit_gpu) {
        return false;
    }
    // Compatibility errors are policy decisions. Retain this source-chain
    // defense for wrappers that have not yet propagated an explicit policy.
    if error_chain_has_identifier_prefix(err, "RunMat:compatibility:") {
        return false;
    }
    error_chain_has_gpu_gather_retry(err, crate::GpuGatherRetry::Requested)
}

fn value_contains_explicit_gpu(value: &Value) -> bool {
    match value {
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_explicit(handle),
        Value::Cell(cell) => cell.data.iter().any(value_contains_explicit_gpu),
        Value::Struct(value) => value.fields.values().any(value_contains_explicit_gpu),
        Value::Object(value) => value.properties.values().any(value_contains_explicit_gpu),
        Value::Closure(value) => value.captures.iter().any(value_contains_explicit_gpu),
        Value::OutputList(values) => values.iter().any(value_contains_explicit_gpu),
        _ => false,
    }
}

fn error_chain_has_gpu_gather_retry(err: &RuntimeError, policy: crate::GpuGatherRetry) -> bool {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(err);
    while let Some(error) = current {
        if error
            .downcast_ref::<RuntimeError>()
            .is_some_and(|error| error.gpu_gather_retry() == policy)
        {
            return true;
        }
        current = error.source();
    }
    false
}

fn error_chain_has_identifier_prefix(err: &RuntimeError, prefix: &str) -> bool {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(err);
    while let Some(error) = current {
        if error
            .downcast_ref::<RuntimeError>()
            .and_then(RuntimeError::identifier)
            .is_some_and(|identifier| identifier.starts_with(prefix))
        {
            return true;
        }
        current = error.source();
    }
    false
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
    use super::{
        call_builtin, gather_if_needed_async, should_retry_with_gpu_gather, value_contains_gpu,
    };
    use futures::executor::block_on;
    use runmat_accelerate_api::{GpuTensorHandle, ThreadProviderGuard};
    use runmat_types::MemberAccess;
    use runmat_value::{Closure, StructValue, Value};
    use runmat_value::{IntegerStorage, Tensor};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_CLASS_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct EmptyBuiltinService;

    impl crate::context::RuntimeBuiltinService for EmptyBuiltinService {
        fn bindings_by_name(&self, _name: &str) -> Vec<crate::builtin::RuntimeBuiltinBinding> {
            Vec::new()
        }
    }

    #[test]
    fn catalog_backed_builtin_dispatches_without_legacy_authority() {
        assert!(runmat_builtins::builtin_function_by_name("full").is_none());
        let input = Value::Num(7.0);
        assert_eq!(
            call_builtin("full", std::slice::from_ref(&input)).unwrap(),
            input
        );
    }

    #[test]
    fn scoped_builtin_authority_does_not_fall_back_to_global_discovery() {
        let ports = crate::context::RuntimeServicePorts::default()
            .with_builtin(std::rc::Rc::new(EmptyBuiltinService));
        let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(ports);
        let _scope = runtime.enter();
        let error = call_builtin("full", &[Value::Num(7.0)]).expect_err("exact registry miss");
        assert_eq!(error.identifier(), Some("RunMat:UndefinedFunction"));
    }

    #[test]
    fn operation_floating_projection_uses_native_download_and_rejects_integers() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let single = Tensor::from_f32(vec![1.25, -2.5], vec![1, 2]).unwrap();
            let single_handle =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &single).unwrap();
            let projected = block_on(
                crate::builtins::common::gpu_helpers::download_floating_projection_async(
                    provider,
                    &single_handle,
                ),
            )
            .unwrap();
            assert_eq!(projected.data, vec![1.25, -2.5]);
            assert_eq!(projected.shape, vec![1, 2]);

            let integer =
                Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]), vec![1, 2])
                    .unwrap();
            let integer_handle =
                crate::builtins::common::gpu_helpers::upload_tensor(provider, &integer).unwrap();
            let error = block_on(
                crate::builtins::common::gpu_helpers::download_floating_projection_async(
                    provider,
                    &integer_handle,
                ),
            )
            .expect_err("floating projection must reject native integer storage");
            assert_eq!(
                error.identifier(),
                Some("RunMat:gpu:IntegerFloatingProjection")
            );

            for handle in [&single_handle, &integer_handle] {
                provider.free(handle).unwrap();
                runmat_accelerate_api::clear_handle_metadata(handle);
            }
        });
    }

    #[test]
    fn compatibility_errors_never_trigger_automatic_gpu_gather_retry() {
        let gpu = Value::GpuTensor(GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 1,
            descriptor: Default::default(),
        });
        let compatibility_error =
            crate::build_runtime_error("example gpuArray call form is a RunMat extension")
                .with_identifier("RunMat:compatibility:ExampleExtension")
                .build();
        assert!(!should_retry_with_gpu_gather(
            &compatibility_error,
            std::slice::from_ref(&gpu)
        ));

        let wrapped_compatibility_error =
            crate::build_runtime_error("GPU implementation failed while checking the call")
                .with_identifier("RunMat:example:GpuFailure")
                .with_source(compatibility_error)
                .build();
        assert!(!should_retry_with_gpu_gather(
            &wrapped_compatibility_error,
            std::slice::from_ref(&gpu)
        ));
        let requested_compatibility_error =
            crate::build_runtime_error("host implementation requested")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Requested)
                .with_source(
                    crate::build_runtime_error("RunMat-only GPU form")
                        .with_identifier("RunMat:compatibility:ExampleExtension")
                        .build(),
                )
                .build();
        assert!(!should_retry_with_gpu_gather(
            &requested_compatibility_error,
            std::slice::from_ref(&gpu)
        ));

        let automatic_gpu = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 5,
            descriptor: Default::default(),
        };
        let automatic_gpu =
            automatic_gpu.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic);
        let ordinary_gpu_error = crate::build_runtime_error("GPU input requires host fallback")
            .with_identifier("RunMat:example:UnsupportedGpuPath")
            .build();
        assert!(!should_retry_with_gpu_gather(
            &ordinary_gpu_error,
            &[Value::GpuTensor(automatic_gpu.clone())]
        ));
        let automatic_gpu =
            automatic_gpu.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        assert!(!should_retry_with_gpu_gather(
            &ordinary_gpu_error,
            &[Value::GpuTensor(automatic_gpu.clone())]
        ));
        runmat_accelerate_api::clear_handle_metadata(&automatic_gpu);

        let terminal_gpu_error = crate::build_runtime_error("GPU input is semantically invalid")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build();
        assert!(!should_retry_with_gpu_gather(
            &terminal_gpu_error,
            &[Value::GpuTensor(GpuTensorHandle {
                shape: vec![1, 1],
                device_id: 0,
                buffer_id: 2,
                descriptor: Default::default(),
            })]
        ));

        let nested_terminal = crate::build_runtime_error("terminal provider decision")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build();
        let wrapped_terminal = crate::build_runtime_error("GPU implementation failed")
            .with_source(nested_terminal)
            .build();
        assert!(!should_retry_with_gpu_gather(
            &wrapped_terminal,
            &[Value::GpuTensor(GpuTensorHandle {
                shape: vec![1, 1],
                device_id: 0,
                buffer_id: 3,
                descriptor: Default::default(),
            })]
        ));

        let nested_request = crate::build_runtime_error("host implementation is required")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Requested)
            .build();
        let wrapped_request = crate::build_runtime_error("provider path unavailable")
            .with_source(nested_request)
            .build();
        let requested_handle = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 4,
            descriptor: Default::default(),
        };
        assert!(should_retry_with_gpu_gather(
            &wrapped_request,
            &[Value::GpuTensor(requested_handle.clone())]
        ));
        let requested_handle =
            requested_handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        assert!(!should_retry_with_gpu_gather(
            &wrapped_request,
            &[Value::GpuTensor(requested_handle.clone())]
        ));
        runmat_accelerate_api::clear_handle_metadata(&requested_handle);
    }

    #[test]
    fn builtin_result_provenance_follows_gpu_input_intent() {
        let explicit = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 91,
            descriptor: Default::default(),
        };
        let automatic = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 92,
            descriptor: Default::default(),
        };
        let explicit_result = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 93,
            descriptor: Default::default(),
        };
        let automatic_result = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 94,
            descriptor: Default::default(),
        };
        let explicit =
            explicit.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let automatic =
            automatic.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic);

        let mut explicit_value = Value::GpuTensor(explicit_result.clone());
        super::propagate_gpu_provenance(
            "plus",
            &[Value::GpuTensor(explicit.clone())],
            &mut explicit_value,
        );
        let mut automatic_value = Value::GpuTensor(automatic_result.clone());
        super::propagate_gpu_provenance(
            "plus",
            &[Value::GpuTensor(automatic.clone())],
            &mut automatic_value,
        );

        assert_eq!(
            runmat_accelerate_api::handle_provenance(&explicit_result),
            None
        );
        let Value::GpuTensor(explicit_value) = explicit_value else {
            unreachable!()
        };
        assert_eq!(
            explicit_value.descriptor.provenance,
            Some(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        );
        assert_eq!(
            runmat_accelerate_api::handle_provenance(&automatic_result),
            None
        );
        let Value::GpuTensor(automatic_value) = automatic_value else {
            unreachable!()
        };
        assert_eq!(
            automatic_value.descriptor.provenance,
            Some(runmat_accelerate_api::GpuHandleProvenance::Automatic)
        );
        for handle in [&explicit, &automatic, &explicit_result, &automatic_result] {
            runmat_accelerate_api::clear_handle_metadata(handle);
        }
    }

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
                descriptor: Default::default(),
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
                descriptor: Default::default(),
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
            descriptor: Default::default(),
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
                descriptor: Default::default(),
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
            crate::class_registry::RuntimeMethod {
                name: child_name.clone(),
                is_static: true,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: ctor_fn_name_for_invoker,
                implicit_class_argument: None,
            },
        );
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: parent_name.clone(),
            parent: None,
            properties: HashMap::new(),
            methods: parent_methods,
        });
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
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
            crate::class_registry::RuntimeMethod {
                name: private_class_name.clone(),
                is_static: true,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Private,
                function_name: "Point.origin".to_string(),
                implicit_class_argument: None,
            },
        );
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
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
            crate::class_registry::RuntimeMethod {
                name: public_class_name.clone(),
                is_static: true,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: unique_class_name("runtime_ctor_missing_body"),
                implicit_class_argument: None,
            },
        );
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
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
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: class_name.clone(),
            parent: None,
            properties: HashMap::new(),
            methods: {
                let mut methods = HashMap::new();
                methods.insert(
                    "zero".to_string(),
                    crate::class_registry::RuntimeMethod {
                        name: "zero".to_string(),
                        is_static: true,
                        is_abstract: false,
                        is_sealed: false,
                        access: MemberAccess::Public,
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
