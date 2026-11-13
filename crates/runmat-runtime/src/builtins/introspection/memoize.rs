//! MATLAB-compatible `memoize` and `MemoizedFunction` support.
pub use runmat_builtins::{
    CLEAR_ALL_MEMOIZED_CACHES_DESCRIPTOR, CLEAR_ALL_MEMOIZED_CACHES_ERROR_TOO_MANY_OUTPUTS,
    CLEAR_CACHE_DESCRIPTOR, CLEAR_CACHE_ERROR_TOO_MANY_OUTPUTS, MEMOIZE_DESCRIPTOR,
    MEMOIZE_ERROR_GC, MEMOIZE_ERROR_INVALID_CACHE_SIZE, MEMOIZE_ERROR_INVALID_FUNCTION,
    MEMOIZE_ERROR_INVALID_INDEX, MEMOIZE_ERROR_INVALID_OBJECT, MEMOIZE_ERROR_TOO_MANY_OUTPUTS,
    MEMOIZE_INTEGER_AUDIT, STATS_DESCRIPTOR, STATS_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_types::MemberAccess;

use std::cell::RefCell;
use std::collections::HashMap;

#[cfg(test)]
use once_cell::sync::Lazy;
#[cfg(test)]
use std::sync::Mutex;

use runmat_builtins::BuiltinErrorDescriptor;
#[cfg(test)]
use runmat_builtins::BuiltinIntegerAuditKind;
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, ComplexTensor, HandleRef, IntValue, LogicalArray, NumericScalar, ObjectInstance,
    OutputListStorage, SharedShape, SparseTensor, StringArray, StructValue, Tensor, Value,
};

use crate::{
    build_runtime_error, builtins::common::tensor, BuiltinResult, RuntimeError,
    OBJECT_SUBSREF_METHOD,
};

pub(crate) use runmat_builtins::MEMOIZED_FUNCTION_CLASS;

const FUNCTION_PROPERTY: &str = "Function";
const ENABLED_PROPERTY: &str = "Enabled";
const CACHE_SIZE_PROPERTY: &str = "CacheSize";
const CACHE_PROPERTY: &str = "__runmat_memoize_cache";
const TOTAL_HITS_PROPERTY: &str = "__runmat_memoize_total_hits";
const TOTAL_MISSES_PROPERTY: &str = "__runmat_memoize_total_misses";
const DEFAULT_CACHE_SIZE: usize = 10;

#[cfg(test)]
mod scoped_cache_tests {
    use super::*;
    use crate::execution::{scoped_callable, RootedValueSequence};
    use std::ops::ControlFlow;

    #[test]
    fn cache_hit_cannot_revive_an_expired_scoped_evaluator() {
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        let payload = || Value::Cell(CellArray::new(vec![Value::Num(3.0)], 1, 1).unwrap());
        let mut calls = 0;
        let result = futures::executor::block_on(scoped_callable::drive(
            None,
            None,
            |handle, _| {
                Ok(async move {
                    let wrapper =
                        memoize_builtin(None, Value::ScopedFunctionHandle(handle)).await?;
                    let rooted = crate::execution::RootedValue::new(wrapper)
                        .map_err(|error| error.into_runtime_error())?;
                    let handle = memoized_handle(rooted.value())?;
                    for _ in 0..2 {
                        let result = call_memoized(None, handle, payload()).await?;
                        assert_eq!(result.value(), &Value::Num(9.0));
                    }
                    RootedValueSequence::new(
                        runmat_value::ValueSequence::single(rooted.value().clone())
                            .map_err(crate::sequence::sequence_error_to_runtime)?,
                    )
                    .map_err(|error| error.into_runtime_error())
                })
            },
            async |_| {
                calls += 1;
                Ok(ControlFlow::<(), _>::Continue(
                    RootedValueSequence::new(
                        runmat_value::ValueSequence::single(Value::Num(9.0))
                            .map_err(crate::sequence::sequence_error_to_runtime)?,
                    )
                    .map_err(|error| error.into_runtime_error())?,
                ))
            },
        ))
        .unwrap();
        let ControlFlow::Continue(result) = result else {
            panic!("completed child")
        };
        assert_eq!(calls, 1, "the second live call used the existing cache");
        let handle = memoized_handle(&result.as_slice()[0]).unwrap();
        let error =
            futures::executor::block_on(call_memoized(None, handle, payload())).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:ScopedCallable:Expired"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(calls, 1);
    }
}

#[cfg(test)]
pub(crate) static MEMOIZE_TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[derive(Debug, Clone)]
struct MemoizedCacheEntry {
    inputs: OutputListStorage,
    nargout: usize,
    output: Value,
    hit_count: usize,
}

thread_local! {
    static MEMOIZE_REGISTRY: RefCell<HashMap<String, runmat_gc::GcHandle>> = RefCell::new(HashMap::new());
}

pub use runmat_builtins::MEMOIZED_SUBSREF_DESCRIPTOR;

#[runtime_builtin(
    name = "memoize",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::memoize"
)]
pub(crate) async fn memoize_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    function: Value,
) -> BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &MEMOIZE_DESCRIPTOR,
        &MEMOIZE_ERROR_TOO_MANY_OUTPUTS,
        "memoize",
    )?;
    let function = canonicalize_memoized_function(function, runtime.as_ref())?;
    ensure_memoized_function_class_registered();
    let key = function_key(&function);

    MEMOIZE_REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        if let Some(target) = registry.get(&key).copied() {
            let existing = runmat_gc::gc_with_value(&target, |target_value| {
                matches!(
                    target_value,
                    Value::Object(object)
                        if object.class_name().is(MEMOIZED_FUNCTION_CLASS)
                            && object.property(FUNCTION_PROPERTY) == Some(&function)
                            && matches!(
                                object.property(crate::HANDLE_VALID_FLAG_PROPERTY),
                                Some(Value::Bool(true))
                            )
                )
            })
            .unwrap_or(false);
            if existing {
                return Ok(handle_from_target(target));
            }
            registry.remove(&key);
        }

        let mut object = runmat_value::UnboundedObjectBuilder::new(MEMOIZED_FUNCTION_CLASS);
        object.insert(FUNCTION_PROPERTY.to_string(), function.clone());
        object.insert(ENABLED_PROPERTY.to_string(), Value::Bool(true));
        object.insert(
            CACHE_SIZE_PROPERTY.to_string(),
            Value::Num(DEFAULT_CACHE_SIZE as f64),
        );
        object.insert(
            crate::HANDLE_VALID_FLAG_PROPERTY.to_string(),
            Value::Bool(true),
        );
        object.insert(CACHE_PROPERTY.to_string(), empty_cache_value()?);
        object.insert(TOTAL_HITS_PROPERTY.to_string(), Value::Num(0.0));
        object.insert(TOTAL_MISSES_PROPERTY.to_string(), Value::Num(0.0));

        let target = runmat_gc::gc_allocate(Value::Object(object.finish()))
            .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?;
        let handle = handle_from_target(target);
        registry.insert(key, target);
        Ok(handle)
    })
}

#[runtime_builtin(
    name = "MemoizedFunction.subsref",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::memoize"
)]
pub(crate) async fn memoized_subsref_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    receiver: Value,
    subscript: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    let read_runtime = runtime.clone();
    let read_first = move |receiver, step| memoized_read_step(read_runtime, receiver, step);
    match runtime.as_ref() {
        Some(runtime) => {
            crate::object::protocol::execute_owned_subsref_in_context(
                runtime, receiver, path, read_first, None,
            )
            .await
        }
        None => {
            crate::object::protocol::execute_owned_subsref(receiver, path, read_first, None).await
        }
    }
}

async fn memoized_read_step(
    runtime: Option<crate::context::RuntimeContext>,
    receiver: Value,
    step: crate::object::indexing::ObjectSubscript,
) -> BuiltinResult<crate::execution::RootedValue> {
    let payload = step.selector_value()?;
    let handle = memoized_handle(&receiver)?;
    match step.kind() {
        crate::object::indexing::ObjectIndexKind::Paren => {
            call_memoized(runtime.as_ref(), handle, payload).await
        }
        crate::object::indexing::ObjectIndexKind::Member => {
            crate::execution::RootedValue::new(memoized_member(handle, payload)?)
                .map_err(|error| error.into_runtime_error())
        }
        crate::object::indexing::ObjectIndexKind::Brace => Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_INDEX,
            "memoize: brace indexing is not supported",
        )),
    }
}

#[runtime_builtin(
    name = "clearCache",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::memoize"
)]
pub(crate) async fn clear_cache_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    receiver: Value,
) -> BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &CLEAR_CACHE_DESCRIPTOR,
        &CLEAR_CACHE_ERROR_TOO_MANY_OUTPUTS,
        "clearCache",
    )?;
    let handle = memoized_handle(&receiver)?;
    preflight_cache_owner(runtime.as_ref(), "clearCache")?;
    clear_cache_for_handle(handle)?;
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "stats",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::memoize"
)]
pub(crate) async fn stats_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    receiver: Value,
) -> BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &STATS_DESCRIPTOR,
        &STATS_ERROR_TOO_MANY_OUTPUTS,
        "stats",
    )?;
    let handle = memoized_handle(&receiver)?;
    preflight_cache_owner(runtime.as_ref(), "stats")?;
    stats_for_handle(handle)
}

#[runtime_builtin(
    name = "clearAllMemoizedCaches",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::memoize"
)]
pub(crate) async fn clear_all_memoized_caches_builtin(
    runtime: Option<crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &CLEAR_ALL_MEMOIZED_CACHES_DESCRIPTOR,
        &CLEAR_ALL_MEMOIZED_CACHES_ERROR_TOO_MANY_OUTPUTS,
        "clearAllMemoizedCaches",
    )?;
    preflight_cache_owner(runtime.as_ref(), "clearAllMemoizedCaches")?;
    MEMOIZE_REGISTRY.with(|registry| -> BuiltinResult<()> {
        let mut registry = registry.borrow_mut();
        let mut failure = None;
        registry.retain(|_, target| {
            if failure.is_some() {
                return true;
            }
            match runmat_gc::gc_with_value_mut(target, |target_value| -> BuiltinResult<bool> {
                let Value::Object(object) = target_value else {
                    return Ok(false);
                };
                if !object.class_name().is(MEMOIZED_FUNCTION_CLASS) {
                    return Ok(false);
                }
                edit_memoized_object(target_value, reset_object_cache)?;
                Ok(true)
            }) {
                Ok(Ok(retain)) => retain,
                Ok(Err(error)) => {
                    failure = Some(error);
                    true
                }
                Err(runmat_gc::GcError::InvalidPointer(_)) => false,
                Err(source) => {
                    failure = Some(
                        crate::build_runtime_error(format!(
                            "{}: {source}",
                            MEMOIZE_ERROR_GC.message
                        ))
                        .with_builtin("clearAllMemoizedCaches")
                        .with_identifier(
                            MEMOIZE_ERROR_GC
                                .identifier
                                .expect("canonical GC identifier"),
                        )
                        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                        .with_source(source)
                        .build(),
                    );
                    true
                }
            }
        });
        match failure {
            Some(mut error) => {
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                Err(error)
            }
            None => Ok(()),
        }
    })?;
    Ok(Value::Num(0.0))
}

/// No finite cache/snapshot owner exists yet. Refuse before callable
/// canonicalization, registry lookup or mutation rather than borrow ambient
/// allocation authority or publish an uncharged cache object.
fn preflight_cache_owner(
    runtime: Option<&crate::context::RuntimeContext>,
    builtin: &'static str,
) -> BuiltinResult<()> {
    use crate::context::host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPolicy,
        HostAllocationReceipt,
    };
    let error = |source| {
        let mut error = host_allocation_runtime_error(
            runtime,
            "RunMat:memoize:HostAllocation",
            "memoized cache requires complete finite ownership",
            source,
        );
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        crate::builtins::common::map_control_flow_with_builtin(error, builtin)
    };
    if HostAllocationReceipt::policy(runtime).map_err(error)? == HostAllocationPolicy::Managed {
        return Err(error(HostAllocationError::MissingManagedOwner));
    }
    if runtime.is_some_and(crate::context::RuntimeContext::is_cancelled) {
        return Err(error(HostAllocationError::Cancelled));
    }
    Ok(())
}

fn ensure_memoized_function_class_registered() {
    if crate::class_registry::get_class(&MEMOIZED_FUNCTION_CLASS.owned()).is_some() {
        return;
    }

    let mut properties = HashMap::new();
    for (name, set_access, default_value) in [
        (FUNCTION_PROPERTY, MemberAccess::Private, None),
        (
            ENABLED_PROPERTY,
            MemberAccess::Public,
            Some(Value::Bool(true)),
        ),
        (
            CACHE_SIZE_PROPERTY,
            MemberAccess::Public,
            Some(Value::Num(DEFAULT_CACHE_SIZE as f64)),
        ),
    ] {
        properties.insert(
            name.into(),
            crate::class_registry::RuntimeProperty {
                name: name.into(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access,
                default_value,
            },
        );
    }

    let mut methods = HashMap::new();
    for (method_name, function_name) in [
        (OBJECT_SUBSREF_METHOD, "MemoizedFunction.subsref"),
        (
            runmat_types::StaticMethodName::new("clearCache"),
            "clearCache",
        ),
        (runmat_types::StaticMethodName::new("stats"), "stats"),
    ] {
        methods.insert(
            method_name.into(),
            crate::class_registry::RuntimeMethod {
                name: method_name.into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: function_name.into(),
                implicit_class_argument: None,
                role: runmat_types::ClassMethodRole::Ordinary,
            },
        );
    }

    crate::class_registry::register_builtin_class(
        MEMOIZED_FUNCTION_CLASS,
        crate::class_registry::RuntimeClass {
            events: Default::default(),
            name: MEMOIZED_FUNCTION_CLASS.into(),
            parent: Some(runmat_types::standard::HANDLE.into()),
            properties,
            methods,
        },
    );
}

fn canonicalize_memoized_function(
    function: Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    match function {
        Value::ScopedFunctionHandle(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::Closure(_) => {
            preflight_cache_owner(runtime, "memoize")?;
            crate::canonicalize_callback_handle_for_semantic_resolution(function)
        }
        _ => Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_FUNCTION,
            MEMOIZE_ERROR_INVALID_FUNCTION.message,
        )),
    }
}

fn handle_from_target(target: runmat_gc::GcHandle) -> Value {
    Value::HandleObject(HandleRef {
        class_name: MEMOIZED_FUNCTION_CLASS.into(),
        target,
        valid: true,
    })
}

fn memoized_handle(value: &Value) -> BuiltinResult<&HandleRef> {
    match value {
        Value::HandleObject(handle)
            if handle.class_name == MEMOIZED_FUNCTION_CLASS && crate::is_handle_valid(handle) =>
        {
            Ok(handle)
        }
        Value::HandleObject(handle) if handle.class_name == MEMOIZED_FUNCTION_CLASS => {
            Err(memoize_error(
                &MEMOIZE_ERROR_INVALID_OBJECT,
                "memoize: MemoizedFunction handle is invalid",
            ))
        }
        other => Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_OBJECT,
            format!("memoize: expected MemoizedFunction object, got {other:?}"),
        )),
    }
}

fn memoized_state(handle: &HandleRef) -> BuiltinResult<(Value, bool, usize)> {
    runmat_gc::gc_with_value(&handle.target, |target| match target {
        Value::Object(object) if object.class_name().is(MEMOIZED_FUNCTION_CLASS) => {
            let function = object.property(FUNCTION_PROPERTY).cloned().ok_or_else(|| {
                memoize_error(
                    &MEMOIZE_ERROR_INVALID_OBJECT,
                    "memoize: missing Function property",
                )
            })?;
            let enabled = match object.property(ENABLED_PROPERTY) {
                Some(Value::Bool(value)) => *value,
                Some(Value::Num(value)) => *value != 0.0,
                Some(Value::Int(value)) => !value.is_zero(),
                Some(Value::LogicalArray(array)) if array.data.len() == 1 => array.data[0] != 0,
                Some(Value::Tensor(tensor)) if tensor::is_scalar_tensor(tensor) => {
                    if let Some(storage) = tensor.integer_storage() {
                        !storage.value_at(0).is_some_and(|value| value.is_zero())
                    } else {
                        tensor::tensor_value_f64(tensor, 0) != 0.0
                    }
                }
                _ => true,
            };
            let cache_size = parse_cache_size(object.property(CACHE_SIZE_PROPERTY))?;
            Ok((function, enabled, cache_size))
        }
        _ => Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_OBJECT,
            "memoize: invalid object target",
        )),
    })
    .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?
}

fn parse_cache_size(value: Option<&Value>) -> BuiltinResult<usize> {
    if let Some(Value::Int(v)) = value {
        return v.try_to_usize().filter(|value| *value >= 1).ok_or_else(|| {
            memoize_error(
                &MEMOIZE_ERROR_INVALID_CACHE_SIZE,
                MEMOIZE_ERROR_INVALID_CACHE_SIZE.message,
            )
        });
    }
    if let Some(Value::Tensor(t)) = value {
        if tensor::is_scalar_tensor(t) {
            if let Some(storage) = t.integer_storage() {
                return storage
                    .value_at(0)
                    .and_then(|value| value.try_to_usize())
                    .filter(|value| *value >= 1)
                    .ok_or_else(|| {
                        memoize_error(
                            &MEMOIZE_ERROR_INVALID_CACHE_SIZE,
                            MEMOIZE_ERROR_INVALID_CACHE_SIZE.message,
                        )
                    });
            }
        }
    }
    let numeric = match value {
        Some(Value::Num(v)) => *v,
        Some(Value::Tensor(t)) if tensor::is_scalar_tensor(t) => tensor::tensor_value_f64(t, 0),
        Some(Value::LogicalArray(a)) if a.data.len() == 1 => a.data[0] as f64,
        Some(Value::Bool(v)) => {
            if *v {
                1.0
            } else {
                0.0
            }
        }
        None => DEFAULT_CACHE_SIZE as f64,
        _ => {
            return Err(memoize_error(
                &MEMOIZE_ERROR_INVALID_CACHE_SIZE,
                MEMOIZE_ERROR_INVALID_CACHE_SIZE.message,
            ))
        }
    };
    if !numeric.is_finite() || numeric < 1.0 || numeric.fract() != 0.0 {
        return Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_CACHE_SIZE,
            MEMOIZE_ERROR_INVALID_CACHE_SIZE.message,
        ));
    }
    if numeric > usize::MAX as f64 || (usize::BITS == 64 && numeric == usize::MAX as f64) {
        return Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_CACHE_SIZE,
            MEMOIZE_ERROR_INVALID_CACHE_SIZE.message,
        ));
    }
    Ok(numeric as usize)
}

async fn call_memoized(
    runtime: Option<&crate::context::RuntimeContext>,
    handle: &HandleRef,
    payload: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    let Value::Cell(cell) = payload else {
        return Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_INDEX,
            "memoize: function invocation payload must be a cell array",
        ));
    };
    let args = cell.data_owner().clone();
    let requested_outputs = crate::current_requested_outputs();
    let (function, enabled, cache_size) = memoized_state(handle)?;

    // A cache hit must not revive an escaped invocation-scoped capability or
    // bypass its original-owner/cancellation/finite-policy admission.
    if let Value::ScopedFunctionHandle(handle) = &function {
        crate::execution::scoped_callable::validate_target(runtime, handle)?;
    }

    if enabled {
        if let Some(output) = take_cache_hit(handle, &args, requested_outputs, cache_size)? {
            return crate::execution::RootedValue::new(output)
                .map_err(|error| error.into_runtime_error());
        }
    }

    let output = match runtime {
        Some(runtime) => {
            crate::call_feval_async_with_outputs_in_context(
                runtime,
                function,
                &args,
                requested_outputs,
            )
            .await?
        }
        None => crate::call_feval_async_with_outputs(function, &args, requested_outputs).await?,
    };
    // Validate the requested prefix before publishing a cache entry. The
    // callable has already run exactly once; an output shortage is not cached.
    use crate::sequence::ResolveValueSequence;
    output.try_map_owned(
        |output| {
            let outputs = crate::call::arguments::adapt_legacy_builtin_result(output)?.resolve(
                runmat_types::SequenceUse::SelectPrefix {
                    count: requested_outputs,
                },
                crate::sequence::SequenceResolutionContext::default(),
            )?;
            let output = if requested_outputs == 1 {
                crate::sequence::into_first_output(outputs)?.expect("one memoized output")
            } else {
                Value::OutputList(outputs)
            };
            if enabled {
                store_cache_miss(handle, args, requested_outputs, output.clone(), cache_size)?;
            }
            Ok(output)
        },
        |error| error.into_runtime_error(),
    )
}

fn take_cache_hit(
    handle: &HandleRef,
    args: &[Value],
    requested_outputs: usize,
    cache_size: usize,
) -> BuiltinResult<Option<Value>> {
    runmat_gc::gc_with_value_mut(&handle.target, |target| -> BuiltinResult<Option<Value>> {
        edit_memoized_object(target, |object| {
            let mut cache = object_cache_value(object.property(CACHE_PROPERTY))?;
            trim_cache(&mut cache, cache_size);
            let mut hit = None;
            for entry in &mut cache {
                if entry.nargout == requested_outputs && values_equal_for_cache(&entry.inputs, args)
                {
                    entry.hit_count = entry.hit_count.saturating_add(1);
                    hit = Some(entry.output.clone());
                    break;
                }
            }
            if hit.is_some() {
                increment_counter(object, TOTAL_HITS_PROPERTY)?;
            }
            set_object_cache(object, cache)?;
            Ok(hit)
        })
    })
    .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?
}

fn store_cache_miss(
    handle: &HandleRef,
    args: OutputListStorage,
    requested_outputs: usize,
    output: Value,
    cache_size: usize,
) -> BuiltinResult<()> {
    runmat_gc::gc_with_value_mut(&handle.target, |target| -> BuiltinResult<()> {
        edit_memoized_object(target, |object| {
            let mut cache = object_cache_value(object.property(CACHE_PROPERTY))?;
            increment_counter(object, TOTAL_MISSES_PROPERTY)?;
            cache.push(MemoizedCacheEntry {
                inputs: args,
                nargout: requested_outputs,
                output,
                hit_count: 0,
            });
            trim_cache(&mut cache, cache_size);
            set_object_cache(object, cache)?;
            Ok(())
        })
    })
    .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?
}

fn trim_cache(cache: &mut Vec<MemoizedCacheEntry>, cache_size: usize) {
    if cache.len() > cache_size {
        let remove = cache.len() - cache_size;
        cache.drain(0..remove);
    }
}

fn memoized_member(handle: &HandleRef, payload: Value) -> BuiltinResult<Value> {
    let member = member_name(payload)?;
    match member.as_str() {
        "stats" => stats_for_handle(handle),
        "clearCache" => {
            clear_cache_for_handle(handle)?;
            Ok(Value::Num(0.0))
        }
        FUNCTION_PROPERTY | ENABLED_PROPERTY | CACHE_SIZE_PROPERTY => {
            runmat_gc::gc_with_value(&handle.target, |target| match target {
                Value::Object(object) => object.property(member.as_str()).cloned(),
                _ => None,
            })
            .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?
            .ok_or_else(|| {
                memoize_error(
                    &MEMOIZE_ERROR_INVALID_INDEX,
                    format!("memoize: unknown MemoizedFunction member '{member}'"),
                )
            })
        }
        _ => Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_INDEX,
            format!("memoize: unknown MemoizedFunction member '{member}'"),
        )),
    }
}

fn member_name(value: Value) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(name)) => Ok(name.as_str().to_owned()),
        Value::CharArray(chars) => super::text::decode_utf16_with_error(chars.data(), || {
            memoize_error(
                &MEMOIZE_ERROR_INVALID_INDEX,
                "memoize: member selector is not valid UTF-16 text",
            )
        }),
        other => Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_INDEX,
            format!("memoize: invalid member selector {other:?}"),
        )),
    }
}

fn clear_cache_for_handle(handle: &HandleRef) -> BuiltinResult<()> {
    runmat_gc::gc_with_value_mut(&handle.target, |target| -> BuiltinResult<()> {
        edit_memoized_object(target, reset_object_cache)
    })
    .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?
}

fn stats_for_handle(handle: &HandleRef) -> BuiltinResult<Value> {
    runmat_gc::gc_with_value_mut(&handle.target, |target| -> BuiltinResult<Value> {
        edit_memoized_object(target, |object| {
            let cache_size = parse_cache_size(object.property(CACHE_SIZE_PROPERTY))?;
            let mut entries = object_cache_value(object.property(CACHE_PROPERTY))?;
            trim_cache(&mut entries, cache_size);
            set_object_cache(object, entries.clone())?;
            let total_hits = object_counter(object, TOTAL_HITS_PROPERTY);
            let total_misses = object_counter(object, TOTAL_MISSES_PROPERTY);

            let mut input_cells = Vec::with_capacity(entries.len());
            let mut nargout = Vec::with_capacity(entries.len());
            let mut output_cells = Vec::with_capacity(entries.len());
            let mut hit_count = Vec::with_capacity(entries.len());
            for entry in &entries {
                input_cells.push(Value::Cell(
                    CellArray::from_shared_storage(
                        entry.inputs.clone(),
                        SharedShape::new_unmanaged(vec![1, entry.inputs.len()]),
                    )
                    .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?,
                ));
                nargout.push(entry.nargout as f64);
                output_cells.push(Value::Cell(output_value_to_cell(
                    &entry.output,
                    entry.nargout,
                )?));
                hit_count.push(entry.hit_count as f64);
            }

            let cache_len = entries.len();
            let mut cache = runmat_value::UnboundedStructValueBuilder::new();
            cache.insert(
                "Inputs",
                Value::Cell(
                    CellArray::new(input_cells, 1, cache_len).map_err(|err| {
                        memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}"))
                    })?,
                ),
            );
            cache.insert(
                "Nargout",
                Value::Tensor(
                    Tensor::new(nargout, vec![1, cache_len]).map_err(|err| {
                        memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}"))
                    })?,
                ),
            );
            cache.insert(
                "Outputs",
                Value::Cell(
                    CellArray::new(output_cells, 1, cache_len).map_err(|err| {
                        memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}"))
                    })?,
                ),
            );
            cache.insert(
                "HitCount",
                Value::Tensor(
                    Tensor::new(hit_count, vec![1, cache_len]).map_err(|err| {
                        memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}"))
                    })?,
                ),
            );
            cache.insert("TotalHits", Value::Num(total_hits as f64));
            cache.insert("TotalMisses", Value::Num(total_misses as f64));

            let mut stats = runmat_value::UnboundedStructValueBuilder::new();
            stats.insert("Cache", Value::Struct(cache.finish()));
            Ok(Value::Struct(stats.finish()))
        })
    })
    .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?
}

fn edit_memoized_object<R>(
    target: &mut Value,
    edit: impl FnOnce(&mut runmat_value::UnboundedObjectBuilder) -> BuiltinResult<R>,
) -> BuiltinResult<R> {
    let Value::Object(object) = target else {
        return Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_OBJECT,
            "memoize: invalid object target",
        ));
    };
    if !object.class_name().is(MEMOIZED_FUNCTION_CLASS) {
        return Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_OBJECT,
            "memoize: invalid object target",
        ));
    }
    let previous = std::mem::replace(target, Value::Num(0.0));
    let Value::Object(object) = previous else {
        unreachable!("memoized target was checked as an object")
    };
    let mut object = object.into_unbounded_builder().map_err(|object| {
        *target = Value::Object(object);
        memoize_error(
            &MEMOIZE_ERROR_INVALID_OBJECT,
            "memoize: admitted object requires an admitted edit generation",
        )
    })?;
    let result = edit(&mut object);
    *target = Value::Object(object.finish());
    result
}

fn empty_cache_value() -> BuiltinResult<Value> {
    Ok(Value::Cell(CellArray::new(Vec::new(), 1, 0).map_err(
        |err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")),
    )?))
}

fn reset_object_cache(object: &mut runmat_value::UnboundedObjectBuilder) -> BuiltinResult<()> {
    object.insert(CACHE_PROPERTY.to_string(), empty_cache_value()?);
    object.insert(TOTAL_HITS_PROPERTY.to_string(), Value::Num(0.0));
    object.insert(TOTAL_MISSES_PROPERTY.to_string(), Value::Num(0.0));
    Ok(())
}

#[cfg(test)]
fn object_cache(object: &ObjectInstance) -> BuiltinResult<Vec<MemoizedCacheEntry>> {
    object_cache_value(object.property(CACHE_PROPERTY))
}

fn object_cache_value(value: Option<&Value>) -> BuiltinResult<Vec<MemoizedCacheEntry>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let Value::Cell(cell) = value else {
        return Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_OBJECT,
            "memoize: internal cache property is invalid",
        ));
    };
    cell.data().iter().map(decode_cache_entry).collect()
}

fn set_object_cache(
    object: &mut runmat_value::UnboundedObjectBuilder,
    entries: Vec<MemoizedCacheEntry>,
) -> BuiltinResult<()> {
    let values = entries
        .into_iter()
        .map(encode_cache_entry)
        .collect::<BuiltinResult<Vec<_>>>()?;
    let value_count = values.len();
    object.insert(
        CACHE_PROPERTY.to_string(),
        Value::Cell(
            CellArray::new(values, 1, value_count)
                .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?,
        ),
    );
    Ok(())
}

fn encode_cache_entry(entry: MemoizedCacheEntry) -> BuiltinResult<Value> {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    let input_len = entry.inputs.len();
    fields.insert(
        "Inputs",
        Value::Cell(
            CellArray::from_shared_storage(
                entry.inputs,
                SharedShape::new_unmanaged(vec![1, input_len]),
            )
            .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))?,
        ),
    );
    fields.insert("Nargout", Value::Num(entry.nargout as f64));
    fields.insert("Output", entry.output);
    fields.insert("HitCount", Value::Num(entry.hit_count as f64));
    Ok(Value::Struct(fields.finish()))
}

fn decode_cache_entry(value: &Value) -> BuiltinResult<MemoizedCacheEntry> {
    let Value::Struct(fields) = value else {
        return Err(memoize_error(
            &MEMOIZE_ERROR_INVALID_OBJECT,
            "memoize: internal cache entry is invalid",
        ));
    };
    let inputs = match fields.fields.get("Inputs") {
        Some(Value::Cell(cell)) => cell.data_owner().clone(),
        _ => {
            return Err(memoize_error(
                &MEMOIZE_ERROR_INVALID_OBJECT,
                "memoize: internal cache entry inputs are invalid",
            ))
        }
    };
    let nargout = numeric_counter(fields.fields.get("Nargout")).ok_or_else(|| {
        memoize_error(
            &MEMOIZE_ERROR_INVALID_OBJECT,
            "memoize: internal cache entry output count is invalid",
        )
    })?;
    let output = fields.fields.get("Output").cloned().ok_or_else(|| {
        memoize_error(
            &MEMOIZE_ERROR_INVALID_OBJECT,
            "memoize: internal cache entry output is missing",
        )
    })?;
    let hit_count = numeric_counter(fields.fields.get("HitCount")).unwrap_or(0);
    Ok(MemoizedCacheEntry {
        inputs,
        nargout,
        output,
        hit_count,
    })
}

fn increment_counter(
    object: &mut runmat_value::UnboundedObjectBuilder,
    name: &str,
) -> BuiltinResult<()> {
    let next = numeric_counter(object.property(name))
        .unwrap_or(0)
        .saturating_add(1);
    object.insert(name.to_string(), Value::Num(next as f64));
    Ok(())
}

fn object_counter(object: &runmat_value::UnboundedObjectBuilder, name: &str) -> usize {
    numeric_counter(object.property(name)).unwrap_or(0)
}

fn numeric_counter(value: Option<&Value>) -> Option<usize> {
    match value {
        Some(Value::Num(value)) => nonnegative_platform_usize(*value),
        Some(Value::Int(value)) => value.try_to_usize(),
        _ => None,
    }
}

fn nonnegative_platform_usize(value: f64) -> Option<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return None;
    }
    if value > usize::MAX as f64 || (usize::BITS == 64 && value == usize::MAX as f64) {
        return None;
    }
    Some(value as usize)
}

fn output_value_to_cell(value: &Value, requested_outputs: usize) -> BuiltinResult<CellArray> {
    let values = match value {
        Value::OutputList(values) if values.has_retained_claim() => {
            return Err(crate::sequence::sequence_error_to_runtime(
                runmat_value::ValueSequenceError::SharedOutputStorage,
            ));
        }
        Value::OutputList(values) => values.as_slice().to_vec(),
        _ if requested_outputs == 0 => Vec::new(),
        value => vec![value.clone()],
    };
    let cols = values.len();
    CellArray::new(values, 1, cols)
        .map_err(|err| memoize_error(&MEMOIZE_ERROR_GC, format!("memoize: {err}")))
}

fn values_equal_for_cache(lhs: &[Value], rhs: &[Value]) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(left, right)| value_equal_for_cache(left, right))
}

fn value_equal_for_cache(lhs: &Value, rhs: &Value) -> bool {
    match (lhs, rhs) {
        (Value::Num(a), Value::Num(b)) => floats_equal_nan(*a, *b),
        (Value::Int(a), Value::Int(b)) => ints_equal(a, b),
        (Value::Complex(ar, ai), Value::Complex(br, bi)) => {
            floats_equal_nan(*ar, *br) && floats_equal_nan(*ai, *bi)
        }
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::CharArray(a), Value::CharArray(b)) => {
            a.rows == b.rows && a.cols == b.cols && a.data() == b.data()
        }
        (Value::StringArray(a), Value::StringArray(b)) => string_arrays_equal(a, b),
        (Value::Tensor(a), Value::Tensor(b)) => tensors_equal(a, b),
        (Value::SparseTensor(a), Value::SparseTensor(b)) => sparse_tensors_equal(a, b),
        (Value::ComplexTensor(a), Value::ComplexTensor(b)) => complex_tensors_equal(a, b),
        (Value::LogicalArray(a), Value::LogicalArray(b)) => logical_arrays_equal(a, b),
        (Value::Cell(a), Value::Cell(b)) => cells_equal(a, b),
        (Value::Struct(a), Value::Struct(b)) => structs_equal(a, b),
        (Value::StructArray(a), Value::StructArray(b)) => {
            a.shape() == b.shape()
                && a.field_names().eq(b.field_names())
                && a.field_names().all(|name| {
                    a.field_values(name)
                        .zip(b.field_values(name))
                        .is_some_and(|(left, right)| {
                            left.len() == right.len()
                                && left
                                    .iter()
                                    .zip(right)
                                    .all(|(left, right)| value_equal_for_cache(left, right))
                        })
                })
        }
        (Value::Object(a), Value::Object(b)) => objects_equal(a, b),
        (Value::HandleObject(a), Value::HandleObject(b)) => a == b,
        (Value::Listener(a), Value::Listener(b)) => a == b,
        (Value::OutputList(a), Value::OutputList(b)) => values_equal_for_cache(a, b),
        (Value::FunctionHandle(a), Value::FunctionHandle(b))
        | (Value::ExternalFunctionHandle(a), Value::ExternalFunctionHandle(b))
        | (Value::MethodFunctionHandle(a), Value::MethodFunctionHandle(b)) => a == b,
        (Value::ClassRef(a), Value::ClassRef(b)) => a == b,
        (Value::Symbolic(a), Value::Symbolic(b)) => a == b,
        (Value::BoundFunctionHandle(a), Value::BoundFunctionHandle(b)) => a == b,
        (Value::Closure(a), Value::Closure(b)) => {
            a.function_name == b.function_name
                && a.binding == b.binding
                && a.source_role == b.source_role
                && a.source_file == b.source_file
                && values_equal_for_cache(&a.captures, &b.captures)
        }
        (Value::MException(a), Value::MException(b)) => a == b,
        (Value::GpuTensor(a), Value::GpuTensor(b)) => a == b,
        _ => false,
    }
}

fn floats_equal_nan(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

fn ints_equal(a: &IntValue, b: &IntValue) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b) && a == b
}

fn tensors_equal(a: &Tensor, b: &Tensor) -> bool {
    if a.shape() != b.shape() || a.numeric_dtype() != b.numeric_dtype() || a.len() != b.len() {
        return false;
    }
    (0..a.len()).all(|index| {
        numeric_scalars_equal_nan(
            a.numeric_value_at(index)
                .expect("validated tensor storage length"),
            b.numeric_value_at(index)
                .expect("validated tensor storage length"),
        )
    })
}

fn sparse_tensors_equal(a: &SparseTensor, b: &SparseTensor) -> bool {
    if a.rows != b.rows
        || a.cols != b.cols
        || a.col_ptrs != b.col_ptrs
        || a.row_indices != b.row_indices
        || a.nnz() != b.nnz()
    {
        return false;
    }
    match (a.integer_storage(), b.integer_storage()) {
        (Some(a), Some(b)) => return a == b,
        (Some(_), None) | (None, Some(_)) => return false,
        (None, None) => {}
    }
    if a.is_complex() || b.is_complex() {
        if !a.is_complex() || !b.is_complex() || a.numeric_dtype() != b.numeric_dtype() {
            return false;
        }
        return (0..a.nnz()).all(|index| {
            let a = a
                .complex_value_at(index)
                .expect("complex sparse storage is consistent");
            let b = b
                .complex_value_at(index)
                .expect("complex sparse storage is consistent");
            floats_equal_nan(a.0, b.0) && floats_equal_nan(a.1, b.1)
        });
    }
    if a.numeric_dtype() != b.numeric_dtype() {
        return false;
    }
    a.materialize_f64()
        .iter()
        .zip(b.materialize_f64())
        .all(|(x, y)| floats_equal_nan(*x, y))
}

fn complex_tensors_equal(a: &ComplexTensor, b: &ComplexTensor) -> bool {
    let len = tensor::complex_tensor_element_len(a);
    if a.shape() != b.shape() || len != tensor::complex_tensor_element_len(b) {
        return false;
    }
    (0..len).all(|index| {
        let (a_real, a_imag) = a
            .numeric_value_at(index)
            .expect("validated complex tensor storage length");
        let (b_real, b_imag) = b
            .numeric_value_at(index)
            .expect("validated complex tensor storage length");
        numeric_scalars_equal_nan(a_real, b_real) && numeric_scalars_equal_nan(a_imag, b_imag)
    })
}

fn numeric_scalars_equal_nan(a: NumericScalar, b: NumericScalar) -> bool {
    match (a, b) {
        (NumericScalar::F64(a), NumericScalar::F64(b)) => floats_equal_nan(a, b),
        (NumericScalar::F32(a), NumericScalar::F32(b)) => a == b || (a.is_nan() && b.is_nan()),
        _ => a == b,
    }
}

fn string_arrays_equal(a: &StringArray, b: &StringArray) -> bool {
    a.shape == b.shape && a.data() == b.data()
}

fn logical_arrays_equal(a: &LogicalArray, b: &LogicalArray) -> bool {
    a.shape == b.shape && a.data == b.data
}

fn cells_equal(a: &CellArray, b: &CellArray) -> bool {
    a.shape == b.shape && values_equal_for_cache(a.data(), b.data())
}

fn structs_equal(a: &StructValue, b: &StructValue) -> bool {
    a.fields.len() == b.fields.len()
        && a.fields
            .iter()
            .zip(&b.fields)
            .all(|((left_name, left), (right_name, right))| {
                left_name == right_name && value_equal_for_cache(left, right)
            })
}

fn objects_equal(a: &ObjectInstance, b: &ObjectInstance) -> bool {
    a.class_name() == b.class_name() && structs_equal(&object_properties(a), &object_properties(b))
}

fn object_properties(object: &ObjectInstance) -> StructValue {
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    for (name, value) in object.properties().iter() {
        fields.insert(name.to_owned(), value.clone());
    }
    fields.finish()
}

fn function_key(function: &Value) -> String {
    match function {
        Value::ScopedFunctionHandle(handle) => format!("scoped:{:?}", handle.instance_id()),
        Value::FunctionHandle(name) => format!("fh:{name}"),
        Value::ExternalFunctionHandle(name) => format!("external:{name}"),
        Value::MethodFunctionHandle(name) => format!("method:{name}"),
        Value::BoundFunctionHandle(handle) => {
            format!("bound:{}:{}", handle.function, handle.name)
        }
        Value::Closure(closure) => format!(
            "closure:{}:{:?}:{}",
            closure.function_name,
            closure.bound_function(),
            values_fingerprint(&closure.captures)
        ),
        other => format!("other:{other:?}"),
    }
}

fn values_fingerprint(values: &[Value]) -> String {
    values
        .iter()
        .map(value_fingerprint)
        .collect::<Vec<_>>()
        .join("|")
}

fn value_fingerprint(value: &Value) -> String {
    match value {
        Value::Num(value) if value.is_nan() => "num:NaN".to_string(),
        Value::Tensor(tensor) => {
            let values = (0..tensor.len())
                .map(|index| {
                    tensor
                        .numeric_value_at(index)
                        .expect("validated tensor storage length")
                })
                .collect::<Vec<_>>();
            format!(
                "tensor:{:?}:{:?}:{values:?}",
                tensor.numeric_dtype(),
                tensor.shape()
            )
        }
        Value::SparseTensor(tensor) => match tensor.integer_storage() {
            Some(storage) => format!(
                "sparse:{}x{}:{:?}:{:?}:{storage:?}",
                tensor.rows, tensor.cols, tensor.col_ptrs, tensor.row_indices
            ),
            None if tensor.is_complex() => format!(
                "sparse-complex:{}x{}:{:?}:{:?}:{:?}:{:?}",
                tensor.rows,
                tensor.cols,
                tensor.col_ptrs,
                tensor.row_indices,
                tensor.numeric_dtype(),
                tensor
                    .materialize_complex_f64()
                    .expect("complex sparse storage")
            ),
            None => format!(
                "sparse:{}x{}:{:?}:{:?}:{:?}:{:?}",
                tensor.rows,
                tensor.cols,
                tensor.col_ptrs,
                tensor.row_indices,
                tensor.numeric_dtype(),
                tensor.materialize_f64()
            ),
        },
        Value::ComplexTensor(tensor) => {
            let values = (0..tensor::complex_tensor_element_len(tensor))
                .map(|index| {
                    tensor
                        .numeric_value_at(index)
                        .expect("validated complex tensor storage length")
                })
                .collect::<Vec<_>>();
            format!("complex:{:?}:{values:?}", tensor.shape())
        }
        Value::Cell(cell) => format!("cell:{:?}:{}", cell.shape, values_fingerprint(cell.data())),
        Value::Struct(struct_value) => format!(
            "struct:{}",
            struct_value
                .fields
                .iter()
                .map(|(name, value)| format!("{name}={}", value_fingerprint(value)))
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::StructArray(array) => format!(
            "struct-array:{:?}:{}:{}",
            array.shape(),
            array
                .field_names()
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
                .join(","),
            (0..array.len())
                .map(|index| array
                    .field_names()
                    .filter_map(|name| array
                        .field_values(name)
                        .and_then(|values| values.get(index)))
                    .map(value_fingerprint)
                    .collect::<Vec<_>>()
                    .join(","))
                .collect::<Vec<_>>()
                .join(";")
        ),
        Value::Closure(closure) => format!(
            "closure:{}:{:?}:{}",
            closure.function_name,
            closure.bound_function(),
            values_fingerprint(&closure.captures)
        ),
        other => format!("{other:?}"),
    }
}

fn memoize_error(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("memoize");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
pub(crate) fn reset_memoize_registry_for_test() {
    MEMOIZE_REGISTRY.with(|registry| {
        registry.borrow_mut().clear();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_public_cache_entries_refuse_before_mutation_and_keep_control_category() {
        use crate::context::host_effect::ManagedHostEffectLedger;
        use crate::context::{
            ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
            RuntimeServicePorts,
        };
        use crate::execution::ExecutionServiceError;
        use std::{error::Error, rc::Rc};
        struct Managed(ManagedHostEffectLedger);
        impl RuntimeParallelService for Managed {
            fn supports(&self, _: ParallelCapability) -> bool {
                false
            }
            fn placement_resources(&self) -> RuntimeParallelResources {
                let snapshot = self.0.snapshot();
                RuntimeParallelResources {
                    memory_available_bytes: Some(snapshot.free_bytes()),
                    epoch: snapshot.epoch,
                    ..Default::default()
                }
            }
            fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
                Some(self.0.clone())
            }
        }
        let ledger = ManagedHostEffectLedger::new(1 << 20);
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        let (receiver, _binding) = memoized();
        let context =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(
                    RuntimeServicePorts::default().with_parallel(Rc::new(Managed(ledger.clone()))),
                );
        for error in [
            futures::executor::block_on(memoize_builtin(
                Some(context.clone()),
                Value::function_handle("not_resolved_before_admission".into()),
            ))
            .unwrap_err(),
            futures::executor::block_on(clear_cache_builtin(
                Some(context.clone()),
                receiver.clone(),
            ))
            .unwrap_err(),
            futures::executor::block_on(stats_builtin(Some(context.clone()), receiver))
                .unwrap_err(),
            futures::executor::block_on(clear_all_memoized_caches_builtin(Some(context.clone())))
                .unwrap_err(),
        ] {
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert!(matches!(
                error
                    .source()
                    .and_then(|source| source.downcast_ref::<ExecutionServiceError>()),
                Some(ExecutionServiceError::Infrastructure(_))
            ));
        }
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        let error = futures::executor::block_on(memoize_builtin(Some(context), Value::Num(0.0)))
            .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:memoize:InvalidFunction"));
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let cancelled =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_cancellation_token(cancellation.token());
        cancelled.set_error_namespace("CacheScope");
        cancellation.cancel(runmat_execution::CancellationReason::User);
        let error = futures::executor::block_on(memoize_builtin(
            Some(cancelled),
            Value::function_handle("not_resolved_before_admission".into()),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("CacheScope:Cancelled"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(matches!(
            error
                .source()
                .and_then(|source| source.downcast_ref::<ExecutionServiceError>()),
            Some(ExecutionServiceError::Cancelled)
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn cache_keys_distinguish_scalar_vector_and_matrix_shapes() {
        let values = [vec![], vec![1], vec![1, 1]]
            .into_iter()
            .map(|shape| Value::Tensor(Tensor::new(vec![7.0], shape).unwrap()))
            .collect::<Vec<_>>();
        for (index, left) in values.iter().enumerate() {
            assert!(value_equal_for_cache(left, &left.clone()));
            for right in &values[index + 1..] {
                assert!(!value_equal_for_cache(left, right));
                assert_ne!(value_fingerprint(left), value_fingerprint(right));
            }
        }
    }

    async fn memoized_subsref_builtin(
        receiver: Value,
        kind: String,
        payload: Value,
    ) -> crate::BuiltinResult<crate::execution::RootedValue> {
        let subscript =
            crate::object::indexing::standard_substruct_fixture_from_parts(&kind, payload)?;
        let runtime = exact_memoized_runtime();
        super::memoized_subsref_builtin(Some(runtime), receiver, subscript).await
    }
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerComplexStorage, IntegerStorage};
    use std::sync::{Arc, Mutex};

    use crate::OBJECT_INDEX_MEMBER;

    #[test]
    fn memoize_constructor_is_integer_inapplicable() {
        assert_eq!(
            MEMOIZE_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
        assert!(MEMOIZE_INTEGER_AUDIT.canonical_builtin.is_none());
    }

    fn counting_invoker(
        counter: Arc<Mutex<usize>>,
    ) -> Arc<crate::user_functions::UnscopedFunctionInvoker> {
        Arc::new(move |_function, args, requested_outputs| {
            let counter = Arc::clone(&counter);
            let args = args.to_vec();
            Box::pin(async move {
                *counter.lock().unwrap() += 1;
                let base = match args.first() {
                    Some(Value::Num(value)) => *value,
                    Some(Value::Tensor(tensor)) if !tensor.is_empty() => {
                        tensor::tensor_value_f64(tensor, 0)
                    }
                    _ => 0.0,
                };
                let sequence = if requested_outputs == 2 {
                    crate::sequence::ValueSequence::comma_separated(vec![
                        Value::Num(base + 1.0),
                        Value::Num(base + 2.0),
                    ])
                    .map_err(crate::sequence::sequence_error_to_runtime)
                } else {
                    crate::sequence::single_value_sequence(Value::Num(base + 1.0))
                }?;
                crate::execution::RootedValueSequence::new(sequence)
                    .map_err(|error| error.into_runtime_error())
            })
        })
    }

    #[test]
    fn clear_cache_is_integer_inapplicable_and_has_no_public_output() {
        assert!(CLEAR_CACHE_DESCRIPTOR.signatures[0].outputs.is_empty());
        for value in [
            IntValue::I8(-1),
            IntValue::I16(-1),
            IntValue::I32(-1),
            IntValue::I64(-1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(u64::MAX),
        ] {
            assert!(block_on(clear_cache_builtin(None, Value::Int(value))).is_err());
        }
    }

    fn echo_first_invoker() -> Arc<crate::user_functions::UnscopedFunctionInvoker> {
        Arc::new(move |_function, args, _requested_outputs| {
            let first = args.first().cloned().unwrap_or(Value::Num(0.0));
            let result = crate::user_functions::test_support::rooted_single_value_sequence(first);
            Box::pin(async move { result })
        })
    }

    fn memoized() -> (Value, crate::user_functions::BoundFunctionInvokerGuard) {
        let origin = runmat_execution::FunctionBindingOrigin {
            program: runmat_execution::Digest::sha256("memoize exact unit fixture"),
            function: runmat_types::ProgramFunctionId(42),
        };
        let binding = runmat_value::BoundFunctionBinding::portable(42, origin);
        let expected = binding.clone();
        let guard = crate::user_functions::install_local_bound_function_invoker(std::rc::Rc::new(
            move |runtime, actual, arguments, outputs, _lexical_environment| {
                assert_eq!(actual, expected, "wrong memoized callback binding");
                assert_eq!(arguments.capture_count(), 0);
                let invoker = crate::user_functions::current_semantic_function_invoker()
                    .expect("numeric unit callback fixture");
                invoker(Some(runtime), actual.function, arguments, outputs)
            },
        ))
        .expect("call-port scope");
        let value = block_on(memoize_builtin(
            None,
            Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
                runmat_value::BoundFunctionHandleData {
                    name: "step".into(),
                    function: 42,
                    authority: binding.authority,
                    source_role: runmat_types::FunctionSourceRole::Unknown,
                    source_file: None,
                },
            )),
        ))
        .expect("memoized function");
        (value, guard)
    }

    fn call_feval_fixture(
        runtime: &crate::context::RuntimeContext,
        function: Value,
        args: &[Value],
        requested_outputs: usize,
    ) -> Result<crate::execution::RootedValue, crate::RuntimeError> {
        block_on(crate::call_feval_async_with_outputs_in_context(
            runtime,
            function,
            args,
            requested_outputs,
        ))
    }

    fn exact_memoized_runtime() -> crate::context::RuntimeContext {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        // This receiver is a RunMat extension; select the fixture's policy
        // on its explicit runtime rather than the ambient test context.
        runtime.set_runmat_extensions_enabled(true);
        {
            let _scope = runtime.enter();
            super::ensure_memoized_function_class_registered();
        }
        runtime
    }

    #[test]
    fn typed_cache_counters_preserve_platform_representable_uint64() {
        assert_eq!(
            numeric_counter(Some(&Value::Int(IntValue::U64(3)))),
            Some(3)
        );
        assert_eq!(
            numeric_counter(Some(&Value::Int(IntValue::U64(u64::MAX)))),
            usize::try_from(u64::MAX).ok()
        );
        assert_eq!(numeric_counter(Some(&Value::Int(IntValue::I64(-1)))), None);
        assert_eq!(numeric_counter(Some(&Value::Num(1.5))), None);
        assert_eq!(
            numeric_counter(Some(&Value::Num(usize::MAX as f64 + 1.0))),
            None
        );
    }

    #[test]
    fn parse_cache_size_reads_typed_integer_tensor_storage_exactly() {
        let tensor =
            Tensor::new_integer(IntegerStorage::U64(vec![3]), vec![1, 1]).expect("cache size");

        assert_eq!(
            parse_cache_size(Some(&Value::Tensor(tensor))).expect("cache size"),
            3
        );
    }

    #[test]
    fn memoized_state_reads_enabled_from_typed_integer_tensor_storage() {
        let function = Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
            runmat_value::BoundFunctionHandleData {
                name: "step".into(),
                function: 42,
                authority: runmat_value::BoundFunctionAuthority::unavailable(42),
                source_role: runmat_types::FunctionSourceRole::Unknown,
                source_file: None,
            },
        ));
        let enabled =
            Tensor::new_integer(IntegerStorage::U64(vec![1]), vec![1, 1]).expect("enabled");

        let mut object = runmat_value::UnboundedObjectBuilder::new(MEMOIZED_FUNCTION_CLASS);
        object.insert(FUNCTION_PROPERTY.to_string(), function);
        object.insert(ENABLED_PROPERTY.to_string(), Value::Tensor(enabled));
        let target = runmat_gc::gc_allocate(Value::Object(object.finish())).expect("target");
        let handle = HandleRef {
            class_name: MEMOIZED_FUNCTION_CLASS.into(),
            target,
            valid: true,
        };

        let (_function, enabled, _cache_size) = memoized_state(&handle).expect("state");
        assert!(enabled);
    }

    #[test]
    fn cache_value_equality_preserves_exact_integer_tensor_storage() {
        let left = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
                vec![1, 2],
            )
            .expect("left integer tensor"),
        );
        let same = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
                vec![1, 2],
            )
            .expect("same integer tensor"),
        );
        let rounded_collision = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 53, u64::MAX]), vec![1, 2])
                .expect("different integer tensor"),
        );

        assert!(value_equal_for_cache(&left, &same));
        assert!(!value_equal_for_cache(&left, &rounded_collision));
        assert_ne!(
            value_fingerprint(&left),
            value_fingerprint(&rounded_collision)
        );
    }

    #[test]
    fn cache_value_equality_distinguishes_native_single_from_double() {
        let single = Value::Tensor(
            Tensor::from_f32(vec![1.25, f32::NAN], vec![1, 2]).expect("single tensor"),
        );
        let same_single = Value::Tensor(
            Tensor::from_f32(vec![1.25, f32::NAN], vec![1, 2]).expect("same single tensor"),
        );
        let double =
            Value::Tensor(Tensor::new(vec![1.25, f64::NAN], vec![1, 2]).expect("double tensor"));

        assert!(value_equal_for_cache(&single, &same_single));
        assert!(!value_equal_for_cache(&single, &double));
        assert_ne!(value_fingerprint(&single), value_fingerprint(&double));
    }

    #[test]
    fn cache_value_equality_preserves_exact_sparse_integer_storage() {
        let left = Value::SparseTensor(
            SparseTensor::new_integer(
                2,
                2,
                vec![0, 1, 2],
                vec![0, 1],
                IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
            )
            .expect("left sparse integer tensor"),
        );
        let rounded_collision = Value::SparseTensor(
            SparseTensor::new_integer(
                2,
                2,
                vec![0, 1, 2],
                vec![0, 1],
                IntegerStorage::U64(vec![1_u64 << 53, u64::MAX]),
            )
            .expect("different sparse integer tensor"),
        );

        assert!(!value_equal_for_cache(&left, &rounded_collision));
        assert_ne!(
            value_fingerprint(&left),
            value_fingerprint(&rounded_collision)
        );
    }

    #[test]
    fn cache_value_equality_and_fingerprint_preserve_complex_sparse_components() {
        let sparse = |imaginary| {
            Value::SparseTensor(
                SparseTensor::new_complex(
                    2,
                    2,
                    vec![0, 1, 2],
                    vec![0, 1],
                    vec![(1.0, imaginary), (f64::NAN, 4.0)],
                )
                .expect("complex sparse"),
            )
        };
        let left = sparse(-2.0);
        let same = sparse(-2.0);
        let different = sparse(2.0);
        assert!(value_equal_for_cache(&left, &same));
        assert!(!value_equal_for_cache(&left, &different));
        assert_eq!(value_fingerprint(&left), value_fingerprint(&same));
        assert_ne!(value_fingerprint(&left), value_fingerprint(&different));
    }

    #[test]
    fn cache_value_equality_preserves_exact_complex_integer_storage() {
        let complex = |real, imag| {
            Value::ComplexTensor(
                ComplexTensor::new_integer(
                    IntegerComplexStorage::new(
                        IntegerStorage::U64(real),
                        IntegerStorage::U64(imag),
                    )
                    .expect("matching complex integer storage"),
                    vec![1, 2],
                )
                .expect("complex integer tensor"),
            )
        };
        let left = complex(vec![(1_u64 << 53) + 1, u64::MAX], vec![0, 1_u64 << 63]);
        let rounded_collision = complex(vec![1_u64 << 53, u64::MAX], vec![0, 1_u64 << 63]);

        assert!(!value_equal_for_cache(&left, &rounded_collision));
        assert_ne!(
            value_fingerprint(&left),
            value_fingerprint(&rounded_collision)
        );
    }

    fn set_handle_property(handle: &HandleRef, name: &str, value: Value) {
        runmat_gc::gc_with_value_mut(&handle.target, |target| {
            edit_memoized_object(target, |object| {
                object.insert(name.to_string(), value);
                Ok(())
            })
            .expect("edit memoized target");
        })
        .expect("mutate handle target");
    }

    fn cache_entries(stats: &Value) -> &StructValue {
        let Value::Struct(stats) = stats else {
            panic!("expected stats struct")
        };
        let Some(Value::Struct(cache)) = stats.fields.get("Cache") else {
            panic!("expected Cache struct")
        };
        cache
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn memoize_returns_shared_handle_for_same_function() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();

        let (first, _first_bound) = memoized();
        let (second, _second_bound) = memoized();

        let (Value::HandleObject(first), Value::HandleObject(second)) = (first, second) else {
            panic!("expected handle objects");
        };
        assert_eq!(first.target, second.target);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn repeated_call_hits_cache_for_same_inputs_and_nargout() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        reset_memoize_registry_for_test();
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            counting_invoker(Arc::clone(&counter)),
        ))
        .expect("call-port scope");
        let (f, _bound) = memoized();
        let runtime = exact_memoized_runtime();

        let first =
            call_feval_fixture(&runtime, f.clone(), &[Value::Num(4.0)], 1).expect("first call");
        let second = call_feval_fixture(&runtime, f, &[Value::Num(4.0)], 1).expect("second call");

        assert_eq!(first, Value::Num(5.0));
        assert_eq!(second, Value::Num(5.0));
        assert_eq!(*counter.lock().unwrap(), 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nan_inputs_match_existing_cache_entry() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        reset_memoize_registry_for_test();
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            counting_invoker(Arc::clone(&counter)),
        ))
        .expect("call-port scope");
        let (f, _bound) = memoized();
        let runtime = exact_memoized_runtime();

        let _ = call_feval_fixture(&runtime, f.clone(), &[Value::Num(f64::NAN)], 1)
            .expect("first call");
        let _ = call_feval_fixture(&runtime, f, &[Value::Num(f64::NAN)], 1).expect("second call");

        assert_eq!(*counter.lock().unwrap(), 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn requested_output_count_is_part_of_cache_key() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            counting_invoker(Arc::clone(&counter)),
        ))
        .expect("call-port scope");
        let (f, _bound) = memoized();
        let runtime = exact_memoized_runtime();
        let _caller = crate::output_count::push_output_count(Some(7)).expect("output-count scope");
        for _ in 0..2 {
            for (count, expected) in [
                (1, Value::Num(3.0)),
                (
                    2,
                    Value::output_list(vec![Value::Num(3.0), Value::Num(4.0)]),
                ),
                (0, Value::output_list(Vec::new())),
            ] {
                let output = call_feval_fixture(&runtime, f.clone(), &[Value::Num(2.0)], count)
                    .expect("requested outputs");
                assert_eq!(output, expected, "requested {count} outputs");
                assert_eq!(crate::output_count::current_output_count(), Some(7));
            }
            assert_eq!(*counter.lock().unwrap(), 3);
        }
    }

    #[test]
    fn feval_invalid_count_has_no_effect_and_shortage_is_not_cached_or_replayed() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            counting_invoker(Arc::clone(&counter)),
        ))
        .expect("call-port scope");
        let (f, _bound) = memoized();
        let _caller = crate::output_count::push_output_count(Some(7)).expect("output-count scope");
        let runtime = exact_memoized_runtime();
        let invalid = call_feval_fixture(
            &runtime,
            f.clone(),
            &[Value::Num(2.0)],
            runmat_value::MAX_VALUE_SEQUENCE_OUTPUTS + 1,
        )
        .unwrap_err();
        assert_eq!(
            invalid.identifier(),
            Some("RunMat:OutputSequenceLimitExceeded")
        );
        assert_eq!(*counter.lock().unwrap(), 0);
        assert_eq!(crate::output_count::current_output_count(), Some(7));
        for expected_calls in 1..=2 {
            // Owner-local cache test: the unit binary can contain another
            // runtime's inventory wrappers with independent invoker TLS.
            // Public dispatch is covered by memoize_feval_contract instead.
            let error = {
                let _requested =
                    crate::output_count::push_output_count(Some(3)).expect("output-count scope");
                let Value::HandleObject(handle) = &f else {
                    panic!("memoized handle")
                };
                block_on(call_memoized(
                    Some(&runtime),
                    handle,
                    Value::Cell(CellArray::new(vec![Value::Num(2.0)], 1, 1).unwrap()),
                ))
                .unwrap_err()
            };
            assert_eq!(
                error.identifier(),
                Some("RunMat:CommaSeparatedListOutputShortage")
            );
            assert_eq!(*counter.lock().unwrap(), expected_calls);
            assert_eq!(crate::output_count::current_output_count(), Some(7));
            let Value::HandleObject(handle) = &f else {
                panic!("memoized handle")
            };
            let cached = runmat_gc::gc_with_value(&handle.target, |target| {
                let Value::Object(object) = target else {
                    panic!("memoized target")
                };
                object_cache(object).unwrap().len()
            })
            .unwrap();
            assert_eq!(
                cached, 0,
                "failed output selection must not publish a cache entry"
            );
        }
    }

    #[test]
    fn cancelled_memoized_feval_restores_output_scope_without_publishing() {
        use std::future::Future;
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 42);
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let counter = Arc::new(Mutex::new(0usize));
        let observed = Arc::clone(&counter);
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |_function, _args, requested| {
                let counter = Arc::clone(&observed);
                Box::pin(async move {
                    assert_eq!(requested, 2);
                    *counter.lock().unwrap() += 1;
                    futures::future::pending().await
                })
            },
        )))
        .expect("call-port scope");
        let f = block_on(memoize_builtin(
            None,
            crate::user_functions::test_support::exact_local_value("step", binding),
        ))
        .expect("memoized function");
        let _caller = crate::output_count::push_output_count(Some(7)).expect("output-count scope");
        let args = [Value::Num(2.0)];
        let mut future = Box::pin(
            runtime
                .scope(async { crate::call_feval_async_with_outputs(f.clone(), &args, 2).await }),
        );
        let waker = futures::task::noop_waker();
        let first_poll = future
            .as_mut()
            .poll(&mut std::task::Context::from_waker(&waker));
        assert!(
            first_poll.is_pending(),
            "public feval returned {first_poll:?}"
        );
        assert_eq!(*counter.lock().unwrap(), 1);
        drop(future);
        assert_eq!(crate::output_count::current_output_count(), Some(7));
        let Value::HandleObject(handle) = f else {
            panic!("memoized handle")
        };
        let cached = runmat_gc::gc_with_value(&handle.target, |target| {
            let Value::Object(object) = target else {
                panic!("memoized target")
            };
            object_cache(object).unwrap().len()
        })
        .unwrap();
        assert_eq!(cached, 0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn disabled_memoized_function_bypasses_cache_and_stats() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        reset_memoize_registry_for_test();
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            counting_invoker(Arc::clone(&counter)),
        ))
        .expect("call-port scope");
        let (f, _bound) = memoized();
        let runtime = exact_memoized_runtime();
        let Value::HandleObject(handle) = &f else {
            panic!("expected handle")
        };
        set_handle_property(handle, ENABLED_PROPERTY, Value::Bool(false));

        let _ = call_feval_fixture(&runtime, f.clone(), &[Value::Num(4.0)], 1).expect("first call");
        let _ =
            call_feval_fixture(&runtime, f.clone(), &[Value::Num(4.0)], 1).expect("second call");
        let stats = block_on(stats_builtin(None, f)).expect("stats");
        let cache = cache_entries(&stats);

        assert_eq!(*counter.lock().unwrap(), 2);
        assert_eq!(cache.fields.get("TotalMisses"), Some(&Value::Num(0.0)));
        assert_eq!(cache.fields.get("TotalHits"), Some(&Value::Num(0.0)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cache_size_evicts_oldest_entries() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        reset_memoize_registry_for_test();
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            counting_invoker(Arc::clone(&counter)),
        ))
        .expect("call-port scope");
        let (f, _bound) = memoized();
        let runtime = exact_memoized_runtime();
        let Value::HandleObject(handle) = &f else {
            panic!("expected handle")
        };
        set_handle_property(handle, CACHE_SIZE_PROPERTY, Value::Num(2.0));

        for value in [1.0, 2.0, 3.0] {
            let _ = call_feval_fixture(&runtime, f.clone(), &[Value::Num(value)], 1).expect("call");
        }
        let _ =
            call_feval_fixture(&runtime, f.clone(), &[Value::Num(1.0)], 1).expect("evicted call");
        let stats = block_on(stats_builtin(None, f)).expect("stats");
        let cache = cache_entries(&stats);

        let Some(Value::Cell(inputs)) = cache.fields.get("Inputs") else {
            panic!("expected Inputs cell")
        };
        assert_eq!(inputs.data().len(), 2);
        assert_eq!(*counter.lock().unwrap(), 4);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn clear_cache_and_clear_all_reset_counters() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        reset_memoize_registry_for_test();
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            counting_invoker(Arc::clone(&counter)),
        ))
        .expect("call-port scope");
        let (f, _bound) = memoized();
        let runtime = exact_memoized_runtime();

        {
            let _outputs =
                crate::output_count::push_output_count(Some(1)).expect("output-count scope");
            let err = block_on(clear_cache_builtin(None, f.clone())).unwrap_err();
            assert_eq!(err.identifier(), Some("RunMat:clearCache:TooManyOutputs"));
        }

        let _ = call_feval_fixture(&runtime, f.clone(), &[Value::Num(8.0)], 1).expect("first call");
        let _ =
            call_feval_fixture(&runtime, f.clone(), &[Value::Num(8.0)], 1).expect("second call");
        block_on(clear_cache_builtin(None, f.clone())).expect("clear cache");
        let stats = block_on(stats_builtin(None, f.clone())).expect("stats");
        let cache = cache_entries(&stats);
        assert_eq!(cache.fields.get("TotalHits"), Some(&Value::Num(0.0)));
        assert_eq!(cache.fields.get("TotalMisses"), Some(&Value::Num(0.0)));

        let _ =
            call_feval_fixture(&runtime, f.clone(), &[Value::Num(8.0)], 1).expect("after clear");
        block_on(clear_all_memoized_caches_builtin(None)).expect("clear all");
        let stats = block_on(stats_builtin(None, f)).expect("stats after all");
        let cache = cache_entries(&stats);
        assert_eq!(cache.fields.get("TotalMisses"), Some(&Value::Num(0.0)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_non_function_handle() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();
        let err = block_on(memoize_builtin(None, Value::Num(1.0))).expect_err("expected error");
        assert_eq!(err.identifier(), Some("RunMat:memoize:InvalidFunction"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn invalid_cache_size_errors_before_dispatch() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        reset_memoize_registry_for_test();
        let (f, _bound) = memoized();
        let runtime = exact_memoized_runtime();
        let Value::HandleObject(handle) = &f else {
            panic!("expected handle")
        };
        set_handle_property(handle, CACHE_SIZE_PROPERTY, Value::Num(0.0));

        let err = call_feval_fixture(&runtime, f, &[Value::Num(4.0)], 1)
            .expect_err("expected cache size error");

        assert_eq!(err.identifier(), Some("RunMat:memoize:InvalidCacheSize"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cache_size_rejects_unrepresentable_double_boundary_before_cast() {
        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };

        let err = parse_cache_size(Some(&Value::Num(boundary))).expect_err("boundary rejected");
        assert_eq!(err.identifier(), Some("RunMat:memoize:InvalidCacheSize"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn member_access_returns_properties_and_stats() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();
        let (f, _bound) = memoized();

        let enabled = block_on(memoized_subsref_builtin(
            f.clone(),
            OBJECT_INDEX_MEMBER.to_string(),
            Value::String(ENABLED_PROPERTY.to_string().into()),
        ))
        .expect("enabled");
        let stats = block_on(memoized_subsref_builtin(
            f,
            OBJECT_INDEX_MEMBER.to_string(),
            Value::String("stats".to_string().into()),
        ))
        .expect("stats");

        assert_eq!(enabled.value(), &Value::Bool(true));
        assert!(matches!(stats.value(), Value::Struct(_)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn internal_cache_property_is_not_exposed_as_public_member() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();
        let (f, _bound) = memoized();

        let err = block_on(memoized_subsref_builtin(
            f,
            OBJECT_INDEX_MEMBER.to_string(),
            Value::String(CACHE_PROPERTY.to_string().into()),
        ))
        .expect_err("internal cache property should not be public");

        assert_eq!(err.identifier(), Some("RunMat:memoize:InvalidIndex"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn invalidated_memoized_handle_is_rejected() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        reset_memoize_registry_for_test();
        let (f, _bound) = memoized();
        let Value::HandleObject(handle) = &f else {
            panic!("expected handle")
        };
        assert!(crate::set_handle_valid(handle, false));

        let err = block_on(stats_builtin(None, f)).expect_err("invalid handle should fail");

        assert_eq!(err.identifier(), Some("RunMat:memoize:InvalidObject"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn handle_valued_outputs_are_reported_from_object_backed_cache() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _lock = MEMOIZE_TEST_LOCK.lock().unwrap();
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        reset_memoize_registry_for_test();
        let _guard =
            crate::user_functions::install_semantic_function_invoker(Some(echo_first_invoker()))
                .expect("call-port scope");
        let (f, _bound) = memoized();
        let runtime = exact_memoized_runtime();
        let payload_target = runmat_gc::gc_allocate(Value::Object(ObjectInstance::new(
            "PayloadHandle".to_string(),
        )))
        .expect("payload allocation");
        let payload = Value::HandleObject(HandleRef {
            class_name: "PayloadHandle".into(),
            target: payload_target,
            valid: true,
        });

        let result = call_feval_fixture(&runtime, f.clone(), std::slice::from_ref(&payload), 1)
            .expect("memoized handle output");
        let stats = block_on(stats_builtin(None, f)).expect("stats");
        let cache = cache_entries(&stats);
        let Some(Value::Cell(outputs)) = cache.fields.get("Outputs") else {
            panic!("expected Outputs cell")
        };
        let Some(Value::Cell(first_entry_outputs)) = outputs.data().first() else {
            panic!("expected first cached output cell")
        };

        assert_eq!(result, payload);
        assert_eq!(first_entry_outputs.data().first(), Some(&payload));
    }
}
