//! Context-owned listener registration and callback dispatch.
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::BuiltinErrorDescriptor;
mod admission;
#[cfg(test)]
mod tests;
use admission::{EventAdmission, EventOperation};
use runmat_value::Value;
use std::cell::RefCell;

#[derive(Default)]
pub(crate) struct EventRegistry {
    next_id: u64,
    listeners: std::collections::HashMap<(usize, String), Vec<runmat_value::Listener>>,
    listener_roots: std::collections::HashMap<u64, ListenerRoots>,
}

impl std::fmt::Debug for EventRegistry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EventRegistry")
            .field("next_id", &self.next_id)
            .field("listener_groups", &self.listeners.len())
            .field("listener_roots", &self.listener_roots.len())
            .finish()
    }
}

struct ListenerRoots {
    _target: runmat_gc::ExplicitRoot,
    _callback: runmat_gc::ExplicitRoot,
}

thread_local! {
    static EVENT_REGISTRY: RefCell<EventRegistry> = RefCell::new(EventRegistry::default());
}

fn with_event_registry<R>(operation: impl FnOnce(&EventRegistry) -> R) -> R {
    if let Some(context) = crate::context::legacy::active() {
        return operation(&context.state().events.borrow());
    }
    EVENT_REGISTRY.with(|registry| operation(&registry.borrow()))
}

fn with_event_registry_mut<R>(operation: impl FnOnce(&mut EventRegistry) -> R) -> R {
    if let Some(context) = crate::context::legacy::active() {
        return operation(&mut context.state().events.borrow_mut());
    }
    EVENT_REGISTRY.with(|registry| operation(&mut registry.borrow_mut()))
}

#[cfg(test)]
pub(super) fn reset_event_registry_for_test() {
    with_event_registry_mut(|registry| {
        *registry = EventRegistry::default();
    });
}

pub(crate) fn invalidate_listener_registration(listener_id: u64) {
    with_event_registry_mut(|registry| {
        for listeners in registry.listeners.values_mut() {
            for listener in listeners.iter_mut() {
                if listener.id == listener_id {
                    listener.valid = false;
                    listener.enabled = false;
                }
            }
        }
        registry.listener_roots.remove(&listener_id);
    });
}

fn canonicalize_listener_callback(callback: Value) -> Result<Value, RuntimeError> {
    crate::canonicalize_callback_handle_for_semantic_resolution(callback)
}

fn listener_callback_is_dispatchable(callback: &Value) -> bool {
    match callback {
        Value::String(text) => text.as_str().is_some_and(|text| !text.trim().is_empty()),
        Value::StringArray(array) => {
            array.data().len() == 1
                && array.data()[0]
                    .as_str()
                    .is_some_and(|text| !text.trim().is_empty())
        }
        Value::CharArray(chars) if chars.rows == 1 => chars
            .data()
            .iter()
            .any(|character| !character.is_whitespace()),
        Value::FunctionHandle(name)
        | Value::ExternalFunctionHandle(name)
        | Value::MethodFunctionHandle(name) => !name.trim().is_empty(),
        Value::BoundFunctionHandle(_) | Value::Closure(_) => true,
        _ => false,
    }
}

fn event_descriptor_error(
    builtin: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(builtin)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn event_descriptor_error_with_source(
    builtin: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(builtin)
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn event_data_error(error: crate::class_registry::EventDataError) -> RuntimeError {
    use crate::class_registry::EventDataError;
    let (descriptor, message) = match &error {
        EventDataError::InvalidSource => (
            &runmat_builtins::NOTIFY_ERROR_TARGET,
            "notify: event source is not a valid handle object",
        ),
        EventDataError::InvalidPayload | EventDataError::InvalidTarget => (
            &runmat_builtins::NOTIFY_ERROR_DATA,
            "notify: data must be a valid event.EventData object",
        ),
        EventDataError::SourceRoot(_)
        | EventDataError::DataRoot(_)
        | EventDataError::Allocation(_)
        | EventDataError::Binding(_) => (
            &runmat_builtins::NOTIFY_ERROR_ARGUMENT_LIFETIME,
            "notify: failed to retain event data",
        ),
    };
    event_descriptor_error_with_source("notify", descriptor, message, error)
}

pub(crate) async fn addlistener_builtin(
    target: Value,
    event_name: String,
    callback: Value,
) -> crate::BuiltinResult<Value> {
    let key_ptr: usize = match &target {
        Value::HandleObject(h) => {
            if !crate::is_handle_valid(h) {
                return Err(event_descriptor_error(
                    "addlistener",
                    &runmat_builtins::ADDLISTENER_ERROR_TARGET,
                    "addlistener: target handle is invalid",
                ));
            }
            runmat_gc::gc_handle_addr(&h.target)
        }
        Value::Object(_) => {
            return Err(event_descriptor_error(
                "addlistener",
                &runmat_builtins::ADDLISTENER_ERROR_TARGET,
                "addlistener: target object must be a handle object",
            ))
        }
        _ => {
            return Err(event_descriptor_error(
                "addlistener",
                &runmat_builtins::ADDLISTENER_ERROR_TARGET,
                "addlistener: target must be handle or object",
            ))
        }
    };
    let Value::HandleObject(target_handle) = &target else {
        unreachable!("validated listener target is a handle object")
    };
    let admission = EventAdmission::for_target(target_handle, &event_name, EventOperation::Listen)?;
    let (target_root, target_class_name) = match &target {
        Value::HandleObject(h) => {
            let class_name = h.class_name.clone();
            (
                runmat_gc::gc_root(h.target).map_err(|source| {
                    event_descriptor_error_with_source(
                        "addlistener",
                        &runmat_builtins::ADDLISTENER_ERROR_ALLOCATION,
                        "addlistener: failed to retain target",
                        source,
                    )
                })?,
                class_name,
            )
        }
        _ => unreachable!(),
    };
    if !listener_callback_is_dispatchable(&callback) {
        return Err(event_descriptor_error(
            "addlistener",
            &runmat_builtins::ADDLISTENER_ERROR_CALLBACK,
            "addlistener: callback must be callable or scalar text",
        ));
    }
    // Callback canonicalization may consult semantic function state or
    // allocate. Root a closure's captured handles before that work, as well
    // as the receiver above, and retain both until the registry owns them.
    let callback_input_root = runmat_gc::gc_allocate_rooted(callback).map_err(|source| {
        event_descriptor_error_with_source(
            "addlistener",
            &runmat_builtins::ADDLISTENER_ERROR_ALLOCATION,
            "addlistener: failed to retain callback input",
            source,
        )
    })?;
    let callback_value =
        runmat_gc::gc_clone_value(&callback_input_root.handle()).map_err(|source| {
            event_descriptor_error_with_source(
                "addlistener",
                &runmat_builtins::ADDLISTENER_ERROR_ALLOCATION,
                "addlistener: failed to read retained callback input",
                source,
            )
        })?;
    let callback = canonicalize_listener_callback(callback_value)?;
    let callback_root = runmat_gc::gc_allocate_rooted(callback).map_err(|source| {
        event_descriptor_error_with_source(
            "addlistener",
            &runmat_builtins::ADDLISTENER_ERROR_ALLOCATION,
            "addlistener: failed to retain callback",
            source,
        )
    })?;
    admission.recheck(target_handle)?;
    let id = with_event_registry_mut(|registry| {
        registry.next_id += 1;
        registry.next_id
    });
    let listener = runmat_value::Listener {
        id,
        target: target_root.handle(),
        target_class_name,
        event_name: event_name.clone().into(),
        callback: callback_root.handle(),
        enabled: true,
        valid: true,
    };
    with_event_registry_mut(|registry| {
        registry
            .listeners
            .entry((key_ptr, event_name))
            .or_default()
            .push(listener.clone());
        registry.listener_roots.insert(
            id,
            ListenerRoots {
                _target: target_root,
                _callback: callback_root,
            },
        );
    });
    Ok(Value::Listener(listener))
}

pub(crate) async fn notify_builtin(
    runtime: Option<&crate::context::RuntimeContext>,
    target: Value,
    event_name: String,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let notification = notify_inner(runtime, target, event_name, rest);
    match runtime {
        Some(runtime) => runtime.scope(notification).await,
        None => notification.await,
    }
}

async fn notify_inner(
    runtime: Option<&crate::context::RuntimeContext>,
    target: Value,
    event_name: String,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let key_ptr: usize = match &target {
        Value::HandleObject(h) => {
            if !crate::is_handle_valid(h) {
                return Err(event_descriptor_error(
                    "notify",
                    &runmat_builtins::NOTIFY_ERROR_TARGET,
                    "notify: target handle is invalid",
                ));
            }
            runmat_gc::gc_handle_addr(&h.target)
        }
        Value::Object(_) => {
            return Err(event_descriptor_error(
                "notify",
                &runmat_builtins::NOTIFY_ERROR_TARGET,
                "notify: target object must be a handle object",
            ))
        }
        _ => {
            return Err(event_descriptor_error(
                "notify",
                &runmat_builtins::NOTIFY_ERROR_TARGET,
                "notify: target must be handle or object",
            ))
        }
    };
    let Value::HandleObject(target_handle) = &target else {
        unreachable!("validated notification target is a handle object")
    };
    let admission = EventAdmission::for_target(target_handle, &event_name, EventOperation::Notify)?;
    let prepared = if rest.len() <= 1 {
        Some(
            crate::class_registry::prepare_event_data(&target, &event_name, rest.first().cloned())
                .map_err(event_data_error)?,
        )
    } else {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::NOTIFY_EXTRA_ARGS_EXTENSION,
            "notify",
        )?;
        None
    };
    let mut to_call: Vec<runmat_value::Listener> = Vec::new();
    with_event_registry(|registry| {
        if let Some(list) = registry.listeners.get(&(key_ptr, event_name.clone())) {
            for l in list {
                if l.valid && l.enabled {
                    to_call.push(l.clone());
                }
            }
        }
    });
    // A callback can invalidate a later listener and trigger collection.
    // Retain the complete snapshot through every callback await/reentry.
    let Value::HandleObject(target_handle) = &target else {
        unreachable!()
    };
    let _target_root = runmat_gc::gc_root(target_handle.target).map_err(|source| {
        event_descriptor_error_with_source(
            "notify",
            &runmat_builtins::NOTIFY_ERROR_TARGET,
            "notify: failed to retain event target",
            source,
        )
    })?;
    let _callback_roots = to_call
        .iter()
        .map(|listener| {
            runmat_gc::gc_root(listener.callback).map_err(|source| {
                event_descriptor_error_with_source(
                    "notify",
                    &runmat_builtins::NOTIFY_ERROR_CALLBACK,
                    "notify: failed to retain registered callback",
                    source,
                )
            })
        })
        .collect::<crate::BuiltinResult<Vec<_>>>()?;
    // The original argument forest survives every listener. Default EventData
    // is the only newly constructed supplied slot; ordinary extra arguments
    // retain their original logical segments, including captured prefixes.
    let supplied = if rest.is_empty() {
        let data = prepared
            .as_ref()
            .expect("empty notification prepares EventData");
        crate::call::BuiltinArguments::from_borrowed(&[data.value().clone()], runtime)?
    } else {
        rest
    };
    let arguments = crate::call::BuiltinArguments::new(
        supplied
            .into_function_arguments()
            .prepend_routing(target.clone(), runtime)?,
    );
    for l in to_call {
        admission.recheck(target_handle)?;
        // Call callback via feval-like protocol.
        let cbv: Value = runmat_gc::gc_clone_value(&l.callback).map_err(|source| {
            event_descriptor_error_with_source(
                "notify",
                &runmat_builtins::NOTIFY_ERROR_CALLBACK,
                "notify: invalid listener callback handle",
                source,
            )
        })?;
        if !listener_callback_is_dispatchable(&cbv) {
            return Err(event_descriptor_error(
                "notify",
                &runmat_builtins::NOTIFY_ERROR_CALLBACK,
                "notify: registered callback is not dispatchable",
            ));
        }
        let _ = crate::call::feval::invoke_retained(runtime, cbv, arguments.fork(), 0, None)
            .await
            .map_err(|mut failure| {
                // A prior listener may already have executed. Preserve the
                // callback failure itself, but never authorize replay of the
                // enclosing notification.
                failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
                failure
            })?;
    }
    Ok(Value::Num(0.0))
}
