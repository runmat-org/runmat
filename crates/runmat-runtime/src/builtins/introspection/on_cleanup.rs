//! MATLAB-compatible `onCleanup` handle object.
pub(crate) mod lifecycle;
use runmat_types::MemberAccess;

use std::collections::HashMap;

#[cfg(test)]
use once_cell::sync::Lazy;
#[cfg(test)]
use std::sync::Mutex;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{HandleRef, ObjectInstance, Value};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(crate) const ON_CLEANUP_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("onCleanup");
const CALLBACK_PROPERTY: &str = "__oncleanup_callback";
const ACTIVE_PROPERTY: &str = "__oncleanup_active";
const ON_CLEANUP_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "onCleanup accepts only a function handle or closure and returns a handle object; numeric values are rejected as callbacks.",
};
const CANCEL_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "cancel accepts lifecycle and execution handles and returns a fixed double status; it has no integer data, control, or class-preserving form.",
};

#[cfg(test)]
pub(crate) static ON_CLEANUP_TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

const ON_CLEANUP_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "cleanupObj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle object that invokes the cleanup function when deleted or cleared.",
}];
const ON_CLEANUP_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "cleanupFun",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zero-input function handle to invoke during cleanup.",
}];
const ON_CLEANUP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "cleanupObj = onCleanup(cleanupFun)",
    inputs: &ON_CLEANUP_INPUTS,
    outputs: &ON_CLEANUP_OUTPUTS,
}];
const ON_CLEANUP_ERROR_INVALID_CALLBACK: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ONCLEANUP.INVALID_CALLBACK",
    identifier: Some("RunMat:onCleanup:InvalidCallback"),
    when: "The cleanup function is not a function handle or closure.",
    message: "onCleanup: cleanupFun must be a function handle",
};
const ON_CLEANUP_ERROR_INVALID_OBJECT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ONCLEANUP.INVALID_OBJECT",
    identifier: Some("RunMat:onCleanup:InvalidObject"),
    when: "A cleanup operation targets a value that is not an onCleanup object.",
    message: "onCleanup: invalid cleanup object",
};
const ON_CLEANUP_ERROR_GC: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ONCLEANUP.GC",
    identifier: Some("RunMat:onCleanup:GcFailure"),
    when: "The cleanup object target cannot be allocated, rooted, or mutated.",
    message: "onCleanup: internal object storage failed",
};
const ON_CLEANUP_ERROR_HOST: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ONCLEANUP.LIFECYCLE_HOST_UNAVAILABLE",
    identifier: Some("RunMat:onCleanup:LifecycleHostUnavailable"),
    when: "No explicit session lifecycle host owns cleanup callback dispatch.",
    message: "onCleanup: lifecycle host is unavailable",
};
const CANCEL_ERROR_RUNTIME_CONTEXT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CANCEL.RUNTIME_CONTEXT",
    identifier: Some("RunMat:parallel:RuntimeContextUnavailable"),
    when: "A task, future, or job is cancelled without an active runtime context.",
    message: "cancel: no active runtime context",
};
const CANCEL_ERROR_EXECUTION_SERVICE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CANCEL.EXECUTION_SERVICE",
    identifier: Some("RunMat:cancel:ExecutionService"),
    when: "The execution service rejects cancellation of a task, future, or job.",
    message: "cancel: execution service failed",
};
const ON_CLEANUP_ERRORS: [BuiltinErrorDescriptor; 4] = [
    ON_CLEANUP_ERROR_INVALID_CALLBACK,
    ON_CLEANUP_ERROR_INVALID_OBJECT,
    ON_CLEANUP_ERROR_GC,
    ON_CLEANUP_ERROR_HOST,
];
const CANCEL_ERRORS: [BuiltinErrorDescriptor; 6] = [
    ON_CLEANUP_ERROR_INVALID_CALLBACK,
    ON_CLEANUP_ERROR_INVALID_OBJECT,
    ON_CLEANUP_ERROR_GC,
    ON_CLEANUP_ERROR_HOST,
    CANCEL_ERROR_RUNTIME_CONTEXT,
    CANCEL_ERROR_EXECUTION_SERVICE,
];
pub const ON_CLEANUP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ON_CLEANUP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ON_CLEANUP_ERRORS,
};

const DELETE_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zero after the cleanup callback has been invoked or skipped.",
}];
const DELETE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "cleanupObj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "onCleanup object to execute and deactivate.",
}];
const CANCEL_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "cleanupObj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "onCleanup object to deactivate without running its callback.",
}];
const DELETE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "status = delete(cleanupObj)",
    inputs: &DELETE_INPUTS,
    outputs: &DELETE_OUTPUTS,
}];
pub const ON_CLEANUP_DELETE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DELETE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &ON_CLEANUP_ERRORS,
};

const CANCEL_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zero after the cleanup callback has been deactivated.",
}];
const CANCEL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "status = cancel(cleanupObj)",
    inputs: &CANCEL_INPUTS,
    outputs: &CANCEL_OUTPUTS,
}];
pub const ON_CLEANUP_CANCEL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CANCEL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CANCEL_ERRORS,
};

#[runtime_builtin(
    name = "onCleanup",
    category = "introspection",
    summary = "Create an object that runs a function when deleted or cleared.",
    keywords = "onCleanup,cleanup,delete,clear,resource,RAII",
    descriptor(crate::builtins::introspection::on_cleanup::ON_CLEANUP_DESCRIPTOR),
    integer_audit(crate::builtins::introspection::on_cleanup::ON_CLEANUP_INTEGER_AUDIT),
    builtin_path = "crate::builtins::introspection::on_cleanup"
)]
pub(crate) async fn on_cleanup_builtin(callback: Value) -> BuiltinResult<Value> {
    let callback = canonicalize_cleanup_callback(callback)?;
    let registry = crate::context::legacy::active()
        .and_then(|runtime| runtime.cleanup_registry())
        .ok_or_else(|| on_cleanup_error(&ON_CLEANUP_ERROR_HOST, ON_CLEANUP_ERROR_HOST.message))?;
    ensure_on_cleanup_class_registered();
    let mut object = ObjectInstance::new(ON_CLEANUP_CLASS.to_string());
    object
        .properties
        .insert(CALLBACK_PROPERTY.to_string(), callback.clone());
    object
        .properties
        .insert(ACTIVE_PROPERTY.to_string(), Value::Bool(true));
    object.properties.insert(
        crate::HANDLE_VALID_FLAG_PROPERTY.to_string(),
        Value::Bool(true),
    );

    let target_root = runmat_gc::gc_allocate_rooted(Value::Object(object))
        .map_err(|err| on_cleanup_error(&ON_CLEANUP_ERROR_GC, format!("onCleanup: {err}")))?;
    let target = target_root.handle();
    registry
        .register(target, callback)
        .map_err(|err| on_cleanup_error(&ON_CLEANUP_ERROR_GC, format!("onCleanup: {err}")))?;

    target_root
        .unroot()
        .map_err(|err| on_cleanup_error(&ON_CLEANUP_ERROR_GC, format!("onCleanup: {err}")))?;

    Ok(Value::HandleObject(HandleRef {
        class_name: ON_CLEANUP_CLASS.into(),
        target,
        valid: true,
    }))
}

#[runtime_builtin(
    name = "__runmat_oncleanup_delete",
    execution_stack = "process",
    category = "introspection",
    summary = "Execute an onCleanup object callback once.",
    keywords = "onCleanup,delete,cleanup",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::introspection::on_cleanup::ON_CLEANUP_DELETE_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::on_cleanup"
)]
pub(crate) async fn on_cleanup_delete_builtin(value: Value) -> BuiltinResult<Value> {
    let cleanup_result = run_on_cleanup_value(&value).await;
    let invalidation_result = match &value {
        Value::HandleObject(handle) if is_on_cleanup_handle(handle) => {
            if crate::set_handle_valid(handle, false) {
                Ok(())
            } else {
                Err(on_cleanup_error(
                    &ON_CLEANUP_ERROR_INVALID_OBJECT,
                    "onCleanup: failed to invalidate cleanup object",
                ))
            }
        }
        _ => Ok(()),
    };
    cleanup_result?;
    invalidation_result?;
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "cancel",
    category = "introspection",
    summary = "Deactivate an onCleanup object without running its callback.",
    keywords = "onCleanup,cancel,cleanup",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::introspection::on_cleanup::ON_CLEANUP_CANCEL_DESCRIPTOR),
    integer_audit(crate::builtins::introspection::on_cleanup::CANCEL_INTEGER_AUDIT),
    builtin_path = "crate::builtins::introspection::on_cleanup"
)]
async fn on_cleanup_cancel_builtin(value: Value) -> BuiltinResult<Value> {
    if let Some(tasks) = crate::parallel::future::output_tasks(&value) {
        let context = crate::context::legacy::active().ok_or_else(|| {
            cancel_error(
                &CANCEL_ERROR_RUNTIME_CONTEXT,
                CANCEL_ERROR_RUNTIME_CONTEXT.message,
            )
        })?;
        for task in tasks {
            context
                .execution()
                .cancel(
                    &Value::Task(task),
                    runmat_execution::CancellationReason::User,
                )
                .map_err(|error| {
                    cancel_error(&CANCEL_ERROR_EXECUTION_SERVICE, format!("cancel: {error}"))
                })?;
        }
        return Ok(Value::Num(0.0));
    }
    let execution_value = crate::parallel::future::execution_value(&value).unwrap_or(&value);
    match execution_value {
        Value::Future(_) | Value::Task(_) | Value::Job(_) => {
            let context = crate::context::legacy::active().ok_or_else(|| {
                cancel_error(
                    &CANCEL_ERROR_RUNTIME_CONTEXT,
                    CANCEL_ERROR_RUNTIME_CONTEXT.message,
                )
            })?;
            context
                .execution()
                .cancel(execution_value, runmat_execution::CancellationReason::User)
                .map_err(|error| {
                    cancel_error(&CANCEL_ERROR_EXECUTION_SERVICE, format!("cancel: {error}"))
                })?;
        }
        _ => cancel_on_cleanup_value(&value)?,
    }
    Ok(Value::Num(0.0))
}

pub(crate) async fn run_cleanup_after_detach(
    detached: &crate::workspace::DetachedWorkspaceValues,
) -> BuiltinResult<()> {
    let Some(runtime) = crate::context::legacy::active() else {
        return Ok(());
    };
    let Some(registry) = runtime.cleanup_registry() else {
        return Ok(());
    };
    lifecycle::dispatch::drain_eligible(&runtime, &registry, &[detached.root_registration()]).await
}

#[cfg(test)]
pub(crate) async fn run_cleanup_for_workspace_value(value: &Value) -> BuiltinResult<()> {
    let detached =
        crate::workspace::DetachedWorkspaceValues::new(vec![("fixture".into(), value.clone())])
            .map_err(crate::workspace::base_workspace_error_to_runtime)?;
    run_cleanup_after_detach(&detached).await
}

fn ensure_on_cleanup_class_registered() {
    if crate::class_registry::get_class(&ON_CLEANUP_CLASS.owned()).is_some() {
        return;
    }
    let mut methods = HashMap::new();
    methods.insert(
        "delete".into(),
        crate::class_registry::RuntimeMethod {
            name: "delete".into(),
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: "__runmat_oncleanup_delete".into(),
            implicit_class_argument: None,
        },
    );
    methods.insert(
        "cancel".into(),
        crate::class_registry::RuntimeMethod {
            name: "cancel".into(),
            is_static: false,
            is_abstract: false,
            is_sealed: false,
            access: MemberAccess::Public,
            function_name: "cancel".into(),
            implicit_class_argument: None,
        },
    );
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: ON_CLEANUP_CLASS.into(),
        parent: Some(runmat_types::standard::HANDLE.into()),
        properties:
            HashMap::<runmat_types::MemberName, crate::class_registry::RuntimeProperty>::new(),
        methods,
    });
}

fn canonicalize_cleanup_callback(callback: Value) -> BuiltinResult<Value> {
    match callback {
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => Ok(crate::canonicalize_callback_handle_for_semantic_resolution(
            callback,
        )),
        _ => Err(on_cleanup_error(
            &ON_CLEANUP_ERROR_INVALID_CALLBACK,
            ON_CLEANUP_ERROR_INVALID_CALLBACK.message,
        )),
    }
}

async fn run_on_cleanup_value(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::HandleObject(handle) if is_on_cleanup_handle(handle) => {
            run_on_cleanup_handle(handle).await
        }
        other => Err(on_cleanup_error(
            &ON_CLEANUP_ERROR_INVALID_OBJECT,
            format!("onCleanup: expected onCleanup object, got {other:?}"),
        )),
    }
}

async fn run_on_cleanup_handle(handle: &HandleRef) -> BuiltinResult<()> {
    if !is_on_cleanup_handle(handle) || !crate::is_handle_valid(handle) {
        return Ok(());
    }
    let runtime = crate::context::legacy::active()
        .ok_or_else(|| on_cleanup_error(&ON_CLEANUP_ERROR_HOST, ON_CLEANUP_ERROR_HOST.message))?;
    let registry = runtime
        .cleanup_registry()
        .ok_or_else(|| on_cleanup_error(&ON_CLEANUP_ERROR_HOST, ON_CLEANUP_ERROR_HOST.message))?;
    let entry = registry.find(handle.target).ok_or_else(|| {
        on_cleanup_error(
            &ON_CLEANUP_ERROR_INVALID_OBJECT,
            "onCleanup: object is not owned by this session",
        )
    })?;
    if entry.phase() == lifecycle::registry::CleanupPhase::Active {
        deactivate_handle(handle)?;
        if registry.queue_explicit(&entry) {
            lifecycle::dispatch::start_queued(&runtime, &registry, entry.clone())?;
        }
    }
    if matches!(
        entry.phase(),
        lifecycle::registry::CleanupPhase::Queued | lifecycle::registry::CleanupPhase::Running
    ) {
        lifecycle::dispatch::wait_queued(&registry, &entry).await
    } else {
        Ok(())
    }
}

fn cancel_on_cleanup_value(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::HandleObject(handle) if is_on_cleanup_handle(handle) => {
            if !crate::is_handle_valid(handle) {
                return Ok(());
            }
            let runtime = crate::context::legacy::active().ok_or_else(|| {
                on_cleanup_error(&ON_CLEANUP_ERROR_HOST, ON_CLEANUP_ERROR_HOST.message)
            })?;
            let registry = runtime.cleanup_registry().ok_or_else(|| {
                on_cleanup_error(&ON_CLEANUP_ERROR_HOST, ON_CLEANUP_ERROR_HOST.message)
            })?;
            let entry = registry.find(handle.target).ok_or_else(|| {
                on_cleanup_error(
                    &ON_CLEANUP_ERROR_INVALID_OBJECT,
                    "onCleanup: object is not owned by this session",
                )
            })?;
            deactivate_handle(handle)?;
            registry.cancel(&entry);
            Ok(())
        }
        other => Err(on_cleanup_error(
            &ON_CLEANUP_ERROR_INVALID_OBJECT,
            format!("onCleanup: expected onCleanup object, got {other:?}"),
        )),
    }
}

fn deactivate_handle(handle: &HandleRef) -> BuiltinResult<()> {
    if !is_on_cleanup_handle(handle) || !crate::is_handle_valid(handle) {
        return Ok(());
    }

    runmat_gc::gc_with_value_mut(&handle.target, |target| {
        let Value::Object(object) = target else {
            return;
        };
        let active = matches!(
            object.properties.get(ACTIVE_PROPERTY),
            Some(Value::Bool(true))
        );
        if !active {
            return;
        }
        object
            .properties
            .insert(ACTIVE_PROPERTY.to_string(), Value::Bool(false));
    })
    .map_err(|err| on_cleanup_error(&ON_CLEANUP_ERROR_GC, format!("onCleanup: {err}")))?;
    Ok(())
}

fn is_on_cleanup_handle(handle: &HandleRef) -> bool {
    handle.class_name == ON_CLEANUP_CLASS
}

fn on_cleanup_error(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("onCleanup");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn cancel_error(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("cancel");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    #[cfg(not(target_arch = "wasm32"))]
    use runmat_value::CellArray;
    #[cfg(not(target_arch = "wasm32"))]
    use std::future::Future;
    #[cfg(not(target_arch = "wasm32"))]
    use std::rc::Rc;
    use std::sync::Arc;

    #[cfg(not(target_arch = "wasm32"))]
    fn hosted_fixture_runtime() -> (
        crate::context::RuntimeContext,
        Rc<crate::context::RuntimeLocalTaskSupervisor>,
        crate::context::RuntimeLifecycleHost,
    ) {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let supervisor = Rc::new(crate::context::RuntimeLocalTaskSupervisor::new());
        let ports = runtime
            .service_ports()
            .clone()
            .with_local_tasks(supervisor.submission_port(runtime.local_identity()));
        let runtime = runtime.with_service_ports(ports);
        let host = crate::context::RuntimeLifecycleHost::new(&runtime, supervisor.clone())
            .expect("exact lifecycle host");
        (runtime, supervisor, host)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn drive<F: Future>(
        supervisor: &Rc<crate::context::RuntimeLocalTaskSupervisor>,
        future: F,
    ) -> F::Output {
        block_on(supervisor.drive_until(future)).expect("local task driver")
    }

    fn callback_invoker(counter: Arc<Mutex<usize>>) -> Arc<crate::user_functions::FunctionInvoker> {
        Arc::new(move |_function, args, requested_outputs| {
            assert!(args.is_empty(), "cleanup callbacks receive no arguments");
            assert_eq!(requested_outputs, 0);
            let counter = Arc::clone(&counter);
            crate::sequence::single_value_future(async move {
                *counter.lock().unwrap() += 1;
                Ok(Value::Tensor(runmat_value::Tensor::zeros(vec![0, 0])))
            })
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn on_cleanup_accepts_function_handles_and_runs_once_on_delete() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        let (runtime, supervisor, _host) = hosted_fixture_runtime();
        let _entry = runtime.enter();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 7);
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            callback_invoker(Arc::clone(&counter)),
        ));
        let cleanup = block_on(on_cleanup_builtin(
            crate::user_functions::test_support::exact_local_value("cleanup", binding),
        ))
        .expect("create cleanup");

        drive(&supervisor, run_on_cleanup_value(&cleanup)).expect("first cleanup");
        drive(&supervisor, run_on_cleanup_value(&cleanup)).expect("second cleanup is no-op");

        assert_eq!(*counter.lock().unwrap(), 1);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn delete_method_invalidates_cleanup_handle() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        let (runtime, supervisor, _host) = hosted_fixture_runtime();
        let _entry = runtime.enter();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 7);
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            callback_invoker(Arc::clone(&counter)),
        ));
        let cleanup = block_on(on_cleanup_builtin(
            crate::user_functions::test_support::exact_local_value("cleanup", binding),
        ))
        .expect("create cleanup");
        let Value::HandleObject(handle) = cleanup.clone() else {
            panic!("expected cleanup handle");
        };

        drive(&supervisor, on_cleanup_delete_builtin(cleanup)).expect("delete cleanup");

        assert_eq!(*counter.lock().unwrap(), 1);
        assert!(
            !crate::is_handle_valid(&handle),
            "delete method should invalidate shared handle target"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn cancel_prevents_later_cleanup() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        let (runtime, supervisor, _host) = hosted_fixture_runtime();
        let _entry = runtime.enter();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 7);
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            callback_invoker(Arc::clone(&counter)),
        ));
        let cleanup = block_on(on_cleanup_builtin(
            crate::user_functions::test_support::exact_local_value("cleanup", binding),
        ))
        .expect("create cleanup");

        cancel_on_cleanup_value(&cleanup).expect("cancel cleanup");
        drive(&supervisor, run_on_cleanup_value(&cleanup)).expect("cleanup remains no-op");

        assert_eq!(*counter.lock().unwrap(), 0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_numeric_and_non_function_handle_callbacks() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        for callback in [Value::Num(1.0), Value::Int(runmat_value::IntValue::I64(1))] {
            let err = block_on(on_cleanup_builtin(callback)).expect_err("expected error");
            assert_eq!(err.identifier(), Some("RunMat:onCleanup:InvalidCallback"));
        }
        let err = block_on(on_cleanup_cancel_builtin(Value::Int(
            runmat_value::IntValue::U64(1),
        )))
        .expect_err("integer is not an onCleanup handle");
        assert_eq!(err.identifier(), Some("RunMat:onCleanup:InvalidObject"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn valid_callback_in_bare_context_fails_before_object_admission() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let callback =
            crate::user_functions::test_support::exact_local_handle(&runtime, "cleanup", 7);
        let error = block_on(on_cleanup_builtin(callback)).expect_err("host is required");
        assert_eq!(
            error.identifier(),
            Some("RunMat:onCleanup:LifecycleHostUnavailable")
        );
        assert!(runtime.cleanup_registry().is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn workspace_cleanup_walks_cells() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        let (runtime, supervisor, _host) = hosted_fixture_runtime();
        let _entry = runtime.enter();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 7);
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            callback_invoker(Arc::clone(&counter)),
        ));
        let cleanup = block_on(on_cleanup_builtin(
            crate::user_functions::test_support::exact_local_value("cleanup", binding),
        ))
        .expect("create cleanup");
        let cell = Value::Cell(CellArray::new(vec![cleanup], 1, 1).expect("cell"));

        drive(&supervisor, run_cleanup_for_workspace_value(&cell)).expect("cleanup");

        assert_eq!(*counter.lock().unwrap(), 1);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn workspace_cleanup_walks_structs_objects_and_output_lists() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        let (runtime, supervisor, _host) = hosted_fixture_runtime();
        let _entry = runtime.enter();
        let a = crate::user_functions::test_support::exact_local_binding(&runtime, 7);
        let b = crate::user_functions::test_support::exact_local_binding(&runtime, 8);
        let c = crate::user_functions::test_support::exact_local_binding(&runtime, 9);
        let _bound = crate::user_functions::test_support::install_bound_forwarders(vec![
            a.clone(),
            b.clone(),
            c.clone(),
        ]);
        let counter = Arc::new(Mutex::new(0usize));
        let _guard = crate::user_functions::install_semantic_function_invoker(Some(
            callback_invoker(Arc::clone(&counter)),
        ));
        let cleanup_a = block_on(on_cleanup_builtin(
            crate::user_functions::test_support::exact_local_value("cleanupA", a),
        ))
        .expect("create cleanup a");
        let cleanup_b = block_on(on_cleanup_builtin(
            crate::user_functions::test_support::exact_local_value("cleanupB", b),
        ))
        .expect("create cleanup b");
        let cleanup_c = block_on(on_cleanup_builtin(
            crate::user_functions::test_support::exact_local_value("cleanupC", c),
        ))
        .expect("create cleanup c");

        let mut struct_value = runmat_value::StructValue::new();
        struct_value.insert("cleanup", cleanup_a);
        let mut object = ObjectInstance::new("CleanupHolder".to_string());
        object.properties.insert("cleanup".to_string(), cleanup_b);
        let composite = Value::OutputList(vec![
            Value::Struct(struct_value),
            Value::Object(object),
            cleanup_c,
        ]);

        drive(&supervisor, run_cleanup_for_workspace_value(&composite)).expect("cleanup");

        assert_eq!(*counter.lock().unwrap(), 3);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn last_returned_root_retirement_wakes_idle_host_once() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        let (runtime, supervisor, _host) = hosted_fixture_runtime();
        let _entry = runtime.enter();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 7);
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let counter = Arc::new(Mutex::new(0usize));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
            callback_invoker(Arc::clone(&counter)),
        ));
        let cleanup = block_on(on_cleanup_builtin(
            crate::user_functions::test_support::exact_local_value("cleanup", binding),
        ))
        .expect("create cleanup");
        let returned = crate::execution::RootedValueSequence::new(
            runmat_value::ValueSequence::single(cleanup).expect("single cleanup"),
        )
        .expect("root caller-held result");
        drive(&supervisor, async { tokio::task::yield_now().await });
        assert_eq!(*counter.lock().unwrap(), 0, "caller still owns result");

        drop(returned);
        drive(&supervisor, async {
            for _ in 0..16 {
                if *counter.lock().unwrap() == 1 {
                    return;
                }
                tokio::task::yield_now().await;
            }
            panic!("idle root retirement did not drive cleanup");
        });
        assert_eq!(*counter.lock().unwrap(), 1, "cleanup must not replay");
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test(async)]
    async fn browser_hosted_delete_drains_callback_without_inline_fallback() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let supervisor = std::rc::Rc::new(crate::context::BrowserLocalTaskSupervisor::new());
        let ports = runtime
            .service_ports()
            .clone()
            .with_local_tasks(supervisor.submission_port(runtime.local_identity()));
        let runtime = runtime.with_service_ports(ports);
        let _host = crate::context::RuntimeLifecycleHost::new(&runtime, supervisor)
            .expect("browser lifecycle host");
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 7);
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let counter = Arc::new(Mutex::new(0usize));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
            callback_invoker(Arc::clone(&counter)),
        ));
        runtime
            .clone()
            .scope(async {
                let cleanup = on_cleanup_builtin(
                    crate::user_functions::test_support::exact_local_value("cleanup", binding),
                )
                .await
                .expect("create browser cleanup");
                on_cleanup_delete_builtin(cleanup)
                    .await
                    .expect("browser delete");
            })
            .await;
        assert_eq!(*counter.lock().unwrap(), 1);
    }
}
