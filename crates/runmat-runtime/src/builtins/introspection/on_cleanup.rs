//! MATLAB-compatible `onCleanup` handle object.
pub(crate) mod lifecycle;
use crate::context::RuntimeContext;
use crate::execution::RootedValue;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
#[cfg(test)]
use once_cell::sync::Lazy;
use runmat_builtins::{
    BuiltinErrorDescriptor, CANCEL_DESCRIPTOR, CANCEL_ERROR_EXECUTION_SERVICE,
    CANCEL_ERROR_OUTPUTS, CANCEL_ERROR_RUNTIME_CONTEXT, ON_CLEANUP_DELETE_DESCRIPTOR,
    ON_CLEANUP_DELETE_ERROR_OUTPUTS, ON_CLEANUP_ERROR_GC, ON_CLEANUP_ERROR_HOST,
    ON_CLEANUP_ERROR_INVALID_CALLBACK, ON_CLEANUP_ERROR_INVALID_OBJECT,
};
use runmat_macros::runtime_builtin;
use runmat_types::MemberAccess;
#[cfg(test)]
use runmat_value::ObjectInstance;
use runmat_value::{HandleRef, Value};
use std::collections::HashMap;
#[cfg(test)]
use std::sync::Mutex;

pub(crate) const ON_CLEANUP_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("onCleanup");
const CALLBACK_PROPERTY: &str = "__oncleanup_callback";
const ACTIVE_PROPERTY: &str = "__oncleanup_active";
#[cfg(test)]
pub(crate) static ON_CLEANUP_TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

fn operation_error(mut error: RuntimeError) -> RuntimeError {
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

#[runtime_builtin(
    name = "onCleanup",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::on_cleanup"
)]
pub(crate) async fn on_cleanup_builtin(callback: Value) -> BuiltinResult<RootedValue> {
    let callback = canonicalize_cleanup_callback(callback)?;
    let registry = crate::context::legacy::active()
        .and_then(|runtime| runtime.cleanup_registry())
        .ok_or_else(|| on_cleanup_error(&ON_CLEANUP_ERROR_HOST, ON_CLEANUP_ERROR_HOST.message))?;
    ensure_on_cleanup_class_registered();
    let mut object = runmat_value::UnboundedObjectBuilder::new(ON_CLEANUP_CLASS);
    object.insert(CALLBACK_PROPERTY.to_string(), callback.clone());
    object.insert(ACTIVE_PROPERTY.to_string(), Value::Bool(true));
    object.insert(
        crate::HANDLE_VALID_FLAG_PROPERTY.to_string(),
        Value::Bool(true),
    );

    let target_root = runmat_gc::gc_allocate_rooted(Value::Object(object.finish()))
        .map_err(|err| on_cleanup_error(&ON_CLEANUP_ERROR_GC, format!("onCleanup: {err}")))?;
    let target = target_root.handle();
    // This producer owns the returned lease before any root retirement can
    // synchronously wake the lifecycle host. The registry's target pin is not
    // an outside owner and cannot by itself make Active ineligible to queue.
    let output = RootedValue::new(Value::HandleObject(HandleRef {
        class_name: ON_CLEANUP_CLASS.into(),
        target,
        valid: true,
    }))
    .map_err(|err| on_cleanup_error(&ON_CLEANUP_ERROR_GC, format!("onCleanup: {err}")))?;
    registry
        .register(target, callback)
        .map_err(|err| on_cleanup_error(&ON_CLEANUP_ERROR_GC, format!("onCleanup: {err}")))?;

    target_root
        .unroot()
        .map_err(|err| on_cleanup_error(&ON_CLEANUP_ERROR_GC, format!("onCleanup: {err}")))?;

    Ok(output)
}

#[runtime_builtin(
    name = "__runmat_oncleanup_delete",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::on_cleanup"
)]
pub(crate) async fn on_cleanup_delete_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &ON_CLEANUP_DELETE_DESCRIPTOR,
        &ON_CLEANUP_DELETE_ERROR_OUTPUTS,
        "__runmat_oncleanup_delete",
    )
    .map_err(operation_error)?;
    // A selected invocation may not invalidate a foreign session's handle.
    // Once ownership is established, dispatch failure still invalidates this
    // same handle, preserving the existing delete lifecycle semantics.
    let invalidate_owned = match &value {
        Value::HandleObject(handle) if is_on_cleanup_handle(handle) => runtime
            .as_ref()
            .and_then(RuntimeContext::cleanup_registry)
            .is_some_and(|registry| registry.find(handle.target).is_some()),
        _ => false,
    };
    let cleanup_result = run_on_cleanup_value(runtime.as_ref(), &value).await;
    let invalidation_result = match &value {
        Value::HandleObject(handle) if invalidate_owned => {
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
    cleanup_result.map_err(operation_error)?;
    invalidation_result.map_err(operation_error)?;
    Ok(Value::Num(0.0))
}

#[runtime_builtin(
    name = "cancel",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::on_cleanup"
)]
async fn on_cleanup_cancel_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &CANCEL_DESCRIPTOR,
        &CANCEL_ERROR_OUTPUTS,
        "cancel",
    )
    .map_err(operation_error)?;
    cancel_target(runtime.as_ref(), &value).map_err(operation_error)?;
    Ok(Value::Num(0.0))
}

fn cancel_target(runtime: Option<&RuntimeContext>, value: &Value) -> BuiltinResult<()> {
    if let Some(tasks) = crate::parallel::future::output_task_values(value) {
        let context = runtime.ok_or_else(|| {
            cancel_error(
                &CANCEL_ERROR_RUNTIME_CONTEXT,
                CANCEL_ERROR_RUNTIME_CONTEXT.message,
            )
        })?;
        for task in tasks {
            context
                .execution()
                .cancel(task, runmat_execution::CancellationReason::User)
                .map_err(|error| {
                    cancel_error(&CANCEL_ERROR_EXECUTION_SERVICE, format!("cancel: {error}"))
                })?;
        }
        return Ok(());
    }
    let execution_value = crate::parallel::future::execution_value(value).unwrap_or(value);
    match execution_value {
        Value::Future(_) | Value::Task(_) | Value::Job(_) => {
            let context = runtime.ok_or_else(|| {
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
        _ => cancel_on_cleanup_value(runtime, value)?,
    }
    Ok(())
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
    let completion = registry.issue_completion()?;
    lifecycle::dispatch::schedule_before_retirement(&runtime, &registry, &[])?;
    lifecycle::dispatch::schedule_completion(
        &runtime,
        &registry,
        &[detached.root_registration()],
        completion,
    )?;
    lifecycle::dispatch::drain_completion(&runtime, &registry, completion).await
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
            role: runmat_types::ClassMethodRole::Ordinary,
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
            role: runmat_types::ClassMethodRole::Ordinary,
        },
    );
    crate::class_registry::register_builtin_class(
        ON_CLEANUP_CLASS,
        crate::class_registry::RuntimeClass {
            events: Default::default(),
            name: ON_CLEANUP_CLASS.into(),
            parent: Some(runmat_types::standard::HANDLE.into()),
            properties:
                HashMap::<runmat_types::MemberName, crate::class_registry::RuntimeProperty>::new(),
            methods,
        },
    );
}

fn canonicalize_cleanup_callback(callback: Value) -> BuiltinResult<Value> {
    match callback {
        Value::ScopedFunctionHandle(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_)
        | Value::Closure(_) => {
            crate::canonicalize_callback_handle_for_semantic_resolution(callback)
        }
        _ => Err(on_cleanup_error(
            &ON_CLEANUP_ERROR_INVALID_CALLBACK,
            ON_CLEANUP_ERROR_INVALID_CALLBACK.message,
        )),
    }
}

async fn run_on_cleanup_value(
    runtime: Option<&RuntimeContext>,
    value: &Value,
) -> BuiltinResult<()> {
    match value {
        Value::HandleObject(handle) if is_on_cleanup_handle(handle) => {
            run_on_cleanup_handle(runtime, handle).await
        }
        other => Err(on_cleanup_error(
            &ON_CLEANUP_ERROR_INVALID_OBJECT,
            format!("onCleanup: expected onCleanup object, got {other:?}"),
        )),
    }
}

async fn run_on_cleanup_handle(
    runtime: Option<&RuntimeContext>,
    handle: &HandleRef,
) -> BuiltinResult<()> {
    if !is_on_cleanup_handle(handle) || !crate::is_handle_valid(handle) {
        return Ok(());
    }
    let runtime = runtime
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
        let completion = registry.issue_completion()?;
        lifecycle::dispatch::schedule_before_retirement(runtime, &registry, &[])?;
        deactivate_handle(handle)?;
        if registry.queue_explicit_for_completion(
            &entry,
            runtime.command_profiling(),
            Some(completion),
        ) {
            let _ = lifecycle::dispatch::start_queued(runtime, &registry, entry.clone());
            // A service may synchronously poll or destroy the submitted task
            // before returning its lease; report that exact terminal receipt.
            return lifecycle::dispatch::drain_completion(runtime, &registry, completion).await;
        }
    }
    // This destructor has already been admitted. Repeated delete is a no-op,
    // including self-delete and two callbacks deleting each other. Only the
    // invocation that admitted it owns its completion wait.
    Ok(())
}

fn cancel_on_cleanup_value(runtime: Option<&RuntimeContext>, value: &Value) -> BuiltinResult<()> {
    match value {
        Value::HandleObject(handle) if is_on_cleanup_handle(handle) => {
            if !crate::is_handle_valid(handle) {
                return Ok(());
            }
            let runtime = runtime.ok_or_else(|| {
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
        let active = matches!(object.property(ACTIVE_PROPERTY), Some(Value::Bool(true)));
        if !active {
            return;
        }
        let previous = std::mem::replace(target, Value::Num(0.0));
        let Value::Object(object) = previous else {
            unreachable!("cleanup target was checked as an object")
        };
        let mut object = match object.into_unbounded_builder() {
            Ok(object) => object,
            Err(object) => {
                *target = Value::Object(object);
                return;
            }
        };
        object.insert(ACTIVE_PROPERTY.to_string(), Value::Bool(false));
        *target = Value::Object(object.finish());
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
    use runmat_execution::profiling::{
        current_command_profiling, CommandProfilingRoot, ProviderObservationTerminal,
    };
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

    #[cfg(not(target_arch = "wasm32"))]
    #[derive(Clone, Copy)]
    enum SynchronousSubmitRetirement {
        Accept,
        Reject,
        PollReadyThenReject,
    }

    #[cfg(not(target_arch = "wasm32"))]
    struct SynchronousRetirementPort {
        delegate: Rc<dyn crate::context::RuntimeLocalTaskService>,
        behavior: SynchronousSubmitRetirement,
    }

    #[cfg(not(target_arch = "wasm32"))]
    impl crate::context::RuntimeLocalTaskService for SynchronousRetirementPort {
        fn owner_token(&self) -> Option<crate::context::LocalTaskOwnerToken> {
            self.delegate.owner_token()
        }

        fn spawn_local(
            &self,
            owner: crate::context::RuntimeContextLocalId,
            future: crate::context::RuntimeLocalTaskFuture,
        ) -> Result<crate::context::RuntimeLocalTaskLease, crate::RuntimeError> {
            match self.behavior {
                SynchronousSubmitRetirement::Accept => {
                    drop(future);
                    self.delegate.spawn_local(owner, Box::pin(async {}))
                }
                SynchronousSubmitRetirement::Reject => {
                    drop(future);
                    Err(
                        crate::build_runtime_error("exact fixture rejected cleanup submission")
                            .with_identifier("RunMat:onCleanup:ExactRejectedPort")
                            .build(),
                    )
                }
                SynchronousSubmitRetirement::PollReadyThenReject => {
                    let mut future = future;
                    let mut context =
                        std::task::Context::from_waker(futures::task::noop_waker_ref());
                    assert!(
                        matches!(
                            future.as_mut().poll(&mut context),
                            std::task::Poll::Ready(())
                        ),
                        "the fixture callback must finish in one poll before rejection"
                    );
                    Err(
                        crate::build_runtime_error("exact fixture rejected after polling callback")
                            .with_identifier("RunMat:onCleanup:PolledThenRejected")
                            .build(),
                    )
                }
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn synchronous_retirement_runtime(
        behavior: SynchronousSubmitRetirement,
    ) -> (
        crate::context::RuntimeContext,
        Rc<crate::context::RuntimeLocalTaskSupervisor>,
        crate::context::RuntimeLifecycleHost,
    ) {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let supervisor = Rc::new(crate::context::RuntimeLocalTaskSupervisor::new());
        let delegate = supervisor.submission_port(runtime.local_identity());
        let ports = runtime
            .service_ports()
            .clone()
            .with_local_tasks(delegate.clone());
        let runtime = runtime.with_service_ports(ports);
        // The lifecycle dispatcher is permanent host work. Admit it through
        // the ordinary supervisor before substituting the callback-only test
        // port on this same session context.
        let host = crate::context::RuntimeLifecycleHost::new(&runtime, supervisor.clone())
            .expect("exact lifecycle host");
        let ports = runtime
            .service_ports()
            .clone()
            .with_local_tasks(Rc::new(SynchronousRetirementPort { delegate, behavior }));
        let runtime = runtime.with_service_ports(ports);
        (runtime, supervisor, host)
    }

    fn callback_invoker(
        counter: Arc<Mutex<usize>>,
    ) -> Arc<crate::user_functions::UnscopedFunctionInvoker> {
        Arc::new(move |_function, args, requested_outputs| {
            assert!(args.is_empty(), "cleanup callbacks receive no arguments");
            assert_eq!(requested_outputs, 0);
            let counter = Arc::clone(&counter);
            Box::pin(async move {
                *counter.lock().unwrap() += 1;
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Tensor(
                    runmat_value::Tensor::zeros(vec![0, 0]),
                ))
            })
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn profiled_callback_invoker(
        observed: Arc<Mutex<Vec<bool>>>,
    ) -> Arc<crate::user_functions::UnscopedFunctionInvoker> {
        Arc::new(move |_function, args, requested_outputs| {
            assert!(args.is_empty());
            assert_eq!(requested_outputs, 0);
            let observed = Arc::clone(&observed);
            Box::pin(async move {
                let admitted = current_command_profiling()
                    .and_then(|command| command.begin_provider_operation());
                observed.lock().unwrap().push(admitted.is_some());
                if let Some(admitted) = admitted {
                    admitted.complete(ProviderObservationTerminal::Missing);
                }
                crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(0.0))
            })
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn cleanup_create_a_explicit_delete_b_uses_only_queue_admission_command() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        let (runtime, supervisor, _host) = hosted_fixture_runtime();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 7);
        let registration = runtime.enter();
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let observed = Arc::new(Mutex::new(Vec::new()));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
            profiled_callback_invoker(Arc::clone(&observed)),
        ))
        .expect("call-port scope");
        drop(registration);
        let create = CommandProfilingRoot::new();
        let created_context = runtime.clone().with_command_profiling(create.receipt());
        let cleanup = {
            let _selected = created_context.enter();
            block_on(on_cleanup_builtin(
                crate::user_functions::test_support::exact_local_value("cleanup", binding),
            ))
            .expect("create under command A")
        };
        let delete = CommandProfilingRoot::new();
        let delete_context = runtime.clone().with_command_profiling(delete.receipt());
        {
            let _selected = delete_context.enter();
            drive(
                &supervisor,
                run_on_cleanup_value(Some(&delete_context), cleanup.value()),
            )
            .expect("delete under B");
        }
        assert_eq!(*observed.lock().unwrap(), [true]);
        let created = create.seal();
        let deleted = delete.seal();
        assert_eq!(created.observations.expect("A valid").missing, 0);
        assert_eq!(created.deferred_work.expect("A valid deferred").admitted, 0);
        assert_eq!(deleted.observations.expect("B valid").missing, 1);
        let work = deleted.deferred_work.expect("B valid deferred");
        assert_eq!((work.admitted, work.pending, work.completed), (1, 0, 1));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn queued_b_drained_c_keeps_b_not_the_drain_command() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        let (runtime, supervisor, _host) = hosted_fixture_runtime();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 7);
        let registration = runtime.enter();
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let observed = Arc::new(Mutex::new(Vec::new()));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
            profiled_callback_invoker(Arc::clone(&observed)),
        ))
        .expect("call-port scope");
        drop(registration);
        let cleanup = {
            let _selected = runtime.enter();
            block_on(on_cleanup_builtin(
                crate::user_functions::test_support::exact_local_value("cleanup", binding),
            ))
            .expect("create cleanup")
        };
        let Value::HandleObject(handle) = cleanup.value() else {
            panic!("expected cleanup handle");
        };
        let registry = runtime.cleanup_registry().expect("host registry");
        let entry = registry.find(handle.target).expect("registered cleanup");
        let b = CommandProfilingRoot::new();
        assert!(registry.queue_explicit(&entry, Some(&b.receipt())));
        let c = CommandProfilingRoot::new();
        let c_runtime = runtime.clone().with_command_profiling(c.receipt());
        lifecycle::dispatch::start_queued(&c_runtime, &registry, entry.clone())
            .expect("dispatch queued callback");
        drive(
            &supervisor,
            lifecycle::dispatch::drain_eligible(&c_runtime, &registry, &[]),
        )
        .expect("drain under C");
        assert_eq!(*observed.lock().unwrap(), [true]);
        let queued = b.seal();
        let drained = c.seal();
        assert_eq!(queued.observations.expect("B valid").missing, 1);
        assert_eq!(drained.observations.expect("C valid").missing, 0);
        let work = queued.deferred_work.expect("B valid deferred");
        assert_eq!((work.admitted, work.pending, work.completed), (1, 0, 1));
        assert_eq!(drained.deferred_work.expect("C valid deferred").admitted, 0);
        drop(cleanup);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn synchronous_accepted_future_drop_reports_abandonment_after_owned_retirement() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        let (runtime, supervisor, _host) =
            synchronous_retirement_runtime(SynchronousSubmitRetirement::Accept);
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 7);
        let registration = runtime.enter();
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        drop(registration);
        let cleanup = {
            let _selected = runtime.enter();
            block_on(on_cleanup_builtin(
                crate::user_functions::test_support::exact_local_value("cleanup", binding),
            ))
            .expect("register cleanup")
        };
        let command = CommandProfilingRoot::new();
        let selected = runtime.clone().with_command_profiling(command.receipt());
        let error = {
            let _selected = selected.enter();
            drive(
                &supervisor,
                run_on_cleanup_value(Some(&selected), cleanup.value()),
            )
            .expect_err("synchronously dropped callback is abandoned")
        };
        assert_eq!(error.identifier(), Some("RunMat:onCleanup:Abandoned"));
        let work = command.seal().deferred_work.expect("valid deferred work");
        assert_eq!((work.admitted, work.pending, work.missing), (1, 0, 1));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn synchronous_rejection_preserves_original_error_and_failed_attribution() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        let (runtime, supervisor, _host) =
            synchronous_retirement_runtime(SynchronousSubmitRetirement::Reject);
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 7);
        let registration = runtime.enter();
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        drop(registration);
        let cleanup = {
            let _selected = runtime.enter();
            block_on(on_cleanup_builtin(
                crate::user_functions::test_support::exact_local_value("cleanup", binding),
            ))
            .expect("register cleanup")
        };
        let command = CommandProfilingRoot::new();
        let selected = runtime.clone().with_command_profiling(command.receipt());
        let error = {
            let _selected = selected.enter();
            drive(
                &supervisor,
                on_cleanup_delete_builtin(Some(selected.clone()), cleanup.value().clone()),
            )
            .expect_err("exact rejection must retain original diagnostic")
        };
        assert_eq!(
            error.identifier(),
            Some("RunMat:onCleanup:ExactRejectedPort")
        );
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        let Value::HandleObject(handle) = cleanup.value() else {
            unreachable!()
        };
        assert!(
            !crate::is_handle_valid(handle),
            "failed delete still invalidates its existing handle"
        );
        let work = command.seal().deferred_work.expect("valid deferred work");
        assert_eq!((work.admitted, work.pending, work.failed), (1, 0, 1));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn synchronous_ready_then_rejected_port_reports_error_without_callback_replay() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        let (runtime, supervisor, _host) =
            synchronous_retirement_runtime(SynchronousSubmitRetirement::PollReadyThenReject);
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 7);
        let registration = runtime.enter();
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let counter = Arc::new(Mutex::new(0usize));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
            callback_invoker(Arc::clone(&counter)),
        ))
        .expect("call-port scope");
        drop(registration);
        let cleanup = {
            let _selected = runtime.enter();
            block_on(on_cleanup_builtin(
                crate::user_functions::test_support::exact_local_value("cleanup", binding),
            ))
            .expect("register cleanup")
        };
        let command = CommandProfilingRoot::new();
        let selected = runtime.clone().with_command_profiling(command.receipt());
        let error = {
            let _selected = selected.enter();
            drive(
                &supervisor,
                run_on_cleanup_value(Some(&selected), cleanup.value()),
            )
            .expect_err("late port rejection remains the caller's exact error")
        };
        assert_eq!(
            error.identifier(),
            Some("RunMat:onCleanup:PolledThenRejected")
        );
        assert_eq!(*counter.lock().unwrap(), 1);
        {
            let _selected = selected.enter();
            drive(
                &supervisor,
                run_on_cleanup_value(Some(&selected), cleanup.value()),
            )
            .expect("terminal cleanup object cannot replay callback");
        }
        assert_eq!(*counter.lock().unwrap(), 1);
        let work = command.seal().deferred_work.expect("valid deferred work");
        assert_eq!((work.admitted, work.pending, work.completed), (1, 0, 1));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn same_poll_task_cancellation_wins_over_ready_callback_without_early_root_retirement() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        let (runtime, supervisor, _host) = hosted_fixture_runtime();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 7);
        let registration = runtime.enter();
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        drop(registration);
        let cleanup = {
            let _selected = runtime.enter();
            block_on(on_cleanup_builtin(
                crate::user_functions::test_support::exact_local_value("cleanup", binding),
            ))
            .expect("register cleanup")
        };
        let Value::HandleObject(handle) = cleanup.value() else {
            panic!("expected cleanup handle");
        };
        let registry = runtime.cleanup_registry().expect("host registry");
        let entry = registry.find(handle.target).expect("registered cleanup");
        let counter = Arc::new(Mutex::new(0usize));
        let registration = runtime.enter();
        let _invoker = crate::user_functions::install_local_semantic_function_invoker(Rc::new({
            let entry = Rc::clone(&entry);
            let counter = Arc::clone(&counter);
            move |_runtime, _function, args, requested_outputs| {
                assert!(args.is_empty());
                assert_eq!(requested_outputs, 0);
                let entry = Rc::clone(&entry);
                let counter = Arc::clone(&counter);
                crate::user_functions::admit_user_function_with(
                    _runtime.as_ref(),
                    move || async move {
                        entry
                            .lease()
                            .expect("task lease attached before poll")
                            .cancel();
                        *counter.lock().unwrap() += 1;
                        crate::user_functions::test_support::rooted_single_value_sequence(
                            Value::Num(0.0),
                        )
                    },
                )
            }
        }))
        .expect("call-port scope");
        drop(registration);
        let command = CommandProfilingRoot::new();
        let selected = runtime.clone().with_command_profiling(command.receipt());
        let error = {
            let _selected = selected.enter();
            drive(
                &supervisor,
                run_on_cleanup_value(Some(&selected), cleanup.value()),
            )
            .expect_err("same-poll cancellation must not complete successfully")
        };
        assert_eq!(error.identifier(), Some("RunMat:onCleanup:Abandoned"));
        assert_eq!(*counter.lock().unwrap(), 1);
        assert_eq!(entry.phase(), lifecycle::registry::CleanupPhase::Abandoned);
        assert!(!entry.test_has_callback_roots_and_deferred_admission());
        let work = command.seal().deferred_work.expect("valid deferred work");
        assert_eq!((work.admitted, work.pending, work.missing), (1, 0, 1));
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
        ))
        .expect("call-port scope");
        let cleanup = block_on(on_cleanup_builtin(
            crate::user_functions::test_support::exact_local_value("cleanup", binding),
        ))
        .expect("create cleanup");

        drive(
            &supervisor,
            run_on_cleanup_value(Some(&runtime), cleanup.value()),
        )
        .expect("first cleanup");
        drive(
            &supervisor,
            run_on_cleanup_value(Some(&runtime), cleanup.value()),
        )
        .expect("second cleanup is no-op");

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
        ))
        .expect("call-port scope");
        let cleanup = block_on(on_cleanup_builtin(
            crate::user_functions::test_support::exact_local_value("cleanup", binding),
        ))
        .expect("create cleanup");
        let Value::HandleObject(handle) = cleanup.value().clone() else {
            panic!("expected cleanup handle");
        };

        let error = drive(
            &supervisor,
            crate::dispatcher::call_builtin_async_with_outputs_in_context(
                &runtime,
                "__runmat_oncleanup_delete",
                &[cleanup.value().clone()],
                2,
            ),
        )
        .expect_err("output preflight must precede callback admission and invalidation");
        assert_eq!(error.identifier(), Some("RunMat:onCleanup:TooManyOutputs"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(crate::is_handle_valid(&handle));
        assert_eq!(*counter.lock().unwrap(), 0);
        let foreign = RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let error = block_on(on_cleanup_delete_builtin(
            Some(foreign),
            cleanup.value().clone(),
        ))
        .expect_err("selected foreign session must not use ambient lifecycle ownership");
        assert_eq!(
            error.identifier(),
            Some("RunMat:onCleanup:LifecycleHostUnavailable")
        );
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(
            crate::is_handle_valid(&handle),
            "foreign invocation must not invalidate an unowned target"
        );
        assert_eq!(*counter.lock().unwrap(), 0);
        let status = drive(
            &supervisor,
            crate::dispatcher::call_builtin_async_with_outputs_in_context(
                &runtime,
                "__runmat_oncleanup_delete",
                &[cleanup.value().clone()],
                1,
            ),
        )
        .expect("delete cleanup");
        assert_eq!(status, Value::Num(0.));

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
        ))
        .expect("call-port scope");
        let cleanup = block_on(on_cleanup_builtin(
            crate::user_functions::test_support::exact_local_value("cleanup", binding),
        ))
        .expect("create cleanup");

        let error = block_on(
            crate::dispatcher::call_builtin_async_with_outputs_in_context(
                &runtime,
                "cancel",
                &[cleanup.value().clone()],
                2,
            ),
        )
        .expect_err("output preflight must not deactivate cleanup");
        assert_eq!(error.identifier(), Some("RunMat:cancel:TooManyOutputs"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        let Value::HandleObject(handle) = cleanup.value() else {
            unreachable!()
        };
        let entry = runtime
            .cleanup_registry()
            .unwrap()
            .find(handle.target)
            .unwrap();
        assert_eq!(entry.phase(), lifecycle::registry::CleanupPhase::Active);
        let status = block_on(
            crate::dispatcher::call_builtin_async_with_outputs_in_context(
                &runtime,
                "cancel",
                &[cleanup.value().clone()],
                1,
            ),
        )
        .expect("cancel cleanup");
        assert_eq!(status, Value::Num(0.));
        drive(
            &supervisor,
            run_on_cleanup_value(Some(&runtime), cleanup.value()),
        )
        .expect("cleanup remains no-op");

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
        let err = block_on(on_cleanup_cancel_builtin(
            None,
            Value::Int(runmat_value::IntValue::U64(1)),
        ))
        .expect_err("integer is not an onCleanup handle");
        assert_eq!(err.identifier(), Some("RunMat:onCleanup:InvalidObject"));
        assert_eq!(err.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }

    fn cancellation_future(runtime: &RuntimeContext) -> runmat_execution::FutureHandle {
        runtime
            .execution()
            .create_future(crate::execution::DeferredCall {
                spawn_safety: runmat_types::SpawnSafetyFact::RequiresIsolation,
                invocation: crate::execution::DeferredInvocation::callable(
                    crate::call::descriptor::CallableDescriptor::resolved(
                        runmat_types::CallableIdentity::Builtin(runmat_types::BuiltinId(
                            "zeros".into(),
                        )),
                        Vec::new(),
                        1,
                        runmat_types::CallableFallbackPolicy::None,
                        crate::call::descriptor::CallableCallKind::Direct,
                    ),
                ),
                retry: runmat_execution::RetryPolicy::Never,
                program_revision: None,
                environment: runmat_execution::EnvironmentSnapshot::default(),
                console: None,
                capabilities: Default::default(),
                program: crate::execution::DeferredProgram::Local,
            })
            .expect("existing execution owner creates the actual future")
    }
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn canonical_cancel_uses_selected_execution_owner_and_preflights_outputs() {
        let runtime = RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let _process =
            runtime.enter_execution_stack(crate::context::RuntimeExecutionStack::Process);
        let ambient = RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let _ambient = ambient.enter();
        let future = cancellation_future(&runtime);
        let target = Value::Future(future.clone().into());
        let error = block_on(
            crate::dispatcher::call_builtin_async_with_outputs_in_context(
                &runtime,
                "cancel",
                std::slice::from_ref(&target),
                2,
            ),
        )
        .expect_err("no cancellation before output preflight");
        assert_eq!(error.identifier(), Some("RunMat:cancel:TooManyOutputs"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(
            runtime.execution().inspect_future(&future).unwrap().state,
            runmat_execution::ExecutionHandleState::Deferred
        );
        let status = block_on(
            crate::dispatcher::call_builtin_async_with_outputs_in_context(
                &runtime,
                "cancel",
                std::slice::from_ref(&target),
                1,
            ),
        )
        .expect("selected context, not unrelated ambient owner");
        assert_eq!(status, Value::Num(0.));
        assert_eq!(
            runtime.execution().inspect_future(&future).unwrap().state,
            runmat_execution::ExecutionHandleState::Cancelled
        );
        let embedding_status = block_on(
            crate::dispatcher::call_builtin_async_with_outputs_in_context(
                &runtime,
                "cancel",
                std::slice::from_ref(&target),
                0,
            ),
        )
        .unwrap();
        // The raw builtin embedding ABI preserves its fixed status value;
        // zero-output discard belongs to the language callable boundary.
        assert_eq!(embedding_status, Value::Num(0.));
        let language_future = cancellation_future(&runtime);
        let descriptor = crate::call::descriptor::CallableDescriptor::resolved(
            runmat_types::CallableIdentity::Builtin(runmat_types::BuiltinId("cancel".into())),
            vec![Value::Future(language_future.clone().into())],
            0,
            runmat_types::CallableFallbackPolicy::None,
            crate::call::descriptor::CallableCallKind::Direct,
        );
        let discarded = block_on(
            crate::call::descriptor::execute_callable_descriptor_in_context(&runtime, descriptor),
        )
        .expect("language cancellation with zero requested outputs");
        assert!(discarded.value().is_empty());
        assert_eq!(
            runtime
                .execution()
                .inspect_future(&language_future)
                .unwrap()
                .state,
            runmat_execution::ExecutionHandleState::Cancelled
        );
        let future = cancellation_future(&runtime);
        let task = runtime.execution().spawn(&future).unwrap();
        let status = block_on(on_cleanup_cancel_builtin(
            Some(runtime.clone()),
            Value::Task(task.clone().into()),
        ))
        .unwrap();
        assert_eq!(status, Value::Num(0.));
        assert_eq!(
            runtime.execution().inspect_task(&task).unwrap().state,
            runmat_execution::ExecutionHandleState::Cancelled
        );
        let future = cancellation_future(&runtime);
        let error = block_on(on_cleanup_cancel_builtin(
            None,
            Value::Future(future.clone().into()),
        ))
        .expect_err("absence of selected owner never falls back to ambient");
        assert_eq!(
            error.identifier(),
            Some("RunMat:parallel:RuntimeContextUnavailable")
        );
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(
            runtime.execution().inspect_future(&future).unwrap().state,
            runmat_execution::ExecutionHandleState::Deferred
        );
        let error = block_on(on_cleanup_cancel_builtin(
            Some(ambient.clone()),
            Value::Future(future.into()),
        ))
        .expect_err("scope mismatch reaches exact selected service");
        assert_eq!(error.identifier(), Some("RunMat:cancel:ExecutionService"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn canonical_cancel_borrows_validated_future_arrays_and_on_all_aggregates() {
        let runtime = RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let _process =
            runtime.enter_execution_stack(crate::context::RuntimeExecutionStack::Process);
        for on_all in [false, true] {
            let first = cancellation_future(&runtime);
            let second = cancellation_future(&runtime);
            let tasks = vec![
                runtime.execution().spawn(&first).unwrap(),
                runtime.execution().spawn(&second).unwrap(),
            ];
            let targets = if on_all {
                crate::parallel::future::wrap_on_all_tasks(tasks.clone()).unwrap()
            } else {
                crate::parallel::future::wrap_tasks(tasks.clone()).unwrap()
            };
            assert_eq!(
                crate::parallel::future::output_task_values(&targets)
                    .unwrap()
                    .count(),
                2
            );
            let status = block_on(
                crate::dispatcher::call_builtin_async_with_outputs_in_context(
                    &runtime,
                    "cancel",
                    &[targets],
                    1,
                ),
            )
            .unwrap();
            assert_eq!(status, Value::Num(0.));
            for task in tasks {
                assert_eq!(
                    runtime.execution().inspect_task(&task).unwrap().state,
                    runmat_execution::ExecutionHandleState::Cancelled
                );
            }
        }
        let future = cancellation_future(&runtime);
        let task = runtime.execution().spawn(&future).unwrap();
        let malformed =
            runmat_value::UnboundedObjectBuilder::new(crate::parallel::future::FEVAL_FUTURE_CLASS)
                .finish();
        let targets = Value::ObjectArray(
            runmat_value::ObjectArray::new(
                crate::parallel::future::FEVAL_FUTURE_CLASS,
                vec![
                    crate::parallel::future::wrap_task(task.clone()),
                    Value::Object(malformed),
                ],
                vec![1, 2],
            )
            .unwrap(),
        );
        assert!(crate::parallel::future::output_task_values(&targets).is_none());
        let error = block_on(
            crate::dispatcher::call_builtin_async_with_outputs_in_context(
                &runtime,
                "cancel",
                &[targets],
                1,
            ),
        )
        .expect_err("malformed second element must precede all task effects");
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_ne!(
            runtime.execution().inspect_task(&task).unwrap().state,
            runmat_execution::ExecutionHandleState::Cancelled
        );
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
        ))
        .expect("call-port scope");
        let cleanup = block_on(on_cleanup_builtin(
            crate::user_functions::test_support::exact_local_value("cleanup", binding),
        ))
        .expect("create cleanup");
        let cell = cleanup
            .try_map_owned(
                |value| {
                    Ok::<_, RuntimeError>(Value::Cell(
                        CellArray::new(vec![value], 1, 1).expect("cell"),
                    ))
                },
                |error| error.into_runtime_error(),
            )
            .expect("root cell before producer lease retires");

        drive(&supervisor, run_cleanup_for_workspace_value(cell.value()))
            .expect("retained cell is not yet eligible");
        assert_eq!(*counter.lock().unwrap(), 0);
        drop(cell);
        drive(&supervisor, async {
            for _ in 0..16 {
                if *counter.lock().unwrap() == 1 {
                    return;
                }
                tokio::task::yield_now().await;
            }
            panic!("nested cell release did not drive cleanup");
        });
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
        ))
        .expect("call-port scope");
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

        let mut struct_value = runmat_value::UnboundedStructValueBuilder::new();
        struct_value.insert("cleanup", cleanup_a.value().clone());
        let object = ObjectInstance::new("CleanupHolder".to_string());
        let mut object = object
            .into_unbounded_builder()
            .expect("ordinary test object");
        object.insert("cleanup".to_string(), cleanup_b.value().clone());
        let object = object.finish();
        let composite = RootedValue::new(Value::output_list(vec![
            Value::Struct(struct_value.finish()),
            Value::Object(object),
            cleanup_c.value().clone(),
        ]))
        .expect("root composite before producer leases retire");
        drop((cleanup_a, cleanup_b, cleanup_c));

        drive(
            &supervisor,
            run_cleanup_for_workspace_value(composite.value()),
        )
        .expect("retained composite is not yet eligible");
        assert_eq!(*counter.lock().unwrap(), 0);
        drop(composite);
        drive(&supervisor, async {
            for _ in 0..16 {
                if *counter.lock().unwrap() == 3 {
                    return;
                }
                tokio::task::yield_now().await;
            }
            panic!("nested composite release did not drive all cleanups");
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn public_returned_root_retirement_wakes_idle_host_once() {
        let _lock = ON_CLEANUP_TEST_LOCK.lock().unwrap();
        let (runtime, supervisor, _host) = hosted_fixture_runtime();
        let creation = CommandProfilingRoot::new();
        let creation_runtime = runtime.clone().with_command_profiling(creation.receipt());
        let _entry = creation_runtime.enter();
        let binding = crate::user_functions::test_support::exact_local_binding(&runtime, 7);
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let callback_origins = Arc::new(Mutex::new(Vec::new()));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
            profiled_callback_invoker(Arc::clone(&callback_origins)),
        ))
        .expect("call-port scope");
        let retired_roots = Arc::new(Mutex::new(Vec::new()));
        let retirement_observer_roots = Arc::clone(&retired_roots);
        let _observer = runmat_gc::RootRetirementObserver::register(Arc::new(move |event| {
            if let runmat_gc::RootRetirement::Explicit(handle) = event {
                let _nested = runmat_gc::RootRetirementObserver::register(Arc::new(|_| {}))
                    .expect("retirement observer may reenter registration");
                retirement_observer_roots
                    .lock()
                    .unwrap()
                    .push(format!("{handle:?}"));
            }
        }))
        .expect("observe synchronous producer allocation-root retirement");
        let cleanup = block_on(
            crate::dispatcher::call_builtin_async_with_outputs_in_context(
                &creation_runtime,
                "onCleanup",
                &[crate::user_functions::test_support::exact_local_value(
                    "cleanup", binding,
                )],
                1,
            ),
        )
        .expect("public onCleanup returns the producer root through canonical publication");
        let Value::HandleObject(handle) = cleanup.value() else {
            panic!("producer must return an onCleanup handle");
        };
        assert!(
            retired_roots
                .lock()
                .unwrap()
                .contains(&format!("{:?}", handle.target)),
            "allocation-root retirement must notify observers before publication"
        );
        assert_eq!(
            runtime
                .cleanup_registry()
                .expect("host registry")
                .find(handle.target)
                .expect("registered cleanup")
                .phase(),
            lifecycle::registry::CleanupPhase::Active,
            "the registry pin alone is not an outside owner; the producer lease prevents early queuing"
        );
        let returned = cleanup;
        drive(&supervisor, async { tokio::task::yield_now().await });
        assert!(
            callback_origins.lock().unwrap().is_empty(),
            "caller still owns result"
        );

        drop(returned);
        drive(&supervisor, async {
            for _ in 0..16 {
                if callback_origins.lock().unwrap().len() == 1 {
                    return;
                }
                tokio::task::yield_now().await;
            }
            panic!("idle root retirement did not drive cleanup");
        });
        assert_eq!(
            *callback_origins.lock().unwrap(),
            [false],
            "idle cleanup has no command origin and must not replay"
        );
        let created = creation.seal();
        assert_eq!(created.observations.expect("creation valid").missing, 0);
        assert_eq!(
            created
                .deferred_work
                .expect("creation deferred valid")
                .admitted,
            0
        );
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
        let registration = runtime.enter();
        let _bound = crate::user_functions::test_support::install_bound_forwarder(binding.clone());
        let counter = Arc::new(Mutex::new(0usize));
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(
            callback_invoker(Arc::clone(&counter)),
        ))
        .expect("call-port scope");
        drop(registration);
        runtime
            .clone()
            .scope(async {
                let cleanup = on_cleanup_builtin(
                    crate::user_functions::test_support::exact_local_value("cleanup", binding),
                )
                .await
                .expect("create browser cleanup");
                crate::dispatcher::call_builtin_async_with_outputs_in_context(
                    &runtime,
                    "__runmat_oncleanup_delete",
                    &[cleanup.value().clone()],
                    1,
                )
                .await
                .expect("browser delete");
            })
            .await;
        assert_eq!(*counter.lock().unwrap(), 1);
    }
}
