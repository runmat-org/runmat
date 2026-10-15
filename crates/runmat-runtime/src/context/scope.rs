use super::RuntimeContext;
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

runmat_thread_local! {
    static ACTIVE_CONTEXTS: RefCell<Vec<RuntimeContext>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn active_runtime_context() -> Option<RuntimeContext> {
    ACTIVE_CONTEXTS.with(|contexts| contexts.borrow().last().cloned())
}

pub(super) fn execution_provider_authorization(
    admission: runmat_accelerate_api::ExecutionProviderAdmission<'_>,
) -> runmat_accelerate_api::ExecutionProviderAuthorization {
    use runmat_accelerate_api::ExecutionProviderAuthorization as Authorization;
    use runmat_accelerate_api::{ExecutionDeviceEffect, ExecutionProviderAdmission};

    let selected = runmat_execution::profiling::current_command_profiling();
    let context = active_runtime_context();
    if let Some(selected) = &selected {
        let matching = context
            .as_ref()
            .and_then(RuntimeContext::command_profiling)
            .is_some_and(|owner| selected.same_command(owner));
        if !matching {
            selected.mark_unproven_device_route();
        }
    }
    if let Some(context) = &context {
        if let Some(context_owner) = context.command_profiling() {
            if !selected
                .as_ref()
                .is_some_and(|owner| owner.same_command(context_owner))
            {
                context_owner.mark_unproven_device_route();
            }
        }
    }

    match admission {
        ExecutionProviderAdmission::CanonicalWgpuSubmit(access) => {
            let matching = context
                .as_ref()
                .and_then(RuntimeContext::command_device_access)
                .is_some_and(|expected| expected.same_device(access) && !access.has_raw_export());
            if !matching {
                if let Some(owner) = selected.or_else(|| {
                    context
                        .as_ref()
                        .and_then(RuntimeContext::command_profiling)
                        .cloned()
                }) {
                    owner.mark_unproven_device_route();
                }
            }
            Authorization::Unrestricted
        }
        ExecutionProviderAdmission::DeviceEffect(effect) => {
            if let Some(owner) = selected
                .or_else(|| context.and_then(|context| context.command_profiling().cloned()))
            {
                match effect {
                    ExecutionDeviceEffect::ImplicitQueueTransfer => {
                        owner.mark_unproven_device_time()
                    }
                    ExecutionDeviceEffect::RawWgpuExport
                    | ExecutionDeviceEffect::RawNativeExport
                    | ExecutionDeviceEffect::PlotDeviceEmission
                    | ExecutionDeviceEffect::ForeignExecution => owner.mark_unproven_device_route(),
                }
            }
            Authorization::Unrestricted
        }
        ExecutionProviderAdmission::Provider(provider) => {
            let authorization = context
                .as_ref()
                .and_then(RuntimeContext::execution_assignment)
                .map_or(Authorization::Unrestricted, |assignment| {
                    if assignment.resources.accelerator_leases.iter().any(|lease| {
                        runmat_accelerate_api::provider_matches_execution_lease(provider, lease)
                    }) {
                        Authorization::Allowed
                    } else {
                        Authorization::Denied
                    }
                });
            if authorization != Authorization::Denied {
                if let Some(owner) = selected {
                    let canonical = context
                        .as_ref()
                        .and_then(RuntimeContext::command_device_provider)
                        .is_some_and(|expected| {
                            runmat_accelerate_api::same_provider_instance(provider, expected)
                        });
                    if !canonical {
                        owner.mark_unproven_device_route();
                    }
                }
            }
            authorization
        }
    }
}

#[must_use]
pub struct RuntimeContextGuard {
    state_identity: *const super::RuntimeContextState,
    _display_format: runmat_value::DisplayFormatSelectionGuard,
    _command_profiling: runmat_execution::profiling::CommandProfilingSelectionGuard,
    _tape: crate::builtins::deep_learning::autodiff::TapePollGuard,
}

impl RuntimeContextGuard {
    pub(crate) fn enter(context: RuntimeContext) -> Self {
        let state_identity = context.state_identity();
        let display_format =
            runmat_value::select_display_format_owner(context.display_format_owner());
        let command_profiling = runmat_execution::profiling::select_command_profiling(
            context.command_profiling().cloned(),
        );
        let tape = crate::builtins::deep_learning::autodiff::TapePollGuard::select(
            context.tape_poll_authority(),
        );
        ACTIVE_CONTEXTS.with(|contexts| contexts.borrow_mut().push(context));
        Self {
            state_identity,
            _display_format: display_format,
            _command_profiling: command_profiling,
            _tape: tape,
        }
    }
}

impl Drop for RuntimeContextGuard {
    fn drop(&mut self) {
        ACTIVE_CONTEXTS.with(|contexts| {
            let popped = contexts.borrow_mut().pop();
            debug_assert_eq!(
                popped.as_ref().map(RuntimeContext::state_identity),
                Some(self.state_identity),
                "runtime contexts must unwind in stack order"
            );
        });
    }
}

/// Future adapter that activates the same context for every poll and while the
/// inner future is destroyed. It never keeps ambient state installed across a
/// yield, so independently polled sessions cannot observe one another.
pub struct ContextFuture<F> {
    context: RuntimeContext,
    future: Option<Pin<Box<F>>>,
}

impl<F> ContextFuture<F> {
    pub(super) fn new(context: RuntimeContext, future: Pin<Box<F>>) -> Self {
        Self {
            context: context
                .with_poll_source_catalog()
                .with_poll_debug_frames()
                .with_poll_call_projection()
                .with_poll_console_capture_projection()
                .with_poll_output_projection(),
            future: Some(future),
        }
    }
}

impl<F: Future> Future for ContextFuture<F> {
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        let _guard = RuntimeContextGuard::enter(this.context.clone());
        let result = this
            .future
            .as_mut()
            .expect("context future polled after completion")
            .as_mut()
            .poll(cx);
        if result.is_ready() {
            // Completion destructors run under this invocation before observing
            // receipts they may complete or invalidate.
            drop(this.future.take());
            if let Some(placement) = this.context.service_ports().placement() {
                placement.drain_feedback();
            }
        }
        result
    }
}

impl<F> Drop for ContextFuture<F> {
    fn drop(&mut self) {
        if let Some(future) = self.future.take() {
            let _guard = RuntimeContextGuard::enter(self.context.clone());
            drop(future);
            if let Some(placement) = self.context.service_ports().placement() {
                placement.drain_feedback();
            }
        }
    }
}

impl<F> Unpin for ContextFuture<F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::RuntimeExecutionStack;
    use crate::execution::RuntimeExecutionService;
    use runmat_execution::profiling::{
        current_command_profiling, CommandProfilingRoot, ProviderObservationTerminal,
    };
    use runmat_types::SourceId;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::task::Wake;

    struct NoopWake;

    struct ExecutionProvider {
        id: &'static str,
        device_id: u32,
    }

    impl runmat_accelerate_api::AccelProvider for ExecutionProvider {
        fn upload(
            &self,
            _host: &runmat_accelerate_api::HostTensorView,
        ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
            anyhow::bail!("not used")
        }

        fn download<'a>(
            &'a self,
            _handle: &'a runmat_accelerate_api::GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            Box::pin(async { anyhow::bail!("not used") })
        }

        fn device_info(&self) -> String {
            self.id.into()
        }

        fn device_id(&self) -> u32 {
            self.device_id
        }

        fn device_info_struct(&self) -> runmat_accelerate_api::ApiDeviceInfo {
            runmat_accelerate_api::ApiDeviceInfo {
                device_id: self.device_id,
                name: self.id.into(),
                vendor: "RunMat".into(),
                memory_bytes: Some(4096),
                backend: Some("test".into()),
            }
        }

        fn execution_accelerator_device(
            &self,
            inventory_epoch: u64,
        ) -> anyhow::Result<Option<runmat_execution::resource::AcceleratorDevice>> {
            runmat_accelerate_api::execution_accelerator_device(
                self,
                self.id,
                "1.0.0",
                runmat_accelerate_api::ExecutionProviderContract::GenericComputeV1,
                &format!("test-device-{}", self.device_id),
                &format!("test-device-{}", self.device_id),
                inventory_epoch,
            )
            .map(Some)
        }
    }

    static LEASED_PROVIDER: ExecutionProvider = ExecutionProvider {
        id: "runmat.test.leased",
        device_id: 901,
    };
    static OTHER_PROVIDER: ExecutionProvider = ExecutionProvider {
        id: "runmat.test.other",
        device_id: 902,
    };

    impl Wake for NoopWake {
        fn wake(self: Arc<Self>) {}
    }

    #[test]
    fn nested_runtime_guards_restore_exact_command_profiling_owner() {
        let first_receipt = CommandProfilingRoot::new();
        let second_receipt = CommandProfilingRoot::new();
        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_command_profiling(first_receipt.receipt());
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_command_profiling(second_receipt.receipt());
        let _first = first.enter();
        let second_guard = second.enter();
        current_command_profiling()
            .expect("second selected")
            .begin_provider_operation()
            .expect("second contribution")
            .complete(ProviderObservationTerminal::Failed);
        drop(second_guard);
        current_command_profiling()
            .expect("first restored")
            .begin_provider_operation()
            .expect("first contribution")
            .complete(ProviderObservationTerminal::Missing);
        assert_eq!(
            first_receipt
                .seal()
                .observations
                .expect("first valid")
                .missing,
            1
        );
        assert_eq!(
            second_receipt
                .seal()
                .observations
                .expect("second valid")
                .failed,
            1
        );
    }

    #[test]
    fn scheduled_context_authorizes_only_its_exact_accelerator_lease() {
        use runmat_accelerate_api::AccelProvider as _;
        use runmat_execution::identity::{AttemptId, ExecutionScopeId, PoolId, TaskId, WorkerId};
        use runmat_execution::resource::{AcceleratorDeviceLease, ResourceAssignment};

        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let attempt_id = AttemptId::derive(&[b"provider-authorization"]);
        let device = LEASED_PROVIDER
            .execution_accelerator_device(7)
            .unwrap()
            .unwrap();
        let lease = AcceleratorDeviceLease::for_attempt(attempt_id, 11, &device).unwrap();
        let assignment = runmat_execution::ProgramExecutionAssignment {
            scope_id: ExecutionScopeId::derive(&[b"scope"]),
            pool_id: PoolId::derive(&[b"pool"]),
            task_id: TaskId::derive(&[b"task"]),
            attempt_id,
            worker_id: WorkerId::derive(&[b"worker"]),
            backend: runmat_execution::PoolBackend::LocalProcesses,
            resources: ResourceAssignment {
                accelerator_leases: vec![lease],
            },
        };

        let _context = runtime.enter();
        assert_eq!(
            execution_provider_authorization(
                runmat_accelerate_api::ExecutionProviderAdmission::Provider(&LEASED_PROVIDER)
            ),
            runmat_accelerate_api::ExecutionProviderAuthorization::Unrestricted
        );
        let _assignment = runtime.enter_execution_assignment(Some(assignment));
        assert_eq!(
            execution_provider_authorization(
                runmat_accelerate_api::ExecutionProviderAdmission::Provider(&LEASED_PROVIDER)
            ),
            runmat_accelerate_api::ExecutionProviderAuthorization::Allowed
        );
        assert_eq!(
            execution_provider_authorization(
                runmat_accelerate_api::ExecutionProviderAdmission::Provider(&OTHER_PROVIDER)
            ),
            runmat_accelerate_api::ExecutionProviderAuthorization::Denied
        );
    }

    #[test]
    fn synchronous_context_guard_restores_the_previous_context() {
        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let first_scope = first.execution().scope_id();
        let second_scope = second.execution().scope_id();
        {
            let _first = first.enter();
            assert_eq!(
                active_runtime_context().unwrap().execution().scope_id(),
                first_scope
            );
            {
                let _second = second.enter();
                assert_eq!(
                    active_runtime_context().unwrap().execution().scope_id(),
                    second_scope
                );
            }
            assert_eq!(
                active_runtime_context().unwrap().execution().scope_id(),
                first_scope
            );
        }
        assert!(active_runtime_context().is_none());
    }

    #[test]
    fn execution_stack_guard_restores_nested_stack_state() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        assert_eq!(runtime.execution_stack(), RuntimeExecutionStack::Process);
        {
            let _segmented = runtime.enter_execution_stack(RuntimeExecutionStack::Segmented);
            assert_eq!(runtime.execution_stack(), RuntimeExecutionStack::Segmented);
            {
                let _process = runtime.enter_execution_stack(RuntimeExecutionStack::Process);
                assert_eq!(runtime.execution_stack(), RuntimeExecutionStack::Process);
            }
            assert_eq!(runtime.execution_stack(), RuntimeExecutionStack::Segmented);
        }
        assert_eq!(runtime.execution_stack(), RuntimeExecutionStack::Process);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn interleaved_context_futures_restore_the_active_session() {
        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let first_scope = first.execution().scope_id();
        let second_scope = second.execution().scope_id();

        let mut first_future = Box::pin(first.scope(std::future::poll_fn(move |_| {
            assert_eq!(
                active_runtime_context()
                    .expect("first context active")
                    .execution()
                    .scope_id(),
                first_scope
            );
            Poll::<()>::Pending
        })));
        let mut second_future = Box::pin(second.scope(std::future::poll_fn(move |_| {
            assert_eq!(
                active_runtime_context()
                    .expect("second context active")
                    .execution()
                    .scope_id(),
                second_scope
            );
            Poll::<()>::Pending
        })));
        let waker = std::task::Waker::from(Arc::new(NoopWake));
        let mut cx = Context::from_waker(&waker);

        assert!(first_future.as_mut().poll(&mut cx).is_pending());
        assert!(active_runtime_context().is_none());
        assert!(second_future.as_mut().poll(&mut cx).is_pending());
        assert!(active_runtime_context().is_none());
    }

    #[test]
    fn interleaved_futures_isolate_source_output_and_cancellation_state() {
        async fn session_future(
            source_id: SourceId,
            source: &'static str,
            outputs: usize,
            cancelled: bool,
        ) {
            let _catalog = crate::source_context::replace_source_catalog(vec![(
                source_id,
                format!("session-{}.m", source_id.0),
                source.to_string(),
            )])
            .expect("session source catalog admission");
            let _source = crate::source_context::replace_current_source_id(Some(source_id));
            let _outputs = crate::output_count::push_output_count(Some(outputs));
            let mut yielded = false;
            std::future::poll_fn(move |_| {
                if yielded {
                    Poll::Ready(())
                } else {
                    yielded = true;
                    Poll::Pending
                }
            })
            .await;
            assert_eq!(
                crate::source_context::current_source().as_deref(),
                Some(source)
            );
            assert_eq!(crate::output_count::current_output_count(), Some(outputs));
            assert_eq!(crate::interrupt::is_cancelled(), cancelled);
        }

        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let first = RuntimeContext::with_cancellation(
            Rc::new(RuntimeExecutionService::new()),
            cancellation.token(),
        );
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        cancellation.cancel(runmat_execution::CancellationReason::User);
        let mut first_future = Box::pin(first.scope(session_future(SourceId(1), "first", 1, true)));
        let mut second_future =
            Box::pin(second.scope(session_future(SourceId(2), "second", 2, false)));
        let waker = std::task::Waker::from(Arc::new(NoopWake));
        let mut cx = Context::from_waker(&waker);

        assert!(first_future.as_mut().poll(&mut cx).is_pending());
        assert!(second_future.as_mut().poll(&mut cx).is_pending());
        assert!(first_future.as_mut().poll(&mut cx).is_ready());
        assert!(second_future.as_mut().poll(&mut cx).is_ready());
        assert!(crate::source_context::current_source().is_none());
        assert_eq!(crate::output_count::current_output_count(), None);
    }

    #[test]
    fn interleaved_futures_isolate_host_interaction_handlers() {
        async fn session_future(label: &'static str) {
            let handler: Arc<crate::interaction::AsyncInteractionHandler> = Arc::new(move |_| {
                Box::pin(async move {
                    Ok(crate::interaction::InteractionResponse::Line(
                        label.to_string(),
                    ))
                })
            });
            let _handler = crate::interaction::replace_async_handler(Some(handler));
            let mut yielded = false;
            std::future::poll_fn(move |_| {
                if yielded {
                    Poll::Ready(())
                } else {
                    yielded = true;
                    Poll::Pending
                }
            })
            .await;
            assert_eq!(
                crate::interaction::request_line_async("", false)
                    .await
                    .unwrap(),
                label
            );
        }

        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let mut first_future = Box::pin(first.scope(session_future("first")));
        let mut second_future = Box::pin(second.scope(session_future("second")));
        let waker = std::task::Waker::from(Arc::new(NoopWake));
        let mut cx = Context::from_waker(&waker);
        assert!(first_future.as_mut().poll(&mut cx).is_pending());
        assert!(second_future.as_mut().poll(&mut cx).is_pending());
        assert!(first_future.as_mut().poll(&mut cx).is_ready());
        assert!(second_future.as_mut().poll(&mut cx).is_ready());
    }

    #[test]
    fn dropping_scoped_future_activates_its_context_for_guard_cleanup() {
        struct DropProbe(runmat_execution::ExecutionScopeId);

        impl Drop for DropProbe {
            fn drop(&mut self) {
                assert_eq!(
                    active_runtime_context()
                        .expect("context active while inner future drops")
                        .execution()
                        .scope_id(),
                    self.0
                );
            }
        }

        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let scope_id = runtime.execution().scope_id();
        let future = async move {
            let _probe = DropProbe(scope_id);
            std::future::pending::<()>().await;
        };
        let mut future = Box::pin(runtime.scope(future));
        let waker = std::task::Waker::from(Arc::new(NoopWake));
        let mut cx = Context::from_waker(&waker);
        assert!(future.as_mut().poll(&mut cx).is_pending());
        drop(future);
        assert!(active_runtime_context().is_none());
    }

    #[test]
    fn workspace_resolver_registration_is_context_scoped_in_production_shape() {
        fn first_lookup(name: &str) -> Option<runmat_value::Value> {
            (name == "x").then_some(runmat_value::Value::Num(1.0))
        }
        fn second_lookup(name: &str) -> Option<runmat_value::Value> {
            (name == "x").then_some(runmat_value::Value::Num(2.0))
        }
        fn empty_snapshot() -> Vec<(String, runmat_value::Value)> {
            Vec::new()
        }
        fn empty_globals() -> Vec<String> {
            Vec::new()
        }

        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        {
            let _scope = RuntimeContextGuard::enter(first.clone());
            crate::workspace::register_workspace_resolver(crate::workspace::WorkspaceResolver {
                lookup: first_lookup,
                snapshot: empty_snapshot,
                globals: empty_globals,
                assign: None,
                clear: None,
                remove: None,
            });
        }
        {
            let _scope = RuntimeContextGuard::enter(second.clone());
            crate::workspace::register_workspace_resolver(crate::workspace::WorkspaceResolver {
                lookup: second_lookup,
                snapshot: empty_snapshot,
                globals: empty_globals,
                assign: None,
                clear: None,
                remove: None,
            });
        }
        {
            let _scope = RuntimeContextGuard::enter(first);
            assert_eq!(
                crate::workspace::lookup("x"),
                Some(runmat_value::Value::Num(1.0))
            );
        }
        {
            let _scope = RuntimeContextGuard::enter(second);
            assert_eq!(
                crate::workspace::lookup("x"),
                Some(runmat_value::Value::Num(2.0))
            );
        }
    }

    #[test]
    fn class_and_static_property_state_is_session_owned() {
        fn class(name: &str) -> crate::class_registry::RuntimeClass {
            crate::class_registry::RuntimeClass {
                events: Default::default(),
                name: name.into(),
                parent: None,
                properties: std::collections::HashMap::from([(
                    "answer".into(),
                    crate::class_registry::RuntimeProperty {
                        name: "answer".into(),
                        is_static: true,
                        is_constant: false,
                        is_dependent: false,
                        get_access: runmat_types::MemberAccess::Public,
                        set_access: runmat_types::MemberAccess::Public,
                        default_value: None,
                    },
                )]),
                methods: std::collections::HashMap::new(),
            }
        }

        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let only_first = runmat_types::ClassIdentity::from("OnlyFirst");
        {
            let _scope = RuntimeContextGuard::enter(first.clone());
            crate::class_registry::register_class(class("OnlyFirst"));
            crate::class_registry::set_static_property_value(
                &only_first,
                "answer",
                runmat_value::Value::Num(1.0),
            );
        }
        {
            let _scope = RuntimeContextGuard::enter(second);
            assert!(crate::class_registry::get_class(&only_first).is_none());
            assert!(
                crate::class_registry::get_static_property_value(&only_first, "answer").is_none()
            );
        }
        {
            let _scope = RuntimeContextGuard::enter(first);
            assert!(crate::class_registry::get_class(&only_first).is_some());
            assert_eq!(
                crate::class_registry::get_static_property_value(&only_first, "answer"),
                Some(runmat_value::Value::Num(1.0))
            );
        }
    }

    #[test]
    fn lazy_builtin_class_registration_is_session_owned() {
        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        {
            let _scope = RuntimeContextGuard::enter(first.clone());
            crate::testing::ensure_testing_classes();
            assert!(
                crate::class_registry::get_class(&crate::testing::TEST_CASE_CLASS.owned())
                    .is_some()
            );
        }
        {
            let _scope = RuntimeContextGuard::enter(second.clone());
            assert!(
                crate::class_registry::get_class(&crate::testing::TEST_CASE_CLASS.owned())
                    .is_none()
            );
            crate::testing::ensure_testing_classes();
            assert!(
                crate::class_registry::get_class(&crate::testing::TEST_CASE_CLASS.owned())
                    .is_some()
            );
        }
        {
            let _scope = RuntimeContextGuard::enter(first);
            assert!(
                crate::class_registry::get_class(&crate::testing::TEST_CASE_CLASS.owned())
                    .is_some()
            );
        }
    }

    #[test]
    fn error_and_compatibility_policy_is_session_owned() {
        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        first.set_error_namespace("First");
        first.set_callstack_limit(11);
        second.set_error_namespace("Second");
        second.set_callstack_limit(22);
        {
            let _scope = RuntimeContextGuard::enter(first);
            crate::compatibility::set_runmat_extensions_enabled(true);
            assert_eq!(active_runtime_context().unwrap().error_namespace(), "First");
            assert_eq!(active_runtime_context().unwrap().callstack_limit(), 11);
            assert!(crate::compatibility::runmat_extensions_enabled());
        }
        {
            let _scope = RuntimeContextGuard::enter(second);
            assert_eq!(
                active_runtime_context().unwrap().error_namespace(),
                "Second"
            );
            assert_eq!(active_runtime_context().unwrap().callstack_limit(), 22);
            assert!(!crate::compatibility::runmat_extensions_enabled());
        }
    }
}
