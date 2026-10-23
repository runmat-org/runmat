use super::*;

pub(in crate::builtins::timing::timer) struct TestClock(
    pub(in crate::builtins::timing::timer) Arc<Mutex<Duration>>,
);

#[derive(Debug)]
struct TestSourceProvenance(SourceId);

impl SourceEvaluationProvenance for TestSourceProvenance {
    fn source_id(&self) -> SourceId {
        self.0
    }
}

pub(in crate::builtins::timing::timer) struct RecordingSourceEvaluator(
    pub(in crate::builtins::timing::timer) Arc<Mutex<Vec<String>>>,
);

impl RuntimeSourceEvaluationService for RecordingSourceEvaluator {
    fn bind_background_authority(&self) -> BuiltinResult<Rc<dyn RuntimeSourceEvaluationService>> {
        Ok(Rc::new(Self(self.0.clone())))
    }

    fn capture_session_source(
        &self,
        runtime: &RuntimeContext,
        source: Arc<str>,
        source_label: Arc<str>,
        requested_outputs: usize,
    ) -> BuiltinResult<SessionSourceEvaluation> {
        assert_eq!(requested_outputs, 0);
        Ok(SessionSourceEvaluation::capture(
            runtime,
            source,
            Rc::new(TestSourceProvenance(SourceId(91))),
            source_label,
            requested_outputs,
        ))
    }

    fn evaluate_session_source(
        &self,
        _: RuntimeContext,
        request: SessionSourceEvaluation,
    ) -> RuntimeServiceFuture<BuiltinResult<crate::execution::RootedValueSequence>> {
        self.0.lock().unwrap().push(request.source().to_string());
        Box::pin(async {
            crate::execution::RootedValueSequence::new(crate::sequence::ValueSequence::empty())
                .map_err(|error| error.into_runtime_error())
        })
    }
}

impl RuntimeTimingService for TestClock {
    fn monotonic_elapsed(&self) -> Duration {
        *self.0.lock().unwrap()
    }

    fn wall_time_since_epoch(&self) -> Duration {
        Duration::from_secs(1_700_000_000) + self.monotonic_elapsed()
    }

    fn local_datetime_to_instant(
        &self,
        local: chrono::NaiveDateTime,
    ) -> Option<crate::context::RuntimeCalendarInstant> {
        Some(crate::context::RuntimeCalendarInstant::from_naive_utc(
            local,
        ))
    }

    fn local_datetime_at_instant(
        &self,
        instant: crate::context::RuntimeCalendarInstant,
    ) -> Option<chrono::NaiveDateTime> {
        instant.at_fixed_offset(0)
    }

    fn process_cpu_seconds(&self) -> Option<f64> {
        None
    }

    fn sleep(&self, duration: Duration) -> RuntimeServiceFuture<Result<(), RuntimeError>> {
        let now = self.0.clone();
        Box::pin(async move {
            *now.lock().unwrap() += duration;
            Ok(())
        })
    }
}

struct TimerTestSession {
    runtime: RuntimeContext,
    supervisor: Rc<RuntimeLocalTaskSupervisor>,
    bindings: Vec<BoundFunctionBinding>,
}

pub(in crate::builtins::timing::timer) struct TimerInvokerGuards {
    _semantic: crate::user_functions::FunctionInvokerGuard,
    _bound: crate::user_functions::BoundFunctionInvokerGuard,
}

thread_local! {
    static TEST_SESSION: RefCell<Option<TimerTestSession>> = const { RefCell::new(None) };
}

pub(in crate::builtins::timing::timer) fn set_test_session(
    runtime: RuntimeContext,
    supervisor: Rc<RuntimeLocalTaskSupervisor>,
) {
    let bindings = (1..=3)
        .map(|function| {
            crate::user_functions::test_support::exact_local_binding(&runtime, function)
        })
        .collect();
    TEST_SESSION.with(|session| {
        *session.borrow_mut() = Some(TimerTestSession {
            runtime,
            supervisor,
            bindings,
        });
    });
}

pub(in crate::builtins::timing::timer) fn timer_callback_value(
    name: &str,
    function: usize,
) -> Value {
    TEST_SESSION.with(|session| {
        let session = session.borrow();
        let binding = session
            .as_ref()
            .expect("test session")
            .bindings
            .get(function.checked_sub(1).expect("fixture function number"))
            .expect("registered fixture binding")
            .clone();
        assert_eq!(binding.function, function);
        crate::user_functions::test_support::exact_local_value(name, binding)
    })
}

pub(in crate::builtins::timing::timer) fn reset_timer_state_for_tests() {
    reset_timer_state_for_tests_with_clock(None);
}

pub(in crate::builtins::timing::timer) fn reset_timer_state_for_tests_with_clock(
    clock: Option<Arc<Mutex<Duration>>>,
) {
    let supervisor = Rc::new(RuntimeLocalTaskSupervisor::new());
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let owner = runtime.local_identity();
    let mut ports =
        RuntimeServicePorts::default().with_local_tasks(supervisor.submission_port(owner));
    if let Some(clock) = clock {
        ports = ports.with_timing(Rc::new(TestClock(clock)));
    }
    set_test_session(runtime.with_service_ports(ports), supervisor);
}

pub(in crate::builtins::timing::timer) fn test_runtime() -> RuntimeContext {
    TEST_SESSION.with(|session| {
        session
            .borrow()
            .as_ref()
            .expect("test session")
            .runtime
            .clone()
    })
}

pub(in crate::builtins::timing::timer) fn test_supervisor() -> Rc<RuntimeLocalTaskSupervisor> {
    TEST_SESSION.with(|session| {
        session
            .borrow()
            .as_ref()
            .expect("test session")
            .supervisor
            .clone()
    })
}

pub(in crate::builtins::timing::timer) fn install_timer_invoker(
    invoker: Option<Arc<crate::user_functions::UnscopedFunctionInvoker>>,
) -> TimerInvokerGuards {
    let _scope = test_runtime().enter();
    let semantic =
        crate::user_functions::install_semantic_function_invoker(invoker).expect("call-port scope");
    let bound = install_timer_bound_forwarders();
    TimerInvokerGuards {
        _semantic: semantic,
        _bound: bound,
    }
}

pub(in crate::builtins::timing::timer) fn install_timer_bound_forwarders(
) -> crate::user_functions::BoundFunctionInvokerGuard {
    let bindings = TEST_SESSION.with(|session| {
        session
            .borrow()
            .as_ref()
            .expect("test session")
            .bindings
            .clone()
    });
    crate::user_functions::test_support::install_bound_forwarders(bindings)
}

pub(in crate::builtins::timing::timer) async fn timer_builtin(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let runtime = test_runtime();
    let args = crate::call::BuiltinArguments::from_borrowed(&args, Some(&runtime))?;
    super::super::timer_builtin(Some(runtime), args).await
}

pub(in crate::builtins::timing::timer) async fn timerfind_builtin(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let runtime = test_runtime();
    let args = crate::call::BuiltinArguments::from_borrowed(&args, Some(&runtime))?;
    super::super::timerfind_builtin(Some(runtime), args).await
}

pub(in crate::builtins::timing::timer) async fn timerfindall_builtin(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let runtime = test_runtime();
    let args = crate::call::BuiltinArguments::from_borrowed(&args, Some(&runtime))?;
    super::super::timerfindall_builtin(Some(runtime), args).await
}

pub(in crate::builtins::timing::timer) async fn timer_start_builtin(
    value: Value,
) -> BuiltinResult<Value> {
    super::super::timer_start_builtin(Some(test_runtime()), None, value).await
}

pub(in crate::builtins::timing::timer) async fn timer_wait_builtin(
    value: Value,
) -> BuiltinResult<Value> {
    let runtime = test_runtime();
    test_supervisor()
        .drive_until(super::super::timer_wait_builtin(Some(runtime), None, value))
        .await?
}

pub(in crate::builtins::timing::timer) async fn timer_delete_builtin(
    value: Value,
) -> BuiltinResult<Value> {
    super::super::timer_delete_builtin(Some(test_runtime()), None, value).await
}

pub(in crate::builtins::timing::timer) async fn timer_stop_builtin(
    value: Value,
) -> BuiltinResult<Value> {
    super::super::timer_stop_builtin(Some(test_runtime()), None, value).await
}
