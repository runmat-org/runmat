use super::*;
use crate::context::{
    RuntimeLocalTaskSupervisor, RuntimeServiceFuture, RuntimeServicePorts,
    RuntimeSourceEvaluationService, RuntimeTimingService, SessionSourceEvaluation,
    SourceEvaluationProvenance,
};
use crate::execution::RuntimeExecutionService;
use futures::executor::block_on;
use runmat_types::SourceId;
use runmat_value::{BoundFunctionBinding, CharArray, IntegerStorage, Tensor};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

struct TestClock(Arc<Mutex<Duration>>);

#[derive(Debug)]
struct TestSourceProvenance(SourceId);

impl SourceEvaluationProvenance for TestSourceProvenance {
    fn source_id(&self) -> SourceId {
        self.0
    }
}

struct RecordingSourceEvaluator(Arc<Mutex<Vec<String>>>);

impl RuntimeSourceEvaluationService for RecordingSourceEvaluator {
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
    ) -> RuntimeServiceFuture<BuiltinResult<crate::sequence::ValueSequence>> {
        self.0.lock().unwrap().push(request.source().to_string());
        Box::pin(async { Ok(crate::sequence::ValueSequence::empty()) })
    }
}

impl RuntimeTimingService for TestClock {
    fn monotonic_elapsed(&self) -> Duration {
        *self.0.lock().unwrap()
    }

    fn wall_time_since_epoch(&self) -> Duration {
        Duration::from_secs(1_700_000_000) + self.monotonic_elapsed()
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

struct TimerInvokerGuards {
    _semantic: crate::user_functions::FunctionInvokerGuard,
    _bound: crate::user_functions::BoundFunctionInvokerGuard,
}

thread_local! {
    static TEST_SESSION: RefCell<Option<TimerTestSession>> = const { RefCell::new(None) };
}

fn set_test_session(runtime: RuntimeContext, supervisor: Rc<RuntimeLocalTaskSupervisor>) {
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

fn timer_callback_value(name: &str, function: usize) -> Value {
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

fn reset_timer_state_for_tests() {
    reset_timer_state_for_tests_with_clock(None);
}

fn reset_timer_state_for_tests_with_clock(clock: Option<Arc<Mutex<Duration>>>) {
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

fn test_runtime() -> RuntimeContext {
    TEST_SESSION.with(|session| {
        session
            .borrow()
            .as_ref()
            .expect("test session")
            .runtime
            .clone()
    })
}

fn test_supervisor() -> Rc<RuntimeLocalTaskSupervisor> {
    TEST_SESSION.with(|session| {
        session
            .borrow()
            .as_ref()
            .expect("test session")
            .supervisor
            .clone()
    })
}

fn install_timer_invoker(
    invoker: Option<Arc<crate::user_functions::FunctionInvoker>>,
) -> TimerInvokerGuards {
    let _scope = test_runtime().enter();
    let semantic = crate::user_functions::install_semantic_function_invoker(invoker);
    let bound = install_timer_bound_forwarders();
    TimerInvokerGuards {
        _semantic: semantic,
        _bound: bound,
    }
}

fn install_timer_bound_forwarders() -> crate::user_functions::BoundFunctionInvokerGuard {
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

async fn timer_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    super::timer_builtin(Some(test_runtime()), args).await
}

async fn timerfind_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    super::timerfind_builtin(Some(test_runtime()), args).await
}

async fn timerfindall_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    super::timerfindall_builtin(Some(test_runtime()), args).await
}

async fn timer_start_builtin(value: Value) -> BuiltinResult<Value> {
    super::timer_start_builtin(Some(test_runtime()), None, value).await
}

async fn timer_wait_builtin(value: Value) -> BuiltinResult<Value> {
    let runtime = test_runtime();
    test_supervisor()
        .drive_until(super::timer_wait_builtin(Some(runtime), None, value))
        .await?
}

async fn timer_delete_builtin(value: Value) -> BuiltinResult<Value> {
    super::timer_delete_builtin(Some(test_runtime()), None, value).await
}

async fn timer_stop_builtin(value: Value) -> BuiltinResult<Value> {
    super::timer_stop_builtin(Some(test_runtime()), None, value).await
}

#[test]
fn timer_method_status_and_relative_startat_are_gated_before_lifecycle_effects() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let timer = block_on(timer_builtin(Vec::new())).expect("timer");
    let runtime = test_runtime();
    let assigned = block_on(super::timer_start_builtin(
        Some(runtime.clone()),
        Some(1),
        timer.clone(),
    ))
    .expect_err("assigned start status is a RunMat extension");
    assert_eq!(
        assigned.identifier(),
        Some("RunMat:compatibility:TimerMethodStatusOutputExtension")
    );
    let relative = block_on(super::timer_startat_builtin(
        Some(runtime.clone()),
        Some(0),
        timer.clone(),
        vec![Value::Num(0.1)],
    ))
    .expect_err("relative startat is a RunMat extension");
    assert_eq!(
        relative.identifier(),
        Some("RunMat:compatibility:TimerStartatRelativeExtension")
    );
    let Value::HandleObject(handle) = &timer else {
        panic!("expected timer handle")
    };
    assert_eq!(string_property(handle, "Running").unwrap(), "off");
    runtime.state().runmat_extensions_enabled.set(true);
    assert!(matches!(
        block_on(super::timer_stop_builtin(Some(runtime), Some(1), timer)),
        Ok(Value::Num(0.0))
    ));
}

#[test]
fn timer_constructor_sets_defaults_and_name_values() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let value = block_on(timer_builtin(vec![
        Value::String("Name".into()),
        Value::String("solver".into()),
        Value::String("Tag".into()),
        Value::String("client".into()),
        Value::String("StartDelay".into()),
        Value::Num(0.25),
    ]))
    .expect("timer");
    let Value::HandleObject(handle) = value else {
        panic!("expected timer handle");
    };
    assert_eq!(string_property(&handle, "Name").unwrap(), "solver");
    assert_eq!(string_property(&handle, "Tag").unwrap(), "client");
    assert_eq!(numeric_property(&handle, "StartDelay").unwrap(), 0.25);
    assert_eq!(string_property(&handle, "Running").unwrap(), "off");
    assert!(crate::is_handle_valid(&handle));
}

#[test]
fn timer_family_declares_exact_integer_property_metadata() {
    for (name, capabilities) in [("timer", 2), ("timerfind", 1), ("timerfindall", 1)] {
        assert!(runmat_builtins::builtin_function_by_name(name).is_none());
        let builtin = runmat_builtins::builtin_catalog_entry_by_name(name)
            .expect("canonical timer catalog identity");
        assert_eq!(builtin.integer_capabilities.len(), capabilities, "{name}");
        assert_eq!(builtin.extensions.len(), 1, "{name}");
        assert!(builtin.integer_capabilities.iter().all(|capability| {
            capability
                .inputs
                .iter()
                .all(|input| input.classes.len() == 8)
        }));
    }
}

#[test]
fn timerfind_filters_visible_timers() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let _hidden = block_on(timer_builtin(vec![
        Value::String("Name".into()),
        Value::String("hidden".into()),
        Value::String("ObjectVisibility".into()),
        Value::String("off".into()),
    ]))
    .expect("hidden timer");
    let visible = block_on(timer_builtin(vec![
        Value::String("Name".into()),
        Value::String("visible".into()),
        Value::String("Tag".into()),
        Value::String("batch".into()),
    ]))
    .expect("visible timer");
    let found = block_on(timerfind_builtin(vec![
        Value::String("Tag".into()),
        Value::String("batch".into()),
    ]))
    .expect("timerfind");
    assert_eq!(found, visible);
    let all = block_on(timerfindall_builtin(vec![])).expect("timerfindall");
    let Value::Cell(cell) = all else {
        panic!("expected cell row of all timers");
    };
    assert_eq!((cell.rows, cell.cols), (1, 2));
}

#[test]
fn timer_start_runs_function_callback_and_updates_state() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let calls = Arc::new(Mutex::new(0usize));
    let invoker_calls = calls.clone();
    let _guard =
        install_timer_invoker(Some(Arc::new(move |_function, args, requested_outputs| {
            assert_eq!(requested_outputs, 0);
            let args = args.to_vec();
            let invoker_calls = Arc::clone(&invoker_calls);
            Box::pin(async move {
                assert_eq!(args.len(), 2);
                let Value::HandleObject(handle) = &args[0] else {
                    panic!("expected timer handle");
                };
                assert!(is_timer_handle(handle));
                let Value::Struct(event) = &args[1] else {
                    panic!("expected event struct");
                };
                assert!(event.fields.contains_key("Type"));
                *invoker_calls.lock().unwrap() += 1;
                Ok(crate::sequence::ValueSequence::empty())
            })
        })));
    let timer = block_on(timer_builtin(vec![
        Value::String("TimerFcn".into()),
        timer_callback_value("tick", 1),
        Value::String("ExecutionMode".into()),
        Value::String("fixedSpacing".into()),
        Value::String("TasksToExecute".into()),
        Value::Num(2.0),
        Value::String("Period".into()),
        Value::Num(0.002),
    ]))
    .expect("timer");
    block_on(timer_start_builtin(timer.clone())).expect("start");
    assert_eq!(
        *calls.lock().unwrap(),
        0,
        "start must return before callback execution"
    );
    block_on(timer_wait_builtin(timer.clone())).expect("wait");
    let Value::HandleObject(handle) = timer else {
        panic!("expected timer handle");
    };
    assert_eq!(*calls.lock().unwrap(), 2);
    assert_eq!(numeric_property(&handle, "TasksExecuted").unwrap(), 2.0);
    assert_eq!(string_property(&handle, "Running").unwrap(), "off");
}

#[test]
fn timer_stop_claims_stop_callback_once_and_wait_does_not_stop() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let calls = Arc::new(Mutex::new(Vec::<String>::new()));
    let recorded = calls.clone();
    let _guard = install_timer_invoker(Some(Arc::new(
        move |function, _args, _requested_outputs| {
            let recorded = recorded.clone();
            let name = format!("{function:?}");
            Box::pin(async move {
                recorded.lock().unwrap().push(name);
                Ok(crate::sequence::ValueSequence::empty())
            })
        },
    )));
    let timer = block_on(timer_builtin(vec![
        Value::String("TimerFcn".into()),
        timer_callback_value("tick", 1),
        Value::String("StopFcn".into()),
        timer_callback_value("done", 2),
        Value::String("StartDelay".into()),
        Value::Num(1.0),
    ]))
    .expect("timer");
    block_on(timer_start_builtin(timer.clone())).expect("start");
    assert!(calls.lock().unwrap().is_empty());
    block_on(timer_stop_builtin(timer.clone())).expect("stop");
    block_on(timer_stop_builtin(timer.clone())).expect("idempotent stop");
    block_on(timer_wait_builtin(timer)).expect("wait stopped timer");
    assert_eq!(calls.lock().unwrap().len(), 1, "StopFcn must run once");
}

#[test]
fn fixed_rate_drop_counts_missed_slots_without_executing_them() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    let clock = Arc::new(Mutex::new(Duration::ZERO));
    reset_timer_state_for_tests_with_clock(Some(clock.clone()));
    let calls = Arc::new(Mutex::new(0usize));
    let callback_calls = calls.clone();
    let callback_clock = clock.clone();
    let _guard = install_timer_invoker(Some(Arc::new(
        move |_function, _args, _requested_outputs| {
            let calls = callback_calls.clone();
            let clock = callback_clock.clone();
            Box::pin(async move {
                *calls.lock().unwrap() += 1;
                if *calls.lock().unwrap() == 1 {
                    *clock.lock().unwrap() += Duration::from_millis(250);
                }
                Ok(crate::sequence::ValueSequence::empty())
            })
        },
    )));
    let timer = block_on(timer_builtin(vec![
        Value::String("TimerFcn".into()),
        timer_callback_value("tick", 1),
        Value::String("ExecutionMode".into()),
        Value::String("fixedRate".into()),
        Value::String("BusyMode".into()),
        Value::String("drop".into()),
        Value::String("Period".into()),
        Value::Num(0.1),
        Value::String("TasksToExecute".into()),
        Value::Num(4.0),
    ]))
    .expect("timer");
    block_on(timer_start_builtin(timer.clone())).expect("start");
    block_on(timer_wait_builtin(timer.clone())).expect("wait");
    let Value::HandleObject(handle) = timer else {
        panic!("timer handle");
    };
    assert_eq!(*calls.lock().unwrap(), 2);
    assert_eq!(numeric_property(&handle, "TasksExecuted").unwrap(), 2.0);
    assert_eq!(*clock.lock().unwrap(), Duration::from_millis(300));
}

#[test]
fn fixed_delay_and_fixed_spacing_use_distinct_callback_anchors() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    for (mode, expected) in [
        ("fixedDelay", Duration::from_millis(250)),
        ("fixedSpacing", Duration::from_millis(350)),
    ] {
        let clock = Arc::new(Mutex::new(Duration::ZERO));
        reset_timer_state_for_tests_with_clock(Some(clock.clone()));
        let calls = Arc::new(Mutex::new(0usize));
        let callback_calls = calls.clone();
        let callback_clock = clock.clone();
        let _guard = install_timer_invoker(Some(Arc::new(
            move |_function, _args, _requested_outputs| {
                let calls = callback_calls.clone();
                let clock = callback_clock.clone();
                Box::pin(async move {
                    let mut count = calls.lock().unwrap();
                    *count += 1;
                    if *count == 1 {
                        *clock.lock().unwrap() += Duration::from_millis(250);
                    }
                    Ok(crate::sequence::ValueSequence::empty())
                })
            },
        )));
        let timer = block_on(timer_builtin(vec![
            Value::String("TimerFcn".into()),
            timer_callback_value("tick", 1),
            Value::String("ExecutionMode".into()),
            Value::String(mode.into()),
            Value::String("Period".into()),
            Value::Num(0.1),
            Value::String("TasksToExecute".into()),
            Value::Num(2.0),
        ]))
        .expect("timer");
        block_on(timer_start_builtin(timer.clone())).expect("start");
        block_on(timer_wait_builtin(timer)).expect("wait");
        assert_eq!(*calls.lock().unwrap(), 2, "{mode}");
        assert_eq!(*clock.lock().unwrap(), expected, "{mode}");
    }
}

#[test]
fn fixed_rate_queue_runs_due_work_and_error_reports_backlog() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    for (busy, expected_calls, expected_error) in [
        ("queue", 4usize, None),
        ("error", 1usize, Some("RunMat:timer:CallbackFailed")),
    ] {
        let clock = Arc::new(Mutex::new(Duration::ZERO));
        reset_timer_state_for_tests_with_clock(Some(clock.clone()));
        let calls = Arc::new(Mutex::new(0usize));
        let callback_calls = calls.clone();
        let callback_clock = clock.clone();
        let _guard = install_timer_invoker(Some(Arc::new(
            move |_function, _args, _requested_outputs| {
                let calls = callback_calls.clone();
                let clock = callback_clock.clone();
                Box::pin(async move {
                    let mut count = calls.lock().unwrap();
                    *count += 1;
                    if *count == 1 {
                        *clock.lock().unwrap() += Duration::from_millis(250);
                    }
                    Ok(crate::sequence::ValueSequence::empty())
                })
            },
        )));
        let timer = block_on(timer_builtin(vec![
            Value::String("TimerFcn".into()),
            timer_callback_value("tick", 1),
            Value::String("ExecutionMode".into()),
            Value::String("fixedRate".into()),
            Value::String("BusyMode".into()),
            Value::String(busy.into()),
            Value::String("Period".into()),
            Value::Num(0.1),
            Value::String("TasksToExecute".into()),
            Value::Num(4.0),
        ]))
        .expect("timer");
        block_on(timer_start_builtin(timer.clone())).expect("start");
        let result = block_on(timer_wait_builtin(timer.clone()));
        assert_eq!(
            result
                .err()
                .and_then(|error| error.identifier().map(str::to_string))
                .as_deref(),
            expected_error,
            "{busy}"
        );
        assert_eq!(*calls.lock().unwrap(), expected_calls, "{busy}");
        let Value::HandleObject(handle) = timer else {
            panic!("timer handle");
        };
        assert_eq!(string_property(&handle, "Running").unwrap(), "off");
    }
}

#[test]
fn reentrant_stop_and_restart_cannot_let_stale_generation_turn_running_off() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let runtime = test_runtime();
    let calls = Rc::new(std::cell::Cell::new(0usize));
    let callback_calls = calls.clone();
    let callback_runtime = runtime.clone();
    let invoker: Rc<crate::user_functions::FunctionInvoker> =
        Rc::new(move |_function, args, requested_outputs| {
            assert_eq!(requested_outputs, 0);
            let handle = args[0].clone();
            let calls = callback_calls.clone();
            let runtime = callback_runtime.clone();
            Box::pin(async move {
                let count = calls.get() + 1;
                calls.set(count);
                if count == 1 {
                    super::timer_stop_builtin(Some(runtime.clone()), None, handle.clone()).await?;
                    super::timer_start_builtin(Some(runtime), None, handle).await?;
                }
                Ok(crate::sequence::ValueSequence::empty())
            })
        });
    let scope = runtime.enter();
    let _guard = crate::user_functions::install_local_semantic_function_invoker(invoker);
    let _bound = install_timer_bound_forwarders();
    drop(scope);
    let timer = block_on(timer_builtin(vec![
        Value::String("TimerFcn".into()),
        timer_callback_value("tick", 1),
    ]))
    .expect("timer");
    block_on(timer_start_builtin(timer.clone())).expect("first start");
    block_on(timer_wait_builtin(timer.clone())).expect("wait follows restarted generation");
    assert_eq!(calls.get(), 2);
    let Value::HandleObject(handle) = timer else {
        panic!("timer handle");
    };
    assert_eq!(string_property(&handle, "Running").unwrap(), "off");
}

#[test]
fn stale_failure_after_restart_does_not_enter_new_generations_error_handler() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let runtime = test_runtime();
    let ticks = Rc::new(std::cell::Cell::new(0usize));
    let errors = Rc::new(std::cell::Cell::new(0usize));
    let stops = Rc::new(std::cell::Cell::new(0usize));
    let (finish_restarted, restarted_completion) = futures::channel::oneshot::channel::<()>();
    let restarted_completion = Rc::new(RefCell::new(Some(restarted_completion)));
    let callback_ticks = ticks.clone();
    let callback_errors = errors.clone();
    let callback_stops = stops.clone();
    let callback_runtime = runtime.clone();
    let callback_completion = restarted_completion.clone();
    let invoker: Rc<crate::user_functions::FunctionInvoker> =
        Rc::new(move |function, args, _requested_outputs| {
            let handle = args[0].clone();
            let runtime = callback_runtime.clone();
            let ticks = callback_ticks.clone();
            let errors = callback_errors.clone();
            let stops = callback_stops.clone();
            let completion = callback_completion.clone();
            Box::pin(async move {
                match function {
                    1 => {
                        let count = ticks.get() + 1;
                        ticks.set(count);
                        if count == 1 {
                            super::timer_stop_builtin(Some(runtime.clone()), None, handle.clone())
                                .await?;
                            super::timer_start_builtin(Some(runtime), None, handle).await?;
                            return Err(crate::build_runtime_error("old generation failed")
                                .with_identifier("RunMat:TimerTest:OldGeneration")
                                .build());
                        }
                        let completion = completion
                            .borrow_mut()
                            .take()
                            .expect("one restarted callback completion");
                        completion.await.expect("release restarted callback");
                    }
                    2 => errors.set(errors.get() + 1),
                    3 => stops.set(stops.get() + 1),
                    _ => panic!("unexpected timer callback"),
                }
                Ok(crate::sequence::ValueSequence::empty())
            })
        });
    let scope = runtime.enter();
    let _guard = crate::user_functions::install_local_semantic_function_invoker(invoker);
    let _bound = install_timer_bound_forwarders();
    drop(scope);
    let timer = block_on(timer_builtin(vec![
        Value::String("TimerFcn".into()),
        timer_callback_value("tick", 1),
        Value::String("ErrorFcn".into()),
        timer_callback_value("failed", 2),
        Value::String("StopFcn".into()),
        timer_callback_value("done", 3),
    ]))
    .expect("timer");
    block_on(timer_start_builtin(timer.clone())).expect("first start");
    let error = block_on(timer_wait_builtin(timer.clone())).expect_err("old failure receipt");
    assert_eq!(error.identifier(), Some("RunMat:TimerTest:OldGeneration"));
    assert_eq!(
        errors.get(),
        0,
        "stale ErrorFcn must not address restarted timer"
    );
    let Value::HandleObject(handle) = timer.clone() else {
        panic!("timer handle");
    };
    assert_eq!(string_property(&handle, "Running").unwrap(), "on");
    finish_restarted
        .send(())
        .expect("restarted callback remains retained");
    block_on(timer_wait_builtin(timer)).expect("new generation");
    assert_eq!(ticks.get(), 2);
    assert_eq!(errors.get(), 0);
    assert_eq!(stops.get(), 2, "one StopFcn per generation");
    assert_eq!(string_property(&handle, "Running").unwrap(), "off");
}

#[test]
fn deleting_a_pending_timer_cancels_its_child_without_late_callback() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let calls = Arc::new(Mutex::new(0usize));
    let recorded = calls.clone();
    let _guard = install_timer_invoker(Some(Arc::new(
        move |_function, _args, _requested_outputs| {
            let recorded = recorded.clone();
            Box::pin(async move {
                *recorded.lock().unwrap() += 1;
                Ok(crate::sequence::ValueSequence::empty())
            })
        },
    )));
    let timer = block_on(timer_builtin(vec![
        Value::String("TimerFcn".into()),
        timer_callback_value("tick", 1),
        Value::String("StartDelay".into()),
        Value::Num(3_600.0),
    ]))
    .expect("timer");
    block_on(timer_start_builtin(timer.clone())).expect("start");
    let host = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test reactor");
    host.block_on(test_supervisor().drive_until(async {
        tokio::task::yield_now().await;
    }))
    .expect("poll timer sleep pending");
    let Value::HandleObject(handle) = timer.clone() else {
        panic!("timer handle");
    };
    assert_eq!(string_property(&handle, "Running").unwrap(), "on");
    block_on(timer_delete_builtin(timer)).expect("delete");
    assert!(!crate::is_handle_valid(&handle));
    assert_eq!(*calls.lock().unwrap(), 0);
    test_supervisor().shutdown();
    assert_eq!(*calls.lock().unwrap(), 0);
}

#[test]
fn failed_timer_callback_delivers_original_exception_then_stop_once() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let order = Arc::new(Mutex::new(Vec::<&'static str>::new()));
    let recorded = order.clone();
    let _guard = install_timer_invoker(Some(Arc::new(move |function, args, requested_outputs| {
        assert_eq!(requested_outputs, 0);
        let recorded = recorded.clone();
        let args = args.to_vec();
        Box::pin(async move {
            match function {
                1 => {
                    recorded.lock().unwrap().push("timer");
                    Err(crate::build_runtime_error("tick failed")
                        .with_identifier("RunMat:TimerTest:Tick")
                        .build())
                }
                2 => {
                    recorded.lock().unwrap().push("error");
                    let Value::Struct(event) = &args[1] else {
                        panic!("error event");
                    };
                    assert_eq!(
                        event.fields.get("Type"),
                        Some(&Value::String("ErrorFcn".into()))
                    );
                    let Some(Value::Struct(data)) = event.fields.get("Data") else {
                        panic!("error data");
                    };
                    let Some(Value::MException(exception)) = data.fields.get("Error") else {
                        panic!("exact exception payload");
                    };
                    assert_eq!(exception.identifier, "RunMat:TimerTest:Tick");
                    assert_eq!(exception.message, "tick failed");
                    Ok(crate::sequence::ValueSequence::empty())
                }
                3 => {
                    recorded.lock().unwrap().push("stop");
                    Ok(crate::sequence::ValueSequence::empty())
                }
                _ => panic!("unexpected timer callback"),
            }
        })
    })));
    let timer = block_on(timer_builtin(vec![
        Value::String("TimerFcn".into()),
        timer_callback_value("tick", 1),
        Value::String("ErrorFcn".into()),
        timer_callback_value("failed", 2),
        Value::String("StopFcn".into()),
        timer_callback_value("done", 3),
    ]))
    .expect("timer");
    block_on(timer_start_builtin(timer.clone())).expect("start");
    let error = block_on(timer_wait_builtin(timer.clone())).expect_err("callback error");
    assert_eq!(error.identifier(), Some("RunMat:TimerTest:Tick"));
    assert_eq!(&*order.lock().unwrap(), &["timer", "error", "stop"]);
    block_on(timer_stop_builtin(timer)).expect("idempotent stop");
    assert_eq!(&*order.lock().unwrap(), &["timer", "error", "stop"]);
}

#[test]
fn text_callback_uses_executor_source_port_and_missing_port_rejects_before_running() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let unsupported = block_on(timer_builtin(vec![
        Value::String("TimerFcn".into()),
        Value::String("answer = 42;".into()),
    ]))
    .expect("text timer");
    block_on(timer_start_builtin(unsupported.clone())).expect_err("missing source service");
    let Value::HandleObject(handle) = unsupported else {
        panic!("timer handle");
    };
    assert_eq!(string_property(&handle, "Running").unwrap(), "off");

    let supervisor = Rc::new(RuntimeLocalTaskSupervisor::new());
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let owner = runtime.local_identity();
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let ports = RuntimeServicePorts::default()
        .with_local_tasks(supervisor.submission_port(owner))
        .with_source_evaluation(Rc::new(RecordingSourceEvaluator(recorded.clone())));
    let runtime = runtime.with_service_ports(ports);
    set_test_session(runtime, supervisor);
    let timer = block_on(timer_builtin(vec![
        Value::String("TimerFcn".into()),
        Value::String("answer = 42;".into()),
    ]))
    .expect("text timer");
    block_on(timer_start_builtin(timer.clone())).expect("admit text callback");
    assert!(recorded.lock().unwrap().is_empty());
    block_on(timer_wait_builtin(timer)).expect("evaluate source callback");
    assert_eq!(&*recorded.lock().unwrap(), &["answer = 42;"]);
}

#[test]
fn timer_delete_invalidates_and_removes_from_find_results() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let timer = block_on(timer_builtin(vec![
        Value::String("TimerFcn".into()),
        Value::String("disp(1)".into()),
    ]))
    .expect("timer");
    let Value::HandleObject(handle) = timer.clone() else {
        panic!("expected timer handle");
    };
    block_on(timer_delete_builtin(timer)).expect("delete");
    assert!(!crate::is_handle_valid(&handle));
    let found = block_on(timerfindall_builtin(vec![])).expect("timerfindall");
    let Value::Cell(cell) = found else {
        panic!("expected empty cell");
    };
    assert_eq!((cell.rows, cell.cols), (1, 0));
}

#[test]
fn timer_handle_cannot_start_in_another_session_before_task_admission() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let timer = block_on(timer_builtin(vec![
        Value::String("TimerFcn".into()),
        timer_callback_value("tick", 1),
    ]))
    .expect("timer");
    let foreign = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let error = block_on(super::timer_start_builtin(
        Some(foreign),
        None,
        timer.clone(),
    ))
    .expect_err("foreign timer start");
    assert_eq!(error.identifier(), Some("RunMat:timer:InvalidHandle"));
    let Value::HandleObject(handle) = timer else {
        panic!("timer handle");
    };
    assert_eq!(string_property(&handle, "Running").unwrap(), "off");
}

#[test]
fn timer_rejects_invalid_property_values() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let err = block_on(timer_builtin(vec![
        Value::CharArray(CharArray::new_row("Period")),
        Value::Num(0.0),
    ]))
    .unwrap_err();
    assert_eq!(
        err.identifier().map(str::to_string),
        Some("RunMat:timer:InvalidProperty".to_string())
    );
}

#[test]
fn timer_numeric_scalar_reads_typed_integer_storage_exactly() {
    let tensor = Tensor::new_integer(IntegerStorage::U16(vec![2026]), vec![1, 1])
        .expect("typed timer scalar");

    assert_eq!(
        numeric_scalar(&Value::Tensor(tensor), "StartDelay").expect("numeric scalar"),
        2026.0
    );

    let wide = Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1])
        .expect("wide typed timer scalar");
    assert!(numeric_scalar(&Value::Tensor(wide), "StartDelay").is_err());
}

#[test]
fn timer_tasks_to_execute_preserves_typed_integer_storage_exactly() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let calls = Arc::new(Mutex::new(0usize));
    let invoker_calls = calls.clone();
    let _guard = install_timer_invoker(Some(Arc::new(
        move |_function, _args, _requested_outputs| {
            let invoker_calls = Arc::clone(&invoker_calls);
            Box::pin(async move {
                *invoker_calls.lock().unwrap() += 1;
                Ok(crate::sequence::ValueSequence::empty())
            })
        },
    )));

    let tasks = Tensor::new_integer(IntegerStorage::U16(vec![2]), vec![1, 1]).expect("typed tasks");
    let timer = block_on(timer_builtin(vec![
        Value::String("TimerFcn".into()),
        timer_callback_value("tick", 1),
        Value::String("ExecutionMode".into()),
        Value::String("fixedSpacing".into()),
        Value::String("TasksToExecute".into()),
        Value::Tensor(tasks),
        Value::String("Period".into()),
        Value::Num(0.002),
    ]))
    .expect("timer");
    let Value::HandleObject(handle) = timer.clone() else {
        panic!("expected timer handle");
    };
    assert_eq!(
        property(&handle, "TasksToExecute").unwrap(),
        Value::Int(IntValue::U16(2))
    );
    assert_eq!(numeric_property(&handle, "TasksToExecute").unwrap(), 2.0);

    let found = block_on(timerfind_builtin(vec![
        Value::String("TasksToExecute".into()),
        Value::Num(2.0),
    ]))
    .expect("timerfind");
    assert_eq!(found, timer);

    block_on(timer_start_builtin(Value::HandleObject(handle))).expect("start");
    block_on(timer_wait_builtin(timer.clone())).expect("wait");
    assert_eq!(*calls.lock().unwrap(), 2);
}

#[test]
fn timer_tasks_to_execute_rejects_invalid_integer_bounds() {
    let negative =
        Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("negative tasks");
    assert!(parse_tasks_to_execute_value(&Value::Tensor(negative)).is_err());

    let zero = Tensor::new_integer(IntegerStorage::U16(vec![0]), vec![1, 1]).expect("zero tasks");
    assert!(parse_tasks_to_execute_value(&Value::Tensor(zero)).is_err());

    let boundary = if usize::BITS == 64 {
        usize::MAX as f64
    } else {
        (usize::MAX as f64) + 1.0
    };
    assert!(parse_tasks_to_execute_value(&Value::Num(boundary)).is_err());
}

#[test]
fn timerfind_compares_signed_and_wide_integer_user_data_exactly() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let negative = block_on(timer_builtin(vec![
        Value::String("UserData".into()),
        Value::Int(IntValue::I64(-7)),
    ]))
    .expect("negative integer user data");
    let wide = block_on(timer_builtin(vec![
        Value::String("UserData".into()),
        Value::Int(IntValue::U64((1_u64 << 53) + 1)),
    ]))
    .expect("wide integer user data");

    assert_eq!(
        block_on(timerfind_builtin(vec![
            Value::String("UserData".into()),
            Value::Num(-7.0),
        ]))
        .expect("signed filter"),
        negative
    );
    assert_eq!(
        block_on(timerfind_builtin(vec![
            Value::String("UserData".into()),
            Value::Int(IntValue::U64((1_u64 << 53) + 1)),
        ]))
        .expect("wide exact filter"),
        wide
    );
    let no_match = block_on(timerfind_builtin(vec![
        Value::String("UserData".into()),
        Value::Num(((1_u64 << 53) + 1) as f64),
    ]))
    .expect("rounded double filter");
    assert!(matches!(no_match, Value::Cell(cell) if cell.data.is_empty()));
}

#[test]
fn timer_dependent_setters_validate_post_construction_values() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let timer = block_on(timer_builtin(vec![])).expect("timer");
    let Value::HandleObject(handle) = timer else {
        panic!("expected timer handle");
    };
    let object = runmat_gc::gc_clone_value(&handle.target).expect("timer object");
    let err = block_on(set_timer_property_object(
        object.clone(),
        "Period",
        Value::Num(0.0),
    ))
    .unwrap_err();
    assert_eq!(
        err.identifier().map(str::to_string),
        Some("RunMat:timer:InvalidProperty".to_string())
    );

    let updated = block_on(set_timer_property_object(
        object,
        "Tag",
        Value::CharArray(CharArray::new_row("batch")),
    ))
    .expect("valid tag update");
    let Value::Object(updated) = updated else {
        panic!("expected updated timer object");
    };
    assert_eq!(
        updated.properties.get("Tag"),
        Some(&Value::String("batch".to_string()))
    );
}
