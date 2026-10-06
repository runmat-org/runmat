use super::*;
use crate::context::{
    RuntimeLocalTaskSupervisor, RuntimeServiceFuture, RuntimeServicePorts, RuntimeTimingService,
};
use crate::execution::RuntimeExecutionService;
use futures::executor::block_on;
use runmat_value::{CharArray, IntegerStorage, Tensor};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

struct TestClock(Arc<Mutex<Duration>>);

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

thread_local! {
    static TEST_SESSION: RefCell<Option<(RuntimeContext, Rc<RuntimeLocalTaskSupervisor>)>> = const { RefCell::new(None) };
}

fn reset_timer_state_for_tests() {
    reset_timer_state_for_tests_with_clock(None);
}

fn reset_timer_state_for_tests_with_clock(clock: Option<Arc<Mutex<Duration>>>) {
    TEST_SESSION.with(|session| {
        let supervisor = Rc::new(RuntimeLocalTaskSupervisor::new());
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let owner = runtime.local_identity();
        let mut ports =
            RuntimeServicePorts::default().with_local_tasks(supervisor.submission_port(owner));
        if let Some(clock) = clock {
            ports = ports.with_timing(Rc::new(TestClock(clock)));
        }
        let runtime = runtime.with_service_ports(ports);
        *session.borrow_mut() = Some((runtime, supervisor));
    });
}

fn test_runtime() -> RuntimeContext {
    TEST_SESSION.with(|session| session.borrow().as_ref().expect("test session").0.clone())
}

fn test_supervisor() -> Rc<RuntimeLocalTaskSupervisor> {
    TEST_SESSION.with(|session| session.borrow().as_ref().expect("test session").1.clone())
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
    super::timer_start_builtin(Some(test_runtime()), value).await
}

async fn timer_wait_builtin(value: Value) -> BuiltinResult<Value> {
    let runtime = test_runtime();
    test_supervisor()
        .drive_until(super::timer_wait_builtin(Some(runtime), value))
        .await?
}

async fn timer_delete_builtin(value: Value) -> BuiltinResult<Value> {
    super::timer_delete_builtin(Some(test_runtime()), value).await
}

async fn timer_stop_builtin(value: Value) -> BuiltinResult<Value> {
    super::timer_stop_builtin(Some(test_runtime()), value).await
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
        let builtin = runmat_builtins::builtin_function_by_name(name).unwrap();
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
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |_function, args, requested_outputs| {
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
        },
    )));
    let timer = block_on(timer_builtin(vec![
        Value::String("TimerFcn".into()),
        Value::BoundFunctionHandle {
            name: "tick".into(),
            function: 1,
        },
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
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
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
        Value::BoundFunctionHandle {
            name: "tick".into(),
            function: 1,
        },
        Value::String("StopFcn".into()),
        Value::BoundFunctionHandle {
            name: "done".into(),
            function: 2,
        },
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
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
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
        Value::BoundFunctionHandle {
            name: "tick".into(),
            function: 1,
        },
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
        Value::BoundFunctionHandle {
            name: "tick".into(),
            function: 1,
        },
    ]))
    .expect("timer");
    let foreign = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let error = block_on(super::timer_start_builtin(Some(foreign), timer.clone()))
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
    let _guard = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
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
        Value::BoundFunctionHandle {
            name: "tick".into(),
            function: 1,
        },
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
