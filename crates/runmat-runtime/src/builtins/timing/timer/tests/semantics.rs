use super::super::setter::set_timer_property_object;
use super::*;
use crate::workspace::{BaseWorkspaceOwner, BaseWorkspaceService};

#[test]
fn text_callback_uses_executor_source_port_and_missing_port_rejects_before_running() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let missing_runtime = test_runtime();
    let missing_ports = missing_runtime
        .service_ports()
        .clone()
        .with_base_workspace(Rc::new(
            BaseWorkspaceService::new(BaseWorkspaceOwner::new()),
        ));
    set_test_session(
        missing_runtime.with_service_ports(missing_ports),
        test_supervisor(),
    );
    let unsupported = block_on(timer_builtin(vec![
        Value::String("TimerFcn".into()),
        Value::String("answer = 42;".into()),
    ]))
    .expect("text timer");
    let missing_source =
        block_on(timer_start_builtin(unsupported.clone())).expect_err("missing source service");
    assert_eq!(
        missing_source.identifier(),
        Some(crate::context::RuntimeCapabilityError::IDENTIFIER)
    );
    assert!(missing_source.message().contains("source_evaluation"));
    let Value::HandleObject(handle) = unsupported else {
        panic!("timer handle");
    };
    assert_eq!(string_property(&handle, "Running").unwrap(), "off");
    assert_eq!(numeric_property(&handle, "TasksExecuted").unwrap(), 0.0);

    let supervisor = Rc::new(RuntimeLocalTaskSupervisor::new());
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let owner = runtime.local_identity();
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let ports = RuntimeServicePorts::default()
        .with_local_tasks(supervisor.submission_port(owner))
        .with_base_workspace(Rc::new(
            BaseWorkspaceService::new(BaseWorkspaceOwner::new()),
        ))
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
    let error = block_on(super::super::timer_start_builtin(
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
                crate::execution::RootedValueSequence::new(crate::sequence::ValueSequence::empty())
                    .map_err(|error| error.into_runtime_error())
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
    assert!(matches!(no_match, Value::Cell(cell) if cell.data().is_empty()));
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
        None,
        object.clone(),
        TimerProperty::Period,
        Value::Num(0.0),
    ))
    .unwrap_err();
    assert_eq!(
        err.identifier().map(str::to_string),
        Some("RunMat:timer:InvalidProperty".to_string())
    );

    let updated = block_on(set_timer_property_object(
        None,
        object,
        TimerProperty::Tag,
        Value::CharArray(CharArray::new_row("batch")),
    ))
    .expect("valid tag update");
    let Value::Object(updated) = updated else {
        panic!("expected updated timer object");
    };
    assert_eq!(
        updated.properties().get("Tag"),
        Some(&Value::String("batch".to_string().into()))
    );
}
