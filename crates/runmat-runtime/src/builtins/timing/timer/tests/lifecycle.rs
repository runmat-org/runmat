use super::*;
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
                rooted_empty_sequence()
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
                rooted_empty_sequence()
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
                rooted_empty_sequence()
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
                    rooted_empty_sequence()
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
                    rooted_empty_sequence()
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
