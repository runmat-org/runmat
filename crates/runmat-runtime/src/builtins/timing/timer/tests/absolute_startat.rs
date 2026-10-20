use super::*;

async fn timer_startat_with_args(
    runtime: Option<RuntimeContext>,
    requested_outputs: Option<usize>,
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let args = crate::call::BuiltinArguments::from_borrowed(&rest, runtime.as_ref())?;
    super::super::timer_startat_builtin(runtime, requested_outputs, value, args).await
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn absolute_startat_matches_exact_date_shape_before_lifecycle_effects() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests_with_clock(Some(Arc::new(Mutex::new(Duration::ZERO))));
    let timers = (0..2)
        .map(|_| {
            block_on(timer_builtin(vec![
                Value::String("TimerFcn".into()),
                Value::FunctionHandle("tick".into()),
            ]))
            .expect("timer")
        })
        .collect::<Vec<_>>();
    let handles = timers
        .iter()
        .map(|timer| match timer {
            Value::HandleObject(handle) => handle.clone(),
            _ => panic!("timer handle expected"),
        })
        .collect();
    let receiver = Value::ObjectArray(
        ObjectArray::from_handles(TIMER_CLASS.to_string(), handles, vec![1, 2]).unwrap(),
    );
    let now = chrono::DateTime::from_timestamp(1_700_000_000, 0)
        .unwrap()
        .naive_utc();
    let dates = (1..=2)
        .map(|seconds| {
            (now + chrono::Duration::seconds(seconds))
                .format("%Y-%m-%d %H:%M:%S")
                .to_string()
        })
        .collect::<Vec<_>>();
    let mismatch = Value::StringArray(StringArray::new(dates.clone(), vec![2]).unwrap());
    let error = block_on(timer_startat_with_args(
        Some(test_runtime()),
        Some(0),
        receiver.clone(),
        vec![mismatch],
    ))
    .expect_err("equal element counts do not replace exact firing-time shape");
    assert_eq!(error.identifier(), Some("RunMat:timer:InvalidInput"));
    for timer in &timers {
        let Value::HandleObject(handle) = timer else {
            panic!("timer handle expected")
        };
        assert_eq!(string_property(handle, "Running").unwrap(), "off");
    }
    let matching = Value::StringArray(StringArray::new(dates, vec![1, 2]).unwrap());
    block_on(timer_startat_with_args(
        Some(test_runtime()),
        Some(0),
        receiver.clone(),
        vec![matching],
    ))
    .expect("matching row-vector firing times");
    for timer in &timers {
        let Value::HandleObject(handle) = timer else {
            panic!("timer handle expected")
        };
        assert_eq!(string_property(handle, "Running").unwrap(), "on");
    }
    block_on(timer_stop_builtin(receiver)).expect("stop timers");
}

#[test]
fn timer_method_status_and_relative_startat_are_gated_before_lifecycle_effects() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let timer = block_on(timer_builtin(Vec::new())).expect("timer");
    let runtime = test_runtime();
    let assigned = block_on(super::super::timer_start_builtin(
        Some(runtime.clone()),
        Some(1),
        timer.clone(),
    ))
    .expect_err("assigned start status is a RunMat extension");
    assert_eq!(
        assigned.identifier(),
        Some("RunMat:compatibility:TimerMethodStatusOutputExtension")
    );
    let relative = block_on(timer_startat_with_args(
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
        block_on(super::super::timer_stop_builtin(
            Some(runtime),
            Some(1),
            timer
        )),
        Ok(Value::Num(0.0))
    ));
}

#[test]
fn absolute_startat_uses_injected_wall_clock_without_relative_extension() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests_with_clock(Some(Arc::new(Mutex::new(Duration::ZERO))));
    let timer = block_on(timer_builtin(vec![
        Value::String("TimerFcn".into()),
        Value::FunctionHandle("tick".into()),
    ]))
    .expect("timer");
    let future = chrono::DateTime::from_timestamp(1_700_000_002, 0)
        .unwrap()
        .naive_utc()
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();
    block_on(timer_startat_with_args(
        Some(test_runtime()),
        Some(0),
        timer.clone(),
        vec![Value::String(future.into())],
    ))
    .expect("absolute date text starts without the RunMat extension");
    let Value::HandleObject(handle) = &timer else {
        panic!("expected timer handle")
    };
    assert_eq!(string_property(handle, "Running").unwrap(), "on");
    block_on(timer_stop_builtin(timer)).expect("stop");
}

#[test]
fn absolute_startat_rejects_past_window_and_mismatched_array_before_start() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests_with_clock(Some(Arc::new(Mutex::new(Duration::ZERO))));
    let timer = block_on(timer_builtin(vec![
        Value::String("TimerFcn".into()),
        Value::FunctionHandle("tick".into()),
    ]))
    .expect("timer");
    let now = chrono::DateTime::from_timestamp(1_700_000_000, 0)
        .unwrap()
        .naive_utc();
    for instant in [
        now - chrono::Duration::seconds(1),
        now + chrono::Duration::days(26),
    ] {
        let date = instant.format("%Y-%m-%d %H:%M:%S").to_string();
        let error = block_on(timer_startat_with_args(
            Some(test_runtime()),
            Some(0),
            timer.clone(),
            vec![Value::String(date.into())],
        ))
        .expect_err("invalid absolute date");
        assert_eq!(error.identifier(), Some("RunMat:timer:InvalidInput"));
    }
    let dates = Value::StringArray(
        runmat_value::StringArray::new(
            vec![
                (now + chrono::Duration::seconds(1))
                    .format("%Y-%m-%d %H:%M:%S")
                    .to_string(),
                (now + chrono::Duration::seconds(2))
                    .format("%Y-%m-%d %H:%M:%S")
                    .to_string(),
            ],
            vec![1, 2],
        )
        .unwrap(),
    );
    let error = block_on(timer_startat_with_args(
        Some(test_runtime()),
        Some(0),
        timer.clone(),
        vec![dates],
    ))
    .expect_err("firing array cannot address a scalar timer");
    assert_eq!(error.identifier(), Some("RunMat:timer:InvalidInput"));
    let Value::HandleObject(handle) = timer else {
        panic!("expected timer handle")
    };
    assert_eq!(string_property(&handle, "Running").unwrap(), "off");
}

#[test]
fn startat_uses_column_major_timer_arrays_and_accepts_nd_object_arrays() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests_with_clock(Some(Arc::new(Mutex::new(Duration::ZERO))));
    let timers = (0..4)
        .map(|_| {
            block_on(timer_builtin(vec![
                Value::String("TimerFcn".into()),
                Value::FunctionHandle("tick".into()),
            ]))
            .expect("timer")
        })
        .collect::<Vec<_>>();
    let cell = Value::Cell(CellArray::new(timers.clone(), 2, 2).unwrap());
    let (ordered, shape) = super::startat::receiver_handles(&test_runtime(), &cell).unwrap();
    let expected = [0, 2, 1, 3]
        .map(|index| match &timers[index] {
            Value::HandleObject(handle) => handle.clone(),
            _ => panic!("timer handle"),
        })
        .to_vec();
    assert_eq!(shape, vec![2, 2]);
    assert_eq!(ordered, expected);

    let object_array = Value::ObjectArray(
        ObjectArray::from_handles(TIMER_CLASS.to_string(), expected, vec![2, 1, 2]).unwrap(),
    );
    let now = chrono::DateTime::from_timestamp(1_700_000_000, 0)
        .unwrap()
        .naive_utc();
    let dates = Value::StringArray(
        StringArray::new(
            (1..=4)
                .map(|seconds| {
                    (now + chrono::Duration::seconds(seconds))
                        .format("%Y-%m-%d %H:%M:%S")
                        .to_string()
                })
                .collect(),
            vec![2, 1, 2],
        )
        .unwrap(),
    );
    block_on(timer_startat_with_args(
        Some(test_runtime()),
        Some(0),
        object_array.clone(),
        vec![dates],
    ))
    .expect("N-D object array of timers");
    for timer in &timers {
        let Value::HandleObject(handle) = timer else {
            panic!("timer handle")
        };
        assert_eq!(string_property(handle, "Running").unwrap(), "on");
    }
    block_on(timer_stop_builtin(object_array)).expect("stop object array");
}

#[test]
fn startat_empty_object_array_preserves_shape_and_has_no_lifecycle_effect() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests_with_clock(Some(Arc::new(Mutex::new(Duration::ZERO))));
    let array =
        Value::ObjectArray(ObjectArray::empty(TIMER_CLASS.to_string(), vec![0, 2, 1]).unwrap());
    let empty_dates = Value::StringArray(StringArray::new(Vec::new(), vec![0, 2, 1]).unwrap());
    block_on(timer_startat_with_args(
        Some(test_runtime()),
        Some(0),
        array,
        vec![empty_dates],
    ))
    .expect("empty matching arrays");
    assert!(test_runtime().timer_session().borrow().handles().is_empty());
}
