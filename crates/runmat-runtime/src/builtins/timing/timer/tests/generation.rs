use super::*;
#[test]
fn reentrant_stop_and_restart_cannot_let_stale_generation_turn_running_off() {
    let _lock = TIMER_TEST_LOCK.lock().unwrap();
    reset_timer_state_for_tests();
    let runtime = test_runtime();
    let calls = Rc::new(std::cell::Cell::new(0usize));
    let callback_calls = calls.clone();
    let callback_runtime = runtime.clone();
    let invoker: Rc<crate::user_functions::FunctionInvoker> =
        Rc::new(move |_runtime, _function, args, requested_outputs| {
            assert_eq!(requested_outputs, 0);
            let handle = args[0].clone();
            let calls = callback_calls.clone();
            let runtime = callback_runtime.clone();
            crate::user_functions::admit_user_function_with(_runtime.as_ref(), move || async move {
                let count = calls.get() + 1;
                calls.set(count);
                if count == 1 {
                    super::super::timer_stop_builtin(Some(runtime.clone()), None, handle.clone())
                        .await?;
                    super::super::timer_start_builtin(Some(runtime), None, handle).await?;
                }
                rooted_empty_sequence()
            })
        });
    let scope = runtime.enter();
    let _guard = crate::user_functions::install_local_semantic_function_invoker(invoker)
        .expect("call-port scope");
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
        Rc::new(move |_runtime, function, args, _requested_outputs| {
            let handle = args[0].clone();
            let runtime = callback_runtime.clone();
            let ticks = callback_ticks.clone();
            let errors = callback_errors.clone();
            let stops = callback_stops.clone();
            let completion = callback_completion.clone();
            crate::user_functions::admit_user_function_with(_runtime.as_ref(), move || async move {
                match function {
                    1 => {
                        let count = ticks.get() + 1;
                        ticks.set(count);
                        if count == 1 {
                            super::super::timer_stop_builtin(
                                Some(runtime.clone()),
                                None,
                                handle.clone(),
                            )
                            .await?;
                            super::super::timer_start_builtin(Some(runtime), None, handle).await?;
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
                rooted_empty_sequence()
            })
        });
    let scope = runtime.enter();
    let _guard = crate::user_functions::install_local_semantic_function_invoker(invoker)
        .expect("call-port scope");
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
                rooted_empty_sequence()
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
                    rooted_empty_sequence()
                }
                3 => {
                    recorded.lock().unwrap().push("stop");
                    rooted_empty_sequence()
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
