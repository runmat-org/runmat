use super::*;

#[test]
fn public_print_keeps_selected_snapshot_across_pending_stage_and_scene_clear() {
    let _plot = crate::builtins::plotting::tests::lock_plot_registry();
    crate::builtins::plotting::tests::ensure_plot_test_env();
    super::super::super::state::reset_hold_state_for_run();
    let _ = super::super::super::state::clear_figure(None);
    let _fs = fs::provider_override_lock();
    futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
        Value::Tensor(runmat_value::Tensor::new(vec![0.0, 1.0, 2.0], vec![3]).unwrap()),
        Value::Tensor(runmat_value::Tensor::new(vec![1.0, 3.0, 2.0], vec![3]).unwrap()),
    ]))
    .expect("plot selected scene");
    let original = clone_figure(current_figure_handle()).expect("original scene");
    let expected =
        futures::executor::block_on(render_png(original, 800, 600)).expect("reference render");
    let (sender, receiver) = oneshot::channel();
    let (start_sender, start_receiver) = oneshot::channel();
    let published = Arc::new(Mutex::new(None));
    let drops = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(PausedProvider {
        gate: Mutex::new(Some(receiver)),
        started: Mutex::new(Some(start_sender)),
        published: published.clone(),
        drops: drops.clone(),
        lease: RevocableLease::live(),
    });
    let _selected = fs::replace_provider(provider.clone());
    let runtime = selected_runtime();
    let _ambient_output =
        crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let arguments = [Value::from("/selected.png")];
    let mut call = Box::pin(dispatch_print(&runtime, &arguments));
    await_operation_start(start_receiver, call.as_mut());
    assert!(
        provider.gate.lock().unwrap().is_none(),
        "public call reached selected stage"
    );
    super::super::super::state::clear_figure(None)
        .expect("clear ambient scene during pending call");
    sender.send(()).expect("retained stage receiver");
    assert!(matches!(
        futures::executor::block_on(call).expect("public print completes").value(),
        Value::OutputList(values) if values.is_empty()
    ));
    assert_eq!(
        published.lock().unwrap().as_deref(),
        Some(expected.as_slice())
    );
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn cancelled_public_print_drops_pending_stage_without_publication_or_replay() {
    let _plot = crate::builtins::plotting::tests::lock_plot_registry();
    crate::builtins::plotting::tests::ensure_plot_test_env();
    super::super::super::state::reset_hold_state_for_run();
    let _ = super::super::super::state::clear_figure(None);
    let _fs = fs::provider_override_lock();
    futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
        Value::Tensor(runmat_value::Tensor::new(vec![0.0, 1.0], vec![2]).unwrap()),
        Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![2]).unwrap()),
    ]))
    .expect("plot selected scene");
    let (sender, receiver) = oneshot::channel();
    let (start_sender, start_receiver) = oneshot::channel();
    let published = Arc::new(Mutex::new(None));
    let drops = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(PausedProvider {
        gate: Mutex::new(Some(receiver)),
        started: Mutex::new(Some(start_sender)),
        published: published.clone(),
        drops: drops.clone(),
        lease: RevocableLease::live(),
    });
    let _selected = fs::replace_provider(provider.clone());
    let source = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    runtime.set_language_mode(crate::context::RuntimeLanguageMode::RunMat);
    runtime.set_runmat_extensions_enabled(true);
    let _ambient_output =
        crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let arguments = [Value::from("/cancelled.png")];
    let mut call = Box::pin(dispatch_print(&runtime, &arguments));
    await_operation_start(start_receiver, call.as_mut());
    assert!(
        provider.gate.lock().unwrap().is_none(),
        "public call reached selected stage"
    );
    source.cancel(runmat_execution::CancellationReason::User);
    let error = futures::executor::block_on(call).expect_err("pending print cancelled");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(matches!(
        std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<crate::execution::ExecutionServiceError>()),
        Some(crate::execution::ExecutionServiceError::Cancelled)
    ));
    assert!(
        sender.send(()).is_err(),
        "dropped stage must drop pending read owner"
    );
    assert!(published.lock().unwrap().is_none());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn revoked_selected_print_lease_after_pending_write_prevents_publication() {
    let _plot = crate::builtins::plotting::tests::lock_plot_registry();
    crate::builtins::plotting::tests::ensure_plot_test_env();
    super::super::super::state::reset_hold_state_for_run();
    let _ = super::super::super::state::clear_figure(None);
    let _fs = fs::provider_override_lock();
    futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
        Value::Tensor(runmat_value::Tensor::new(vec![0.0, 1.0], vec![2]).unwrap()),
        Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![2]).unwrap()),
    ]))
    .expect("plot selected scene");
    let (sender, receiver) = oneshot::channel();
    let (start_sender, start_receiver) = oneshot::channel();
    let published = Arc::new(Mutex::new(None));
    let drops = Arc::new(AtomicUsize::new(0));
    let lease = RevocableLease::live();
    let provider = Arc::new(PausedProvider {
        gate: Mutex::new(Some(receiver)),
        started: Mutex::new(Some(start_sender)),
        published: published.clone(),
        drops: drops.clone(),
        lease: lease.clone(),
    });
    let _selected = fs::replace_provider(provider.clone());
    let runtime = selected_runtime();
    let _ambient_output =
        crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let arguments = [Value::from("/revoked.png")];
    let mut call = Box::pin(dispatch_print(&runtime, &arguments));
    await_operation_start(start_receiver, call.as_mut());
    assert!(
        provider.gate.lock().unwrap().is_none(),
        "selected stage owns pending write"
    );
    lease.revoke();
    sender.send(()).expect("resume retained write");
    let error = futures::executor::block_on(call).expect_err("revoked selected lease");
    assert_eq!(error.identifier(), super::PRINT_ERROR_IO.identifier);
    assert!(error.message().contains("revoked"));
    assert!(published.lock().unwrap().is_none());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
