use super::*;

#[test]
fn unpolled_public_print_drop_has_no_filesystem_or_provider_effects() {
    let _accelerator = crate::builtins::common::test_support::accel_test_lock();
    let _plot = crate::builtins::plotting::tests::lock_plot_registry();
    crate::builtins::plotting::tests::ensure_plot_test_env();
    super::super::super::state::reset_hold_state_for_run();
    let _ = super::super::super::state::clear_figure(None);
    let _fs = fs::provider_override_lock();

    let (_download_sender, download_receiver) = oneshot::channel();
    let (download_started_sender, _download_started_receiver) = oneshot::channel();
    let accelerator = Box::leak(Box::new(PausedDownloadProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        gate: Mutex::new(Some(download_receiver)),
        started: Mutex::new(Some(download_started_sender)),
        downloads: AtomicUsize::new(0),
    }));
    let _selected_accelerator = ThreadProviderGuard::set(Some(accelerator));
    let handle = accelerator
        .upload(&HostTensorView {
            data: &[1.0],
            shape: &[1, 1],
        })
        .expect("resident handle fixture");

    let (_stage_sender, stage_receiver) = oneshot::channel();
    let (stage_started_sender, _stage_started_receiver) = oneshot::channel();
    let published = Arc::new(Mutex::new(None));
    let drops = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(PausedProvider {
        gate: Mutex::new(Some(stage_receiver)),
        started: Mutex::new(Some(stage_started_sender)),
        published: published.clone(),
        drops: drops.clone(),
        lease: RevocableLease::live(),
    });
    let _selected_filesystem = fs::replace_provider(provider.clone());
    let runtime = selected_runtime();
    let arguments = [Value::GpuTensor(handle), Value::from("/unpolled.png")];

    let call = Box::pin(dispatch_print(&runtime, &arguments));
    drop(call);

    assert_eq!(accelerator.downloads.load(Ordering::SeqCst), 0);
    assert!(accelerator.gate.lock().unwrap().is_some());
    assert!(accelerator.started.lock().unwrap().is_some());
    assert!(provider.gate.lock().unwrap().is_some());
    assert!(provider.started.lock().unwrap().is_some());
    assert!(published.lock().unwrap().is_none());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
}

#[test]
fn dropping_public_print_at_pending_stage_retires_stage_without_publication_or_replay() {
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
    let runtime = selected_runtime();
    let _ambient_output =
        crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let arguments = [Value::from("/dropped.png")];
    let mut call = Box::pin(dispatch_print(&runtime, &arguments));
    await_operation_start(start_receiver, call.as_mut());
    assert!(provider.gate.lock().unwrap().is_none());

    drop(call);

    assert!(
        sender.send(()).is_err(),
        "dropping the public future must retire the pending stage"
    );
    assert!(published.lock().unwrap().is_none());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(provider.started.lock().unwrap().is_none());
}

#[test]
fn dropping_public_print_at_pending_gather_retires_input_without_stage_or_replay() {
    let _accelerator = crate::builtins::common::test_support::accel_test_lock();
    let _plot = crate::builtins::plotting::tests::lock_plot_registry();
    crate::builtins::plotting::tests::ensure_plot_test_env();
    super::super::super::state::reset_hold_state_for_run();
    let _ = super::super::super::state::clear_figure(None);
    let _fs = fs::provider_override_lock();
    futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
        Value::Tensor(runmat_value::Tensor::new(vec![0.0, 1.0], vec![2]).unwrap()),
        Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![2]).unwrap()),
    ]))
    .expect("selected figure");

    let (download_sender, download_receiver) = oneshot::channel();
    let (download_started_sender, download_started_receiver) = oneshot::channel();
    let accelerator = Box::leak(Box::new(PausedDownloadProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        gate: Mutex::new(Some(download_receiver)),
        started: Mutex::new(Some(download_started_sender)),
        downloads: AtomicUsize::new(0),
    }));
    let _selected_accelerator = ThreadProviderGuard::set(Some(accelerator));
    let figure_number = [f64::from(current_figure_handle().as_u32())];
    let handle = accelerator
        .upload(&HostTensorView {
            data: &figure_number,
            shape: &[1, 1],
        })
        .expect("real resident scalar handle");

    let (_stage_sender, stage_receiver) = oneshot::channel();
    let (stage_started_sender, _stage_started_receiver) = oneshot::channel();
    let published = Arc::new(Mutex::new(None));
    let drops = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(PausedProvider {
        gate: Mutex::new(Some(stage_receiver)),
        started: Mutex::new(Some(stage_started_sender)),
        published: published.clone(),
        drops: drops.clone(),
        lease: RevocableLease::live(),
    });
    let _selected_filesystem = fs::replace_provider(provider.clone());
    let runtime = selected_runtime();
    let _ambient_output =
        crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let arguments = [Value::GpuTensor(handle), Value::from("/dropped-gather.png")];
    let mut call = Box::pin(dispatch_print(&runtime, &arguments));
    await_operation_start(download_started_receiver, call.as_mut());
    assert_eq!(accelerator.downloads.load(Ordering::SeqCst), 1);

    drop(call);

    assert!(
        download_sender.send(()).is_err(),
        "dropping the public future must retire its selected gather"
    );
    assert_eq!(accelerator.downloads.load(Ordering::SeqCst), 1);
    assert!(provider.gate.lock().unwrap().is_some(), "no stage began");
    assert!(provider.started.lock().unwrap().is_some());
    assert!(published.lock().unwrap().is_none());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
}
