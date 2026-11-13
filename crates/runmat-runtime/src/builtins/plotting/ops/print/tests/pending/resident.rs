use super::*;

#[test]
fn resident_handle_selects_scene_after_pending_gather_then_retains_it_through_export() {
    let _accelerator = crate::builtins::common::test_support::accel_test_lock();
    let _plot = crate::builtins::plotting::tests::lock_plot_registry();
    crate::builtins::plotting::tests::ensure_plot_test_env();
    super::super::super::state::reset_hold_state_for_run();
    let _ = super::super::super::state::clear_figure(None);
    let _fs = fs::provider_override_lock();
    futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
        Value::Tensor(runmat_value::Tensor::new(vec![0.0, 1.0, 2.0], vec![3]).unwrap()),
        Value::Tensor(runmat_value::Tensor::new(vec![1.0, 3.0, 2.0], vec![3]).unwrap()),
    ]))
    .expect("initial figure");
    let figure = current_figure_handle();
    let original = clone_figure(figure).expect("initial scene");
    let original_png =
        futures::executor::block_on(render_png(original, 800, 600)).expect("initial render");

    let (download_sender, download_receiver) = oneshot::channel();
    let (download_started_sender, download_started_receiver) = oneshot::channel();
    let accelerator = Box::leak(Box::new(PausedDownloadProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        gate: Mutex::new(Some(download_receiver)),
        started: Mutex::new(Some(download_started_sender)),
        downloads: AtomicUsize::new(0),
    }));
    let _selected_accelerator = ThreadProviderGuard::set(Some(accelerator));
    let handle = accelerator
        .upload(&HostTensorView {
            data: &[f64::from(figure.as_u32())],
            shape: &[1, 1],
        })
        .expect("real resident scalar handle");

    let (stage_sender, stage_receiver) = oneshot::channel();
    let (stage_started_sender, stage_started_receiver) = oneshot::channel();
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
    let arguments = [Value::GpuTensor(handle), Value::from("/resident.png")];
    let mut call = Box::pin(dispatch_print(&runtime, &arguments));
    await_operation_start(download_started_receiver, call.as_mut());
    assert_eq!(accelerator.downloads.load(Ordering::SeqCst), 1);
    assert!(
        provider.gate.lock().unwrap().is_some(),
        "no filesystem stage may begin before resident handle materialization"
    );
    let other_filesystem = Arc::new(fs::MemoryFsProvider::with_current_dir("/"));
    let _replacement_filesystem = fs::replace_provider(other_filesystem.clone());

    super::super::super::state::clear_figure(Some(figure))
        .expect("mutate selected figure during gather");
    futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
        Value::Tensor(runmat_value::Tensor::new(vec![0.0, 1.0, 2.0], vec![3]).unwrap()),
        Value::Tensor(runmat_value::Tensor::new(vec![10.0, 20.0, 30.0], vec![3]).unwrap()),
    ]))
    .expect("replace selected scene");
    let updated = clone_figure(figure).expect("updated scene");
    let updated_png =
        futures::executor::block_on(render_png(updated, 800, 600)).expect("updated render");
    assert_ne!(original_png, updated_png, "fixture must change the scene");

    download_sender.send(()).expect("retained gather receiver");
    await_operation_start(stage_started_receiver, call.as_mut());
    assert_eq!(accelerator.downloads.load(Ordering::SeqCst), 1);
    super::super::super::state::clear_figure(Some(figure))
        .expect("clear ambient figure after resolved scene was captured");
    stage_sender.send(()).expect("retained stage receiver");
    assert!(matches!(
        futures::executor::block_on(call).expect("public resident print").value(),
        Value::OutputList(values) if values.is_empty()
    ));
    assert_eq!(
        published.lock().unwrap().as_deref(),
        Some(updated_png.as_slice())
    );
    assert!(other_filesystem.read_project_path("/resident.png").is_err());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn wide_resident_integer_handle_uses_one_authenticated_download() {
    let _accelerator = crate::builtins::common::test_support::accel_test_lock();
    let _plot = crate::builtins::plotting::tests::lock_plot_registry();
    crate::builtins::plotting::tests::ensure_plot_test_env();
    super::super::super::state::reset_hold_state_for_run();
    let _ = super::super::super::state::clear_figure(None);
    let _fs = fs::provider_override_lock();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
        Value::Tensor(runmat_value::Tensor::new(vec![0.0, 1.0], vec![2]).unwrap()),
        Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![2]).unwrap()),
    ]))
    .expect("selected figure");
    let figure = current_figure_handle();
    let expected = futures::executor::block_on(render_png(
        clone_figure(figure).expect("selected scene"),
        800,
        600,
    ))
    .expect("reference render");

    let (download_sender, download_receiver) = oneshot::channel();
    let (download_started_sender, download_started_receiver) = oneshot::channel();
    let accelerator = Box::leak(Box::new(PausedDownloadProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        gate: Mutex::new(Some(download_receiver)),
        started: Mutex::new(Some(download_started_sender)),
        downloads: AtomicUsize::new(0),
    }));
    let _selected_accelerator = ThreadProviderGuard::set(Some(accelerator));
    let number = [u64::from(figure.as_u32())];
    let handle = accelerator
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::U64(&number),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .expect("real U64 resident handle");

    let (stage_sender, stage_receiver) = oneshot::channel();
    let (stage_started_sender, stage_started_receiver) = oneshot::channel();
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
    let arguments = [Value::GpuTensor(handle), Value::from("/integer.png")];
    let mut call = Box::pin(dispatch_print(&runtime, &arguments));
    await_operation_start(download_started_receiver, call.as_mut());
    assert_eq!(accelerator.downloads.load(Ordering::SeqCst), 1);
    assert!(provider.gate.lock().unwrap().is_some());
    download_sender.send(()).expect("retained integer download");
    await_operation_start(stage_started_receiver, call.as_mut());
    assert_eq!(
        accelerator.downloads.load(Ordering::SeqCst),
        1,
        "the prepared integer payload must not be downloaded again"
    );
    stage_sender.send(()).expect("retained stage receiver");
    assert!(matches!(
        futures::executor::block_on(call).expect("resident integer print").value(),
        Value::OutputList(values) if values.is_empty()
    ));
    assert_eq!(
        published.lock().unwrap().as_deref(),
        Some(expected.as_slice())
    );
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn cancelled_pending_resident_gather_has_no_stage_or_download_replay() {
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
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    );
    runtime.set_language_mode(crate::context::RuntimeLanguageMode::RunMat);
    runtime.set_runmat_extensions_enabled(true);
    let _ambient_output =
        crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let arguments = [
        Value::GpuTensor(handle),
        Value::from("/cancelled-gather.png"),
    ];
    let mut call = Box::pin(dispatch_print(&runtime, &arguments));
    await_operation_start(download_started_receiver, call.as_mut());
    assert_eq!(accelerator.downloads.load(Ordering::SeqCst), 1);
    cancellation.cancel(runmat_execution::CancellationReason::User);
    let error = futures::executor::block_on(call).expect_err("cancelled resident gather");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(matches!(
        std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<crate::execution::ExecutionServiceError>()),
        Some(crate::execution::ExecutionServiceError::Cancelled)
    ));
    assert!(
        download_sender.send(()).is_err(),
        "download future must be dropped"
    );
    assert_eq!(accelerator.downloads.load(Ordering::SeqCst), 1);
    assert!(provider.gate.lock().unwrap().is_some(), "no stage began");
    assert!(published.lock().unwrap().is_none());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
}
