use super::*;
use crate::context::host_effect::{AudioDecodeOutput, AudioFrameRange, HostEffectWorkload};
use crate::context::{RuntimeContext, RuntimeServicePorts};
use futures::task::noop_waker_ref;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::{
    CancellationReason, ResourceBoundUnavailable, ResourceByteBound, WholeWorkObservationOutcome,
};
use std::future::Future;
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::task::{Context, Poll};

fn fixture() -> (Arc<CountedFs>, runmat_filesystem::ProviderGuard) {
    let fs = Arc::new(CountedFs::new("/audio"));
    fs.inner
        .write_project_path("/audio/tone.wav", &pcm16_wav(8_000, 1, &[0, 16_384]))
        .expect("fixture WAV");
    let guard = runmat_filesystem::replace_provider(fs.clone());
    (fs, guard)
}

fn known() -> Option<crate::context::host_effect::HostEffectResourceEvidence> {
    // Synthetic service evidence for selector/receipt protocol tests only.
    // This is not a production provider or codec peak-memory certificate.
    Some(evidence(
        ResourceByteBound::Known(4 * 1024),
        ResourceByteBound::Known(256 * 1024),
    ))
}

fn value() -> Value {
    Value::from("/audio/tone.wav")
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn finite_missing_and_unknown_evidence_reject_both_identities_before_file_effects() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (fs, _guard) = fixture();
    let provider: Arc<dyn runmat_filesystem::FsProvider> = fs.clone();
    let no_port = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_parallel(Rc::new(Budget::new(Some(1024))))
                .with_placement(Rc::new(Placement::default())),
        );
    for error in [
        block_on(audioread_builtin(
            Some(no_port.clone()),
            value(),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap_err(),
        block_on(audioinfo_builtin(Some(no_port), value())).unwrap_err(),
    ] {
        assert_eq!(
            error.identifier(),
            Some("RunMat:Placement:InsufficientResourceEvidence")
        );
    }
    assert_eq!(fs.effects(), (0, 0, 0));

    for bounds in [
        None,
        Some(evidence(
            ResourceByteBound::Known(256),
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoPeakMemoryBound),
        )),
        Some(evidence(
            ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound),
            ResourceByteBound::Known(512),
        )),
    ] {
        let (runtime, _, placement) = context(provider.clone(), bounds, Some(512 * 1024));
        let err = block_on(audioread_builtin(
            Some(runtime.clone()),
            value(),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("finite decode requires complete evidence");
        assert_eq!(
            err.identifier(),
            Some("RunMat:Placement:InsufficientResourceEvidence")
        );
        let err = block_on(audioinfo_builtin(Some(runtime), value()))
            .expect_err("finite metadata requires complete evidence");
        assert_eq!(
            err.identifier(),
            Some("RunMat:Placement:InsufficientResourceEvidence")
        );
        assert_eq!(placement.selections.get(), 0);
    }
    assert_eq!(fs.effects(), (0, 0, 0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn insufficient_known_budget_and_invalid_controls_have_zero_file_effects() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (fs, _guard) = fixture();
    let provider: Arc<dyn runmat_filesystem::FsProvider> = fs.clone();
    let (runtime, resources, placement) = context(provider, known(), Some(260 * 1024 - 1));
    let err = block_on(audioread_builtin(
        Some(runtime.clone()),
        value(),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("known demand exceeds budget");
    assert_eq!(err.identifier(), Some("RunMat:Placement:ResourceLimit"));
    assert_eq!(fs.effects(), (0, 0, 0));

    let calls = resources.calls.get();
    let err = block_on(audioread_builtin(
        Some(runtime.clone()),
        Value::Num(3.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("invalid filename");
    assert_eq!(err.identifier(), Some("RunMat:audioread:InvalidArgument"));
    let err = block_on(audioread_builtin(
        Some(runtime.clone()),
        value(),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("int16")]),
    ))
    .expect_err("invalid datatype");
    assert_eq!(err.identifier(), Some("RunMat:audioread:InvalidArgument"));
    let err = block_on(audioinfo_builtin(Some(runtime.clone()), Value::Num(3.0)))
        .expect_err("invalid metadata filename");
    assert_eq!(err.identifier(), Some("RunMat:audioinfo:InvalidArgument"));
    {
        let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        assert!(block_on(audioread_builtin(
            Some(runtime.clone()),
            value(),
            crate::builtins::common::test_support::builtin_arguments(vec![])
        ))
        .is_err());
    }
    {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        assert!(block_on(audioinfo_builtin(Some(runtime), value())).is_err());
    }
    assert_eq!(resources.calls.get(), calls);
    assert_eq!(placement.selections.get(), 0);
    assert_eq!(fs.effects(), (0, 0, 0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn admitted_output_forms_and_metadata_finish_one_complete_call_each() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (fs, _guard) = fixture();
    let provider: Arc<dyn runmat_filesystem::FsProvider> = fs.clone();
    let (runtime, resources, placement) = context(provider, known(), Some(512 * 1024));
    let direct = block_on(audioread_builtin(
        Some(runtime.clone()),
        value(),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect("implicit samples");
    assert_eq!(tensor(direct).materialize_f64(), vec![0.0, 0.5]);
    {
        let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        assert!(output_list(
            block_on(audioread_builtin(
                Some(runtime.clone()),
                value(),
                crate::builtins::common::test_support::builtin_arguments(vec![])
            ))
            .expect("discarded output")
        )
        .is_empty());
    }
    {
        let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let result = output_list(
            block_on(audioread_builtin(
                Some(runtime.clone()),
                value(),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .expect("one output"),
        );
        assert_eq!(result.len(), 1);
        assert_eq!(tensor(result[0].clone()).materialize_f64(), vec![0.0, 0.5]);
    }
    {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let result = output_list(
            block_on(audioread_builtin(
                Some(runtime.clone()),
                value(),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .expect("two outputs"),
        );
        assert_eq!(result[1], Value::Num(8000.0));
    }
    let info = block_on(audioinfo_builtin(Some(runtime.clone()), value())).expect("metadata");
    assert_eq!(field(&info, "TotalSamples"), &Value::Num(2.0));
    {
        let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        let info =
            block_on(audioinfo_builtin(Some(runtime), value())).expect("zero-output metadata");
        assert!(output_list(info).is_empty());
    }
    assert_eq!(resources.calls.get(), 6);
    assert_eq!(placement.selections.get(), 6);
    assert_eq!(placement.observations.borrow().len(), 6);
    assert!(placement
        .observations
        .borrow()
        .iter()
        .all(|o| o.outcome == WholeWorkObservationOutcome::Success));
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    assert_eq!(fs.effects().0, 6);
    let demands = placement.demands.borrow();
    assert_eq!(demands[1].retained_bytes, ResourceByteBound::Known(0));
    assert_eq!(
        demands[1].scratch_bytes,
        ResourceByteBound::Known(256 * 1024)
    );
    assert_eq!(demands[5].retained_bytes, ResourceByteBound::Known(0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn relative_source_and_typed_range_mode_reach_exact_captured_owner() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (fs, _guard) = fixture();
    // Installing a provider preserves an existing virtual session directory.
    // Establish this test's relative-path base explicitly on every platform;
    // the provider guard restores the prior directory as well as its owner.
    runmat_filesystem::set_current_dir("/audio").expect("relative audio fixture directory");
    let provider: Arc<dyn runmat_filesystem::FsProvider> = fs.clone();
    let (runtime, resources, placement) = context(provider, known(), Some(512 * 1024));
    let range = Value::Tensor(Tensor::new(vec![2.0, f64::INFINITY], vec![1, 2]).expect("range"));
    let result = block_on(audioread_builtin(
        Some(runtime),
        Value::from("tone.wav"),
        crate::builtins::common::test_support::builtin_arguments(vec![
            range,
            Value::from("native"),
        ]),
    ))
    .expect("captured relative source");
    assert_eq!(
        tensor(result).integer_storage(),
        Some(&IntegerStorage::I16(vec![16_384]))
    );
    let source = std::path::Path::new("/audio").join("tone.wav");
    let expected = HostEffectWorkload::AudioDecode {
        source: &source,
        range: Some(AudioFrameRange {
            first: 2,
            last: None,
        }),
        datatype: AudioReadDataType::Native,
        output: AudioDecodeOutput::Samples,
    };
    assert_eq!(
        resources.workloads.borrow().as_slice(),
        &[expected.identity_bytes()]
    );
    assert_eq!(placement.observations.borrow().len(), 1);
    assert_eq!(fs.effects().0, 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn operational_error_cancel_and_pending_drop_retire_probe_without_replay() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (fs, _guard) = fixture();
    let provider: Arc<dyn runmat_filesystem::FsProvider> = fs.clone();
    let (runtime, resources, placement) = context(provider, known(), Some(512 * 1024));
    let err = block_on(audioread_builtin(
        Some(runtime.clone()),
        Value::from("/audio/missing.wav"),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("missing input");
    // The exact-resource test service declines unknown paths before file I/O.
    assert_eq!(
        err.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(placement.selections.get(), 0);

    fs.inner
        .write_project_path("/audio/tone.wav", b"bad audio")
        .expect("replace fixture with malformed bytes");
    let err = block_on(audioread_builtin(
        Some(runtime.clone()),
        value(),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("invalid source coding");
    assert_eq!(err.identifier(), Some("RunMat:audioread:UnsupportedFormat"));
    assert_eq!(placement.observations.borrow().len(), 1);
    assert_eq!(
        placement.observations.borrow()[0].outcome,
        WholeWorkObservationOutcome::OperationalFailure
    );
    let info_error = block_on(audioinfo_builtin(Some(runtime.clone()), value()))
        .expect_err("invalid metadata source");
    assert_eq!(
        info_error.identifier(),
        Some("RunMat:audioinfo:UnsupportedFormat")
    );
    assert_eq!(placement.observations.borrow().len(), 2);
    assert_eq!(
        placement.observations.borrow()[1].outcome,
        WholeWorkObservationOutcome::OperationalFailure
    );

    let cancellation = CancellationSource::new();
    let cancelled = runtime
        .clone()
        .with_cancellation_token(cancellation.token());
    cancellation.cancel(CancellationReason::User);
    let err = block_on(audioread_builtin(
        Some(cancelled),
        value(),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("cancelled");
    assert_eq!(err.identifier(), Some("RunMat:Cancelled"));
    let info_error = block_on(audioinfo_builtin(
        Some(
            runtime
                .clone()
                .with_cancellation_token(cancellation.token()),
        ),
        value(),
    ))
    .expect_err("cancelled metadata");
    assert_eq!(info_error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(placement.observations.borrow().len(), 2);
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);

    fs.pending_open.store(true, Ordering::SeqCst);
    let opens_before = fs.effects().0;
    let mut future = Box::pin(audioread_builtin(
        Some(runtime.clone()),
        value(),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ));
    assert!(matches!(
        future
            .as_mut()
            .poll(&mut Context::from_waker(noop_waker_ref())),
        Poll::Pending
    ));
    assert_eq!(placement.probes.load(Ordering::SeqCst), 1);
    drop(future);
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    assert_eq!(fs.effects().0, opens_before + 1);
    assert_eq!(placement.observations.borrow().len(), 2);
    assert_eq!(resources.calls.get(), 4);

    let never_polled = audioread_builtin(
        Some(runtime.clone()),
        value(),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    );
    drop(never_polled);
    assert_eq!(resources.calls.get(), 4);
    assert_eq!(fs.effects().0, opens_before + 1);
    let mut pending_info = Box::pin(audioinfo_builtin(Some(runtime.clone()), value()));
    assert!(matches!(
        pending_info
            .as_mut()
            .poll(&mut Context::from_waker(noop_waker_ref())),
        Poll::Pending
    ));
    assert_eq!(placement.probes.load(Ordering::SeqCst), 1);
    drop(pending_info);
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    assert_eq!(resources.calls.get(), 5);
    assert_eq!(fs.effects().0, opens_before + 2);
    let never_polled_info = audioinfo_builtin(Some(runtime.clone()), value());
    drop(never_polled_info);
    assert_eq!(resources.calls.get(), 5);
}
