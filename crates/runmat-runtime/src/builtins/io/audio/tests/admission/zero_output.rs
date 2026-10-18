use super::*;
use crate::context::host_effect::{HostEffectOutput, HostEffectWorkload};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::{ResourceBoundUnavailable, ResourceByteBound, WholeWorkObservationOutcome};
use std::sync::atomic::Ordering;
use std::sync::Arc;

fn fixture() -> (Arc<CountedFs>, runmat_filesystem::ProviderGuard) {
    let fs = Arc::new(CountedFs::new("/audio"));
    fs.inner
        .write_project_path("/audio/tone.wav", &pcm16_wav(8_000, 1, &[0, 16_384]))
        .expect("fixture WAV");
    let guard = runmat_filesystem::replace_provider(fs.clone());
    (fs, guard)
}

fn source() -> Value {
    Value::from("/audio/tone.wav")
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn discarded_audioinfo_has_no_retained_value_but_still_needs_scratch() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (fs, _guard) = fixture();
    let provider: Arc<dyn runmat_filesystem::FsProvider> = fs.clone();
    // Synthetic selector evidence, not a production codec or allocator certificate.
    let bounds = evidence(
        ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound),
        ResourceByteBound::Known(256 * 1024),
    );
    let (runtime, resources, placement) = context(provider.clone(), Some(bounds), Some(256 * 1024));
    let value = {
        let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        block_on(audioinfo_builtin(Some(runtime.clone()), source())).expect("discarded metadata")
    };
    assert!(output_list(value.clone()).is_empty());
    assert!(matches!(
        &value,
        Value::OutputList(values) if values.is_empty()
    ));
    let sequence = crate::call::arguments::adapt_legacy_builtin_result(value)
        .expect("empty output list adapts to an empty sequence");
    assert!(sequence.is_empty());
    assert_eq!(
        sequence.kind(),
        runmat_value::ValueSequenceKind::CommaSeparated
    );
    assert_eq!(placement.selections.get(), 1);
    assert_eq!(placement.observations.borrow().len(), 1);
    assert_eq!(
        placement.observations.borrow()[0].outcome,
        WholeWorkObservationOutcome::Success
    );
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    assert_eq!(
        placement.demands.borrow()[0].retained_bytes,
        ResourceByteBound::Known(0)
    );
    assert_eq!(
        placement.demands.borrow()[0].scratch_bytes,
        ResourceByteBound::Known(256 * 1024)
    );
    let effects = fs.effects();
    assert!(
        effects.0 > 0,
        "zero output still reads and validates the file"
    );

    let returned = HostEffectWorkload::AudioMetadata {
        source: std::path::Path::new("/audio/tone.wav"),
        output: HostEffectOutput::ReturnedValue,
    };
    let discarded = HostEffectWorkload::AudioMetadata {
        source: std::path::Path::new("/audio/tone.wav"),
        output: HostEffectOutput::EffectOnly,
    };
    assert_ne!(returned.identity_bytes(), discarded.identity_bytes());
    assert_eq!(resources.workloads.borrow()[0], discarded.identity_bytes());
    for requested in [Some(1), None] {
        let _outputs =
            crate::output_count::push_output_count(requested).expect("output-count scope");
        let error = block_on(audioinfo_builtin(Some(runtime.clone()), source()))
            .expect_err("returned metadata requires its own retained bound");
        assert_eq!(
            error.identifier(),
            Some("RunMat:Placement:InsufficientResourceEvidence")
        );
    }
    assert_eq!(
        resources.workloads.borrow()[1..],
        [returned.identity_bytes(), returned.identity_bytes()]
    );
    assert_eq!(
        fs.effects(),
        effects,
        "rejected forms have no additional file effects"
    );

    let scratch_unknown = evidence(
        ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoRetainedValueBound),
        ResourceByteBound::Unavailable(ResourceBoundUnavailable::NoPeakMemoryBound),
    );
    let (runtime, _, placement) = context(provider, Some(scratch_unknown), Some(256 * 1024));
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let error = block_on(audioinfo_builtin(Some(runtime), source()))
        .expect_err("discarded metadata still requires scratch evidence");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(placement.selections.get(), 0);
    assert_eq!(fs.effects(), effects);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn discarded_audioinfo_validates_format_and_preserves_missing_file_errors() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (fs, _guard) = fixture();
    fs.inner
        .write_project_path("/audio/tone.wav", b"not audio")
        .expect("malformed fixture");
    let provider: Arc<dyn runmat_filesystem::FsProvider> = fs.clone();
    let bounds = evidence(
        ResourceByteBound::Known(0),
        ResourceByteBound::Known(256 * 1024),
    );
    let (runtime, _, placement) = context(provider, Some(bounds), Some(256 * 1024));
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let error = block_on(audioinfo_builtin(Some(runtime), source()))
        .expect_err("discard must still validate audio format");
    assert_eq!(
        error.identifier(),
        Some("RunMat:audioinfo:UnsupportedFormat")
    );
    assert_eq!(placement.observations.borrow().len(), 1);
    assert_eq!(
        placement.observations.borrow()[0].outcome,
        WholeWorkObservationOutcome::OperationalFailure
    );
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    let effects = fs.effects();
    assert!(effects.0 > 0);

    let missing = block_on(audioinfo_builtin(None, Value::from("/audio/missing.wav")))
        .expect_err("missing input remains an I/O error");
    assert_eq!(missing.identifier(), Some("RunMat:audioinfo:Io"));
    assert_eq!(fs.effects().0, effects.0 + 1);
    assert_eq!(placement.observations.borrow().len(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn discarded_audioinfo_cancellation_during_synchronous_read_never_publishes() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (fs, _guard) = fixture();
    let provider: Arc<dyn runmat_filesystem::FsProvider> = fs.clone();
    let bounds = evidence(
        ResourceByteBound::Known(0),
        ResourceByteBound::Known(256 * 1024),
    );
    let (runtime, _, placement) = context(provider, Some(bounds), Some(256 * 1024));
    let cancellation = CancellationSource::new();
    fs.cancel_on_read
        .lock()
        .expect("cancel-on-read lock")
        .replace(cancellation.clone());
    let runtime = runtime.with_cancellation_token(cancellation.token());
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let error = block_on(audioinfo_builtin(Some(runtime), source()))
        .expect_err("synchronous read cancellation precedes publication");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(cancellation.token().is_cancelled());
    assert_eq!(placement.selections.get(), 1);
    assert!(placement.observations.borrow().is_empty());
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    assert_eq!(fs.effects().0, 1, "the selected owner was opened once");
}
