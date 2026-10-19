use super::*;
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
use futures::executor::block_on;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::ResourceByteBound;
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn service_cancellation_with_unmarked_token_releases_receipt_without_feedback() {
    let _lock = runmat_filesystem::provider_override_lock();
    let filesystem = Arc::new(CountedFs::new("/audio"));
    let _owner = runmat_filesystem::replace_provider(filesystem.clone());
    let provider: Arc<dyn runmat_filesystem::FsProvider> = filesystem.clone();
    let (runtime, resources, placement) = context(
        provider,
        Some(evidence(
            ResourceByteBound::Known(4096),
            ResourceByteBound::Known(8192),
        )),
        Some(16_384),
    );
    let input = FsInputLease::capture(
        PathBuf::from("/audio/tone.wav"),
        &AUDIOREAD_ERROR_IO,
        AUDIOREAD_BUILTIN_NAME,
    )
    .expect("captured source authority");
    let receipt = select_decode(
        Some(&runtime),
        &input,
        AudioreadOptions {
            range: None,
            datatype: AudioReadDataType::Double,
        },
        None,
    )
    .expect("typed audio route selected")
    .expect("host receipt");
    assert_eq!(resources.calls.get(), 1);
    assert_eq!(placement.probes.load(Ordering::SeqCst), 1);

    let source = CancellationSource::new();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        || Box::pin(async { Err(ExecutionServiceError::Cancelled) }),
    );
    let mut cooperation = AudioCooperation::with_budget(budget, AUDIOREAD_BUILTIN_NAME);
    let error = block_on(cooperation.completed(1)).expect_err("service-origin cancellation");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(cooperation.was_cancelled());
    assert!(!source.token().is_cancelled());
    assert!(!runtime.is_cancelled());
    let result: BuiltinResult<Value> = Err(error);
    finish_receipt(Some(receipt), &result, cooperation.was_cancelled());
    assert_eq!(placement.probes.load(Ordering::SeqCst), 0);
    assert!(placement.observations.borrow().is_empty());
    assert_eq!(filesystem.effects(), (0, 0, 0));
}
