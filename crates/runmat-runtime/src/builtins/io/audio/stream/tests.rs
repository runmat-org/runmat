use super::*;
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
use futures::executor::block_on;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::cell::Cell;
use std::num::NonZeroU64;
use std::rc::Rc;
use std::sync::Arc;

const PCM16: &[u8] = &[
    0x52, 0x49, 0x46, 0x46, 0x28, 0, 0, 0, 0x57, 0x41, 0x56, 0x45, 0x66, 0x6d, 0x74, 0x20, 0x10, 0,
    0, 0, 1, 0, 1, 0, 0x40, 0x1f, 0, 0, 0x80, 0x3e, 0, 0, 2, 0, 16, 0, 0x64, 0x61, 0x74, 0x61, 4,
    0, 0, 0, 0, 0, 0, 0x40,
];

fn input() -> (
    std::sync::MutexGuard<'static, ()>,
    fs::ProviderGuard,
    FsInputLease,
) {
    let guard = fs::provider_override_lock();
    assert_eq!(
        PCM16.len(),
        48,
        "two complete PCM16 frames after the header"
    );
    let provider = Arc::new(fs::MemoryFsProvider::with_current_dir("/audio"));
    provider
        .write_project_path("/audio/a.wav", PCM16)
        .expect("fixture");
    let owner = fs::replace_provider(provider);
    fs::set_current_dir("/audio").expect("cwd");
    let input = FsInputLease::capture(
        PathBuf::from("a.wav"),
        &AUDIOREAD_ERROR_IO,
        AUDIOREAD_BUILTIN_NAME,
    )
    .expect("lease");
    (guard, owner, input)
}

#[test]
fn stream_decodes_inclusive_open_ended_range() {
    let (_guard, _owner, input) = input();
    let mut cooperation = AudioCooperation::new(None, AUDIOREAD_BUILTIN_NAME);
    let options = AudioreadOptions {
        range: Some(AudioSampleRange {
            first: 2,
            last: None,
        }),
        datatype: AudioReadDataType::Double,
    };
    let decoded = block_on(decode_wave_stream(&input, options, &mut cooperation)).expect("range");
    assert_eq!(decoded.samples.shape, vec![1, 1]);
    assert_eq!(decoded.samples.materialize_f64(), vec![0.5]);
    assert_eq!(decoded.sample_rate, 8000.0);
}

#[test]
fn cancellation_during_decode_prevents_result_publication() {
    let (_guard, _owner, input) = input();
    let source = Rc::new(CancellationSource::new());
    let cancel = source.clone();
    let turns = Rc::new(Cell::new(0));
    let counted = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).expect("quantum")),
        source.token(),
        move || {
            counted.set(counted.get() + 1);
            if counted.get() == 3 {
                cancel.cancel(CancellationReason::User);
            }
            Box::pin(async { Ok(()) })
        },
    );
    let mut cooperation = AudioCooperation::with_budget(budget, AUDIOREAD_BUILTIN_NAME);
    let options = AudioreadOptions {
        range: None,
        datatype: AudioReadDataType::Native,
    };
    let error =
        block_on(decode_wave_stream(&input, options, &mut cooperation)).expect_err("cancelled");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(
        turns.get(),
        3,
        "header scan completed before decode cancellation"
    );
}

#[test]
fn unpolled_decode_does_not_start_a_cpu_turn() {
    let (_guard, _owner, input) = input();
    let calls = Rc::new(Cell::new(0));
    let counted = calls.clone();
    let source = CancellationSource::new();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).expect("quantum")),
        source.token(),
        move || {
            counted.set(counted.get() + 1);
            Box::pin(async { Ok(()) })
        },
    );
    let mut cooperation = AudioCooperation::with_budget(budget, AUDIOREAD_BUILTIN_NAME);
    let future = decode_wave_stream(
        &input,
        AudioreadOptions {
            range: None,
            datatype: AudioReadDataType::Double,
        },
        &mut cooperation,
    );
    drop(future);
    assert_eq!(calls.get(), 0);
}
