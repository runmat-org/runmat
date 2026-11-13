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

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
    assert_eq!(decoded.samples.shape(), &[1, 1]);
    assert_eq!(decoded.samples.materialize_f64(), vec![0.5]);
    assert_eq!(decoded.sample_rate, 8000.0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_sparse_large_wave_decodes_a_small_requested_range() {
    use std::fs::{self as std_fs, OpenOptions};
    use std::io::Write;

    let _lock = fs::provider_override_lock();
    let stamp = runmat_time::system_time_now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "runmat-audio-small-range-{}-{stamp}",
        std::process::id()
    ));
    let mut header = PCM16.to_vec();
    let data_bytes = 1_000_000_000u32;
    header[4..8].copy_from_slice(&(36 + data_bytes).to_le_bytes());
    header[40..44].copy_from_slice(&data_bytes.to_le_bytes());
    let mut writer = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .expect("isolated sparse WAV");
    writer.write_all(&header).expect("WAV prefix");
    writer
        .set_len(44 + u64::from(data_bytes))
        .expect("sparse WAV extent");
    let _owner = fs::replace_provider(Arc::new(fs::NativeFsProvider));
    let input = FsInputLease::capture(path.clone(), &AUDIOREAD_ERROR_IO, AUDIOREAD_BUILTIN_NAME)
        .expect("native source");
    let opened = block_on(input.open()).expect("opened source");
    assert_eq!(opened.opened_extent(), Some(44 + u64::from(data_bytes)));
    drop(opened);
    let mut cooperation = AudioCooperation::new(None, AUDIOREAD_BUILTIN_NAME);
    let decoded = block_on(decode_wave_stream(
        &input,
        AudioreadOptions {
            range: Some(AudioSampleRange {
                first: 1,
                last: Some(1),
            }),
            datatype: AudioReadDataType::Double,
        },
        &mut cooperation,
    ))
    .expect("one-frame decode");
    assert_eq!(decoded.samples.shape(), &[1, 1]);
    assert_eq!(decoded.samples.materialize_f64(), vec![0.0]);
    drop(writer);
    std_fs::remove_file(path).expect("remove isolated WAV");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
