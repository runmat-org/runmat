use super::*;
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
use flate2::write::ZlibEncoder;
use flate2::Compression;
use futures::executor::block_on;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::io::Write;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn header() -> Vec<u8> {
    let mut bytes = vec![0u8; MAT_HEADER_LEN];
    bytes[126] = b'I';
    bytes[127] = b'M';
    bytes
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn compressed_stream_matches_plain_numeric_class_and_value() {
    let plain = block_on(
        crate::builtins::io::mat::save::encode_workspace_to_mat_bytes(&[(
            "answer".into(),
            Value::Num(42.25),
        )]),
    )
    .expect("encode fixture");
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(&plain[MAT_HEADER_LEN..])
        .expect("compress fixture");
    let compressed = encoder.finish().expect("finish fixture");
    let mut wrapped = plain[..MAT_HEADER_LEN].to_vec();
    wrapped.extend_from_slice(&MI_COMPRESSED.to_le_bytes());
    wrapped.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
    wrapped.extend_from_slice(&compressed);
    let expected = read_mat_reader(&mut Cursor::new(plain)).expect("plain decode");
    let mut cooperation = MatDecodeCooperation::new(None, None);
    let actual = block_on(read_mat_reader_cooperative(
        &mut Cursor::new(wrapped),
        &mut cooperation,
    ))
    .expect("streamed compressed decode");
    assert_eq!(actual, expected);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn service_cancellation_interrupts_a_large_tag_before_its_tail_is_read() {
    let mut bytes = header();
    bytes.extend_from_slice(&99u32.to_le_bytes());
    bytes.extend_from_slice(&(2 * 64 * 1024u32).to_le_bytes());
    bytes.resize(bytes.len() + 2 * 64 * 1024, 7);
    let source = CancellationSource::new();
    let turns = Arc::new(AtomicUsize::new(0));
    let observed = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            let call = observed.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                if call >= 1 {
                    Err(ExecutionServiceError::Cancelled)
                } else {
                    Ok(())
                }
            })
        },
    );
    let mut reader = Cursor::new(bytes.clone());
    let mut cooperation = MatDecodeCooperation::with_budget(budget);
    let error = block_on(read_mat_reader_cooperative(&mut reader, &mut cooperation))
        .expect_err("service cancels inside declared tag body");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(
        error
            .source
            .as_deref()
            .and_then(|cause| cause.downcast_ref::<ExecutionServiceError>()),
        Some(&ExecutionServiceError::Cancelled)
    );
    assert!((reader.position() as usize) < bytes.len());
    assert!(!source.token().is_cancelled());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn preexisting_cancellation_reads_no_header_bytes() {
    let source = CancellationSource::new();
    source.cancel(CancellationReason::User);
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        || Box::pin(async { Ok(()) }),
    );
    let mut reader = Cursor::new(header());
    let mut cooperation = MatDecodeCooperation::with_budget(budget);
    let error = block_on(read_mat_reader_cooperative(&mut reader, &mut cooperation))
        .expect_err("cancel before codec read");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(reader.position(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn restoration_cancellation_drops_all_unpublished_entries() {
    let source = CancellationSource::new();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        || Box::pin(async { Err(ExecutionServiceError::Cancelled) }),
    );
    let entries = vec![
        ("first".into(), Value::Num(1.0)),
        ("second".into(), Value::Num(2.0)),
    ];
    let mut cooperation = MatDecodeCooperation::with_budget(budget);
    let error = block_on(super::super::restore_loaded_entries(
        entries,
        None,
        Some(&mut cooperation),
    ))
    .expect_err("restoration cannot publish partial entries");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(!source.token().is_cancelled());
}
