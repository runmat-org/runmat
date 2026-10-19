use super::*;
use crate::execution::CpuCooperationPolicy;
use flate2::{read::GzDecoder, write::GzEncoder, Compression, GzBuilder};
use futures::task::noop_waker;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::cell::Cell;
use std::future::Future;
use std::io::{Read, Write};
use std::num::NonZeroU64;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

fn legacy_compress(bytes: &[u8]) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(bytes).unwrap();
    encoder.finish().unwrap()
}

#[test]
fn stepped_codec_interoperates_with_existing_flate2_at_buffer_boundaries() {
    for length in [
        0,
        1,
        BUFFER_BYTES - 1,
        BUFFER_BYTES,
        BUFFER_BYTES + 1,
        131_073,
    ] {
        let bytes = (0..length)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        let encoded =
            futures::executor::block_on(transform(&bytes, Operation::Compress, None)).unwrap();
        let mut legacy_decoded = Vec::new();
        GzDecoder::new(encoded.as_slice())
            .read_to_end(&mut legacy_decoded)
            .unwrap();
        assert_eq!(legacy_decoded, bytes);
        assert_eq!(
            futures::executor::block_on(transform(
                &legacy_compress(&bytes),
                Operation::Decompress,
                None,
            ))
            .unwrap(),
            bytes,
        );
    }
}

#[test]
fn first_member_and_truncated_stream_contracts_are_preserved() {
    let mut members = legacy_compress(b"first");
    members.extend(legacy_compress(b"second"));
    assert_eq!(
        futures::executor::block_on(transform(&members, Operation::Decompress, None)).unwrap(),
        b"first",
    );
    let valid = legacy_compress(b"payload");
    let error = futures::executor::block_on(transform(
        &valid[..valid.len() - 1],
        Operation::Decompress,
        None,
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:gunzip:RuntimeError"));
    assert_eq!(
        error
            .source
            .as_ref()
            .unwrap()
            .downcast_ref::<io::Error>()
            .unwrap()
            .kind(),
        io::ErrorKind::UnexpectedEof,
    );
}

struct PendingTurn(Rc<Cell<usize>>);

impl Future for PendingTurn {
    type Output = Result<(), ExecutionServiceError>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

impl Drop for PendingTurn {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

#[test]
fn long_header_and_payload_work_reach_selected_pending_and_retire_on_cancel_or_drop() {
    let mut encoder = GzBuilder::new()
        .filename(vec![b'n'; BUFFER_BYTES * 3])
        .comment(vec![b'c'; BUFFER_BYTES * 2])
        .write(Vec::new(), Compression::default());
    encoder.write_all(b"body").unwrap();
    let encoded = encoder.finish().unwrap();
    for operation in [Operation::Compress, Operation::Decompress] {
        for cancel in [false, true] {
            let source = CancellationSource::new();
            let dropped = Rc::new(Cell::new(0));
            let retirements = dropped.clone();
            let mut budget = CpuCooperationBudget::with_test_turn(
                CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(BUFFER_BYTES as u64).unwrap()),
                source.token(),
                move || Box::pin(PendingTurn(retirements.clone())),
            );
            let mut future = Box::pin(transform(&encoded, operation, Some(&mut budget)));
            let waker = noop_waker();
            assert!(future
                .as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_pending());
            if cancel {
                source.cancel(CancellationReason::User);
                let Poll::Ready(Err(error)) =
                    future.as_mut().poll(&mut Context::from_waker(&waker))
                else {
                    panic!("selected cancellation must retire the pending codec turn");
                };
                assert_eq!(error.identifier(), Some("RunMat:parallel:Cancelled"));
            }
            drop(future);
            assert_eq!(dropped.get(), 1);
        }
    }
}

#[test]
fn highly_compressed_output_charges_decoded_bytes_to_the_selected_budget() {
    let bytes = vec![7; BUFFER_BYTES * 32];
    let encoded = legacy_compress(&bytes);
    let source = CancellationSource::new();
    let turns = Rc::new(Cell::new(0));
    let count = turns.clone();
    let mut budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(BUFFER_BYTES as u64).unwrap()),
        source.token(),
        move || {
            count.set(count.get() + 1);
            Box::pin(std::future::ready(Ok(())))
        },
    );
    let decoded = futures::executor::block_on(transform(
        &encoded,
        Operation::Decompress,
        Some(&mut budget),
    ))
    .unwrap();
    assert_eq!(decoded, bytes);
    assert!(turns.get() >= 32);
}
