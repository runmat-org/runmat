use super::*;
use crate::execution::CpuCooperationPolicy;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::num::NonZeroU64;

#[test]
fn chunked_fill_stops_before_publishing_unfinished_output() {
    let cancellation = CancellationSource::new();
    let mut budget = Some(CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(OUTPUT_WORK_CHUNK as u64).unwrap()),
        cancellation.token(),
        move || {
            cancellation.cancel(CancellationReason::User);
            Box::pin(async { Ok(()) })
        },
    ));
    let mut output = vec![0.0; OUTPUT_WORK_CHUNK * 2];
    let error = futures::executor::block_on(fill_chunks(
        &mut output,
        |_, chunk| {
            chunk.fill(1.0);
            chunk.len()
        },
        &mut budget,
        None,
    ))
    .expect_err("cancelled work must not publish an output");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(output[0], 1.0);
    assert_eq!(output[OUTPUT_WORK_CHUNK], 0.0);
}
