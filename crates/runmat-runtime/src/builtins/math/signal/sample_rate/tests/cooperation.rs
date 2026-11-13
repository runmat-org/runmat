use super::*;

#[test]
fn host_sample_rate_observes_cancellation_during_mapping() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::cell::Cell;
    use std::num::NonZeroU64;
    use std::rc::Rc;

    let source = Rc::new(CancellationSource::new());
    let cancel_on_turn = source.clone();
    let turns = Rc::new(Cell::new(0));
    let observed = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(SIGNAL_WORK_CHUNK).unwrap()),
        source.token(),
        move || {
            observed.set(observed.get() + 1);
            cancel_on_turn.cancel(CancellationReason::User);
            Box::pin(async { Ok(()) })
        },
    );
    let mut cooperation = SignalCooperation::with_budget(budget, DOWNSAMPLE_NAME);
    let input = SampleInput::Real {
        storage: NumericStorage::F64(vec![1.0; SIGNAL_WORK_CHUNK as usize * 2]),
        shape: vec![1, SIGNAL_WORK_CHUNK as usize * 2],
    };
    let error = block_on(apply_sample_rate(
        input,
        1,
        0,
        SampleOp::Down,
        DOWNSAMPLE_NAME,
        &mut cooperation,
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(turns.get(), 1);
}
