use super::*;
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::{cell::Cell, num::NonZeroU64, rc::Rc};

#[test]
fn immediately_ready_slice_read_stops_at_execution_quantum_without_output() {
    let payload = DataArrayPayload::zeros("f64".into(), vec![1, 8_192]).unwrap();
    let cancellation = Rc::new(CancellationSource::new());
    let turns = Rc::new(Cell::new(0));
    let cancel = cancellation.clone();
    let counted = turns.clone();
    let mut budget = Some(CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        cancellation.token(),
        move || {
            counted.set(counted.get() + 1);
            cancel.cancel(CancellationReason::User);
            Box::pin(async { Ok(()) })
        },
    ));
    let error = futures::executor::block_on(read_slice_payload(
        &payload,
        &Value::String(":".into()),
        &mut budget,
    ))
    .expect_err("read slice must abort rather than publish a partial payload");
    assert_eq!(error.identifier.as_deref(), Some("RunMat:Cancelled"));
    assert_eq!(turns.get(), 1);
}

#[test]
fn large_zero_allocation_stops_at_bounded_work_quantum() {
    let cancellation = Rc::new(CancellationSource::new());
    let cancel = cancellation.clone();
    let turns = Rc::new(Cell::new(0));
    let counted = turns.clone();
    let mut budget = Some(CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(4_096).unwrap()),
        cancellation.token(),
        move || {
            counted.set(counted.get() + 1);
            cancel.cancel(CancellationReason::User);
            Box::pin(async { Ok(()) })
        },
    ));
    let error = futures::executor::block_on(DataArrayPayload::zeros_cooperative(
        "uint8".into(),
        vec![1, 65_536],
        &mut budget,
    ))
    .expect_err("allocation must not publish a partial payload after cancellation");
    assert_eq!(error.identifier.as_deref(), Some("RunMat:Cancelled"));
    assert_eq!(turns.get(), 1);
}
