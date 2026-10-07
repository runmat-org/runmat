use super::*;
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::cell::Cell;
use std::num::NonZeroU64;
use std::rc::Rc;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn materialization_preserves_rank_zero_and_rank_one_image_geometry() {
    let scalar = Tensor::new(vec![0.5], Vec::new()).expect("rank-zero image");
    let mut cooperation = ImageCooperation::new(None, "imwrite");
    let scalar = futures::executor::block_on(materialize_direct_image(&scalar, &mut cooperation))
        .expect("rank-zero image is one grayscale pixel");
    assert_eq!((scalar.rows, scalar.cols, scalar.channels), (1, 1, 1));

    let row = Tensor::new(vec![0.0, 0.5, 1.0], vec![3]).expect("rank-one image");
    let mut cooperation = ImageCooperation::new(None, "imwrite");
    let row = futures::executor::block_on(materialize_direct_image(&row, &mut cooperation))
        .expect("rank-one image is one grayscale row");
    assert_eq!((row.rows, row.cols, row.channels), (1, 3, 1));
}

#[test]
fn host_pixel_conversion_yields_and_cancellation_discards_partial_output() {
    let source = CancellationSource::new();
    let turns = Rc::new(Cell::new(0));
    let counted = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            counted.set(counted.get() + 1);
            Box::pin(async { Ok(()) })
        },
    );
    let input = Tensor::new(vec![0.5; 64 * 64 * 3], vec![64, 64, 3]).unwrap();
    let mut cooperation = ImageCooperation::with_budget(budget, "imwrite");
    let output =
        futures::executor::block_on(materialize_direct_image(&input, &mut cooperation)).unwrap();
    assert_eq!(output.channels, 3);
    assert!(
        turns.get() >= 3,
        "fill and conversion must both charge work"
    );

    let source = Rc::new(CancellationSource::new());
    let cancel_on_turn = source.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            cancel_on_turn.cancel(CancellationReason::User);
            Box::pin(async { Ok(()) })
        },
    );
    let mut cooperation = ImageCooperation::with_budget(budget, "imwrite");
    let error =
        match futures::executor::block_on(materialize_direct_image(&input, &mut cooperation)) {
            Ok(_) => panic!("cancelled fill must not publish partial image"),
            Err(error) => error,
        };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
}

#[test]
fn unpolled_conversion_has_no_work() {
    let source = CancellationSource::new();
    let turns = Rc::new(Cell::new(0));
    let counted = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            counted.set(counted.get() + 1);
            Box::pin(async { Ok(()) })
        },
    );
    let input = Tensor::new(vec![0.5; 4], vec![2, 2]).unwrap();
    let mut cooperation = ImageCooperation::with_budget(budget, "imwrite");
    drop(materialize_direct_image(&input, &mut cooperation));
    assert_eq!(turns.get(), 0);
}
