use super::*;
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_value::StringArray;
use std::cell::Cell;
use std::num::NonZeroU64;
use std::rc::Rc;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn rank_one_data_is_one_box_and_rank_zero_has_no_observations() {
    let rank_one = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap());
    let boxes = build_box_stats(rank_one, None, &BoxplotOptions::default()).expect("one box");
    assert_eq!(boxes.len(), 1);
    assert_eq!(boxes[0].observation_count, 3);
    assert_eq!(boxes[0].median, 2.0);

    let rank_zero = Value::Tensor(Tensor::new(vec![7.0], vec![]).unwrap());
    let error = build_box_stats(rank_zero, None, &BoxplotOptions::default())
        .expect_err("rank-zero tensor has no matrix observations");
    assert_eq!(
        error.identifier(),
        BOXPLOT_ERROR_INVALID_ARGUMENT.identifier
    );
    assert!(error.message().contains("at least one non-NaN"));
}

#[test]
fn ungrouped_matrix_preserves_column_order_and_numerics_with_cooperation() {
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
    let mut cooperation = super::super::cooperation::BoxplotCooperation::with_budget(budget);
    let data = Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap());
    let layout = futures::executor::block_on(build_box_layout_cooperative(
        data,
        None,
        &BoxplotOptions::default(),
        &mut cooperation,
    ))
    .unwrap();
    assert_eq!(layout.boxes.len(), 2);
    assert_eq!(layout.boxes[0].median, 2.0);
    assert_eq!(layout.boxes[1].median, 3.0);
    assert_eq!(layout.boxes[0].observation_count, 2);
    assert_eq!(layout.boxes[1].observation_count, 2);
    assert!(turns.get() >= 4);
}

#[test]
fn cancellation_during_ungrouped_sample_copy_prevents_layout() {
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
    let mut cooperation = super::super::cooperation::BoxplotCooperation::with_budget(budget);
    let data = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
    let error = futures::executor::block_on(build_box_layout_cooperative(
        data,
        None,
        &BoxplotOptions::default(),
        &mut cooperation,
    ))
    .expect_err("cancelled sample preparation cannot produce a layout");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
}

#[test]
fn matrix_group_retry_never_relabels_service_failure_as_semantic_error() {
    for group_count in [8, 4] {
        let source = CancellationSource::new();
        let turns = Rc::new(Cell::new(0));
        let counted = turns.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || {
                counted.set(counted.get() + 1);
                Box::pin(async {
                    Err(crate::execution::ExecutionServiceError::Infrastructure(
                        "group turn failed".into(),
                    ))
                })
            },
        );
        let mut cooperation = super::super::cooperation::BoxplotCooperation::with_budget(budget);
        let group = Value::StringArray(
            StringArray::new(
                (0..group_count).map(|index| format!("g{index}")).collect(),
                vec![group_count, 1],
            )
            .unwrap(),
        );
        let data = Value::Tensor(Tensor::new(vec![1.0; 8], vec![4, 2]).unwrap());
        let error = futures::executor::block_on(build_box_layout_cooperative(
            data,
            Some(group),
            &BoxplotOptions::default(),
            &mut cooperation,
        ))
        .expect_err("service failure must stop either first or row-count retry");
        assert_eq!(error.identifier(), BOXPLOT_ERROR_INTERNAL.identifier);
        assert!(error.source.is_some());
        assert_eq!(turns.get(), 1);
    }
}
