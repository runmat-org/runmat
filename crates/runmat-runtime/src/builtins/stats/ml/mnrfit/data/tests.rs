use super::*;
use crate::builtins::common::tensor;
use futures::executor::block_on;
use runmat_value::{IntegerStorage, ObjectInstance, StringArray};

#[test]
fn categorical_response_uses_table_owned_borrowed_code_decoding() {
    let prefix = "c".repeat(16_385);
    let first = format!("{prefix}a");
    let second = format!("{prefix}b");
    let mut object = ObjectInstance::new(runmat_types::standard::CATEGORICAL.to_string());
    object.replace_property_unbounded(
        "Codes".into(),
        Value::Tensor(Tensor::new_integer(IntegerStorage::U64(vec![2, 1, 2]), vec![3, 1]).unwrap()),
    );
    object.replace_property_unbounded(
        "Categories".into(),
        Value::StringArray(
            StringArray::new(vec![first.clone(), second.clone()], vec![1, 2]).unwrap(),
        ),
    );
    let response = block_on(response_outcomes(
        Value::Object(object),
        3,
        ResponseExactness::ValidateDownloaded,
        &mut FitCooperation::new(None),
    ))
    .expect("categorical codes use the canonical table decoder without eager category copies");
    let ClassNames::Text(classes) = response.classes else {
        panic!("categorical responses must retain text class labels");
    };
    assert_eq!(classes, vec![first, second]);
    assert_eq!(response.outcomes, vec![vec![0.0, 1.0, 0.0]]);
}

#[test]
fn pending_browser_style_turn_cancels_during_design_without_publication() {
    use futures::task::noop_waker;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::cell::Cell;
    use std::future::Future;
    use std::num::NonZeroU64;
    use std::rc::Rc;
    use std::task::{Context, Poll};

    let source = CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    let dropped = Rc::new(Cell::new(false));
    struct PendingTurn(Rc<Cell<bool>>);
    impl Future for PendingTurn {
        type Output = Result<(), crate::execution::ExecutionServiceError>;
        fn poll(self: std::pin::Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }
    impl Drop for PendingTurn {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }
    let dropped_turn = dropped.clone();
    let budget = crate::execution::CpuCooperationBudget::with_test_turn(
        crate::execution::CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || Box::pin(PendingTurn(dropped_turn.clone())),
    );
    let mut cooperation = FitCooperation::with_budget(&runtime, budget);
    let rows = 20_000;
    let x = Tensor::new(vec![1.0; rows], vec![rows, 1]).unwrap();
    let response = ResponseOutcomes {
        outcomes: vec![vec![1.0; rows]],
        classes: ClassNames::Numeric(vec![1.0, 2.0]),
        weights: vec![1.0; rows],
    };
    let check = || check_cancellation(Some(&runtime));
    let options = Options::default();
    let mut pending = Box::pin(prepare_data(
        &x,
        response,
        &options,
        &check,
        &mut cooperation,
    ));
    let waker = noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(pending.as_mut().poll(&mut context).is_pending());
    source.cancel(CancellationReason::User);
    let Poll::Ready(Err(error)) = pending.as_mut().poll(&mut context) else {
        panic!("cancelled pending design must not return prepared data");
    };
    assert!(error
        .identifier()
        .is_some_and(|identifier| identifier.ends_with(":Cancelled")));
    drop(pending);
    assert!(dropped.get(), "pending service turn must be dropped");
}

#[test]
fn cancellation_during_design_copy_discards_prepared_data() {
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::cell::Cell;
    use std::rc::Rc;

    let source = CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    let rows = 20_000;
    let x = Tensor::new(vec![1.0; rows], vec![rows, 1]).unwrap();
    let response = ResponseOutcomes {
        outcomes: vec![vec![1.0; rows]],
        classes: ClassNames::Numeric(vec![1.0, 2.0]),
        weights: vec![1.0; rows],
    };
    let probes = Cell::new(0);
    let error = block_on(prepare_data(
        &x,
        response,
        &Options::default(),
        &|| {
            let current = probes.get() + 1;
            probes.set(current);
            if current == 6 {
                source.cancel(CancellationReason::User);
            }
            check_cancellation(Some(&runtime))
        },
        &mut FitCooperation::new(None),
    ))
    .expect_err("cancellation during design copying cannot publish prepared data");
    assert!(probes.get() >= 6);
    assert!(error
        .identifier()
        .is_some_and(|identifier| identifier.ends_with(":Cancelled")));
}

#[test]
fn cancellation_during_compaction_discards_partial_work() {
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::cell::Cell;
    use std::rc::Rc;

    let source = CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    let rows = 20_000;
    let mut x_values = vec![1.0; rows];
    x_values[rows - 1] = f64::NAN;
    let x = Tensor::new(x_values, vec![rows, 1]).unwrap();
    let values = tensor::tensor_values_f64_cow(&x);
    let mut design = vec![1.0; rows];
    let mut outcomes = vec![vec![1.0; rows]];
    let mut weights = vec![1.0; rows];
    let probes = Cell::new(0);
    let error = block_on(compact_complete_cases(
        CompleteCaseBuffers {
            x: &x,
            values: &values,
            cols: 1,
            design: &mut design,
            outcomes: &mut outcomes,
            weights: &mut weights,
        },
        &|| {
            let current = probes.get() + 1;
            probes.set(current);
            if current == 8 {
                source.cancel(CancellationReason::User);
            }
            check_cancellation(Some(&runtime))
        },
        &mut FitCooperation::new(None),
    ))
    .expect_err("cancellation during compaction cannot publish partial fit data");
    assert!(probes.get() >= 8);
    assert!(error
        .identifier()
        .is_some_and(|identifier| identifier.ends_with(":Cancelled")));
}

#[test]
fn adjacent_finite_numeric_labels_remain_distinct_classes() {
    let next = f64::from_bits(1.0f64.to_bits() + 1);
    let response = block_on(numeric_outcomes(
        vec![1.0, next, 1.0, next],
        vec![4, 1],
        4,
        &mut FitCooperation::new(None),
    ))
    .expect("adjacent representable class values are distinct");
    let ClassNames::Numeric(classes) = response.classes else {
        panic!("numeric Y must retain numeric class labels");
    };
    assert_eq!(classes, vec![1.0, next]);
    assert_eq!(response.outcomes, vec![vec![1.0, 0.0, 1.0, 0.0]]);

    let signed_zero = block_on(numeric_outcomes(
        vec![-0.0, 0.0, 1.0],
        vec![3, 1],
        3,
        &mut FitCooperation::new(None),
    ))
    .expect("signed zeros denote the same numeric class");
    assert_eq!(signed_zero.classes.len(), 2);
    assert_eq!(signed_zero.outcomes, vec![vec![1.0, 1.0, 0.0]]);
}

#[test]
fn grouped_finite_counts_reject_unrepresentable_row_and_total() {
    let overflowing_row = Tensor::new(vec![1.0e308, 1.0e308], vec![1, 2]).unwrap();
    let err = block_on(count_outcomes(
        &overflowing_row,
        &mut FitCooperation::new(None),
    ))
    .err()
    .expect("row total must stay finite");
    assert!(err.message().contains("grouped count total"));

    let overflowing_observations =
        Tensor::new(vec![1.0e308, 1.0e308, 0.0, 0.0], vec![2, 2]).unwrap();
    let err = block_on(count_outcomes(
        &overflowing_observations,
        &mut FitCooperation::new(None),
    ))
    .err()
    .expect("aggregate observation count must stay finite");
    assert!(err.message().contains("grouped count total"));

    let omitted_missing_row = Tensor::new(vec![1.0e308, 1.0, f64::NAN, 1.0], vec![2, 2]).unwrap();
    let response = block_on(count_outcomes(
        &omitted_missing_row,
        &mut FitCooperation::new(None),
    ))
    .expect("a missing row does not contribute an overflowing total");
    assert_eq!(response.weights, vec![0.0, 2.0]);
}

#[test]
fn finite_weight_products_and_totals_must_remain_representable() {
    let x = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
    let response = ResponseOutcomes {
        outcomes: vec![vec![1.0, 0.0]],
        classes: ClassNames::Numeric(vec![1.0, 2.0]),
        weights: vec![2.0, 2.0],
    };
    let options = Options {
        weights: Some(vec![f64::MAX, 1.0]),
        ..Options::default()
    };
    let err = block_on(prepare_data(
        &x,
        response,
        &options,
        &|| Ok(()),
        &mut FitCooperation::new(None),
    ))
    .expect_err("finite operand multiplication may not silently drop a row");
    assert!(err.message().contains("weighted observation count"));

    let response = ResponseOutcomes {
        outcomes: vec![vec![1.0, 0.0]],
        classes: ClassNames::Numeric(vec![1.0, 2.0]),
        weights: vec![1.0, 1.0],
    };
    let options = Options {
        weights: Some(vec![-1.0, 1.0]),
        ..Options::default()
    };
    let err = block_on(prepare_data(
        &x,
        response,
        &options,
        &|| Ok(()),
        &mut FitCooperation::new(None),
    ))
    .expect_err("a negative option weight must not be silently omitted");
    assert!(err.message().contains("nonnegative"));

    let response = ResponseOutcomes {
        outcomes: vec![vec![1.0, 0.0]],
        classes: ClassNames::Numeric(vec![1.0, 2.0]),
        weights: vec![1.0e308, 1.0e308],
    };
    let err = block_on(prepare_data(
        &x,
        response,
        &Options::default(),
        &|| Ok(()),
        &mut FitCooperation::new(None),
    ))
    .expect_err("finite weights with an unrepresentable total must reject");
    assert!(err.message().contains("weights total"));
}
