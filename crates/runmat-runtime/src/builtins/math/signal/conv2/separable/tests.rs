use futures::executor::block_on;
use futures::task::noop_waker;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::cell::Cell;
use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

use super::*;
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};

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
fn pending_separable_preparation_cancels_or_abandons_without_publication() {
    for cancel in [true, false] {
        let source = CancellationSource::new();
        let dropped = Rc::new(Cell::new(0));
        let turns = dropped.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || Box::pin(PendingTurn(turns.clone())),
        );
        let mut cooperation = SignalCooperation::with_budget(budget, "conv2");
        let signal = real_matrix(2, 2, &[1.0, 3.0, 2.0, 4.0]);
        let column = [Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
        let row = [Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        let mut work = Box::pin(convolve_separable(
            &signal,
            &column,
            &row,
            Conv2Mode::Full,
            false,
            &mut cooperation,
        ));
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(matches!(work.as_mut().poll(&mut context), Poll::Pending));
        if cancel {
            source.cancel(CancellationReason::User);
            let Poll::Ready(Err(error)) = work.as_mut().poll(&mut context) else {
                panic!("cancelled convolution must not publish a matrix");
            };
            assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
            assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
            assert!(matches!(
                error
                    .source
                    .as_deref()
                    .and_then(|cause| cause.downcast_ref::<ExecutionServiceError>()),
                Some(ExecutionServiceError::Cancelled)
            ));
        }
        drop(work);
        assert_eq!(dropped.get(), 1, "pending task turn must be dropped");
    }
}

#[test]
fn pending_result_materialization_cannot_publish_after_cancellation() {
    let source = CancellationSource::new();
    let dropped = Rc::new(Cell::new(0));
    let turns = dropped.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || Box::pin(PendingTurn(turns.clone())),
    );
    let mut cooperation = SignalCooperation::with_budget(budget, "conv2");
    let result = real_matrix(2, 2, &[1.0, 3.0, 2.0, 4.0]);
    let mut publish = Box::pin(super::super::matrix_to_value(
        result,
        false,
        false,
        &mut cooperation,
    ));
    let waker = noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(publish.as_mut().poll(&mut context), Poll::Pending));
    source.cancel(CancellationReason::User);
    let Poll::Ready(Err(error)) = publish.as_mut().poll(&mut context) else {
        panic!("cancelled publication must not expose a tensor");
    };
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    drop(publish);
    assert_eq!(dropped.get(), 1);
}

fn real_matrix(rows: usize, cols: usize, data: &[f64]) -> Matrix {
    Matrix {
        rows,
        cols,
        data: data
            .iter()
            .copied()
            .map(|real| Complex::new(real, 0.0))
            .collect(),
    }
}

#[test]
fn two_pass_full_same_and_valid_use_the_original_signal_geometry() {
    let signal = real_matrix(2, 2, &[1.0, 3.0, 2.0, 4.0]);
    let column = [Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
    let row = [Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0)];
    for (mode, shape, expected) in [
        (
            Conv2Mode::Full,
            (3, 3),
            vec![1.0, 5.0, 6.0, 1.0, 3.0, 2.0, -2.0, -8.0, -8.0],
        ),
        (Conv2Mode::Same, (2, 2), vec![3.0, 2.0, -8.0, -8.0]),
        (Conv2Mode::Valid, (1, 1), vec![3.0]),
    ] {
        let mut cooperation = SignalCooperation::new(None, "conv2");
        let output = block_on(convolve_separable(
            &signal,
            &column,
            &row,
            mode,
            false,
            &mut cooperation,
        ))
        .expect("separable convolution");
        assert_eq!((output.rows, output.cols), shape);
        assert_eq!(
            output.data.iter().map(|value| value.re).collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn empty_and_short_valid_cases_do_not_allocate_full_intermediates() {
    let empty = real_matrix(0, 3, &[]);
    let column = [Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
    let row = [Complex::new(1.0, 0.0)];
    let mut cooperation = SignalCooperation::new(None, "conv2");
    let same = block_on(convolve_separable(
        &empty,
        &column,
        &row,
        Conv2Mode::Same,
        false,
        &mut cooperation,
    ))
    .expect("empty Same");
    assert_eq!((same.rows, same.cols), (0, 3));
    let full = block_on(convolve_separable(
        &empty,
        &column,
        &row,
        Conv2Mode::Full,
        false,
        &mut cooperation,
    ))
    .expect("empty Full");
    assert_eq!((full.rows, full.cols), (0, 0));
    let short = real_matrix(1, 1, &[2.0]);
    let valid = block_on(convolve_separable(
        &short,
        &column,
        &row,
        Conv2Mode::Valid,
        false,
        &mut cooperation,
    ))
    .expect("short Valid");
    assert_eq!((valid.rows, valid.cols), (0, 0));
}

#[test]
fn balanced_factors_keep_finite_extreme_products_finite() {
    let signal = real_matrix(1, 1, &[2.0]);
    for (column_value, row_value) in [(1e308, 1e-308), (1e-308, 1e308)] {
        let column = [Complex::new(column_value, 0.0)];
        let row = [Complex::new(row_value, 0.0)];
        let mut cooperation = SignalCooperation::new(None, "conv2");
        let output = block_on(convolve_separable(
            &signal,
            &column,
            &row,
            Conv2Mode::Full,
            false,
            &mut cooperation,
        ))
        .expect("finite extreme product");
        let expected = 2.0 * (column_value * row_value);
        assert!(output.data[0].re.is_finite());
        assert!((output.data[0].re - expected).abs() <= 1e-12);
    }
}

#[test]
fn balanced_complex_and_single_passes_preserve_output_domain() {
    let signal = real_matrix(1, 1, &[2.0]);
    let column = [Complex::new(1e308, 0.0)];
    let row = [Complex::new(0.0, 1e-308)];
    let mut cooperation = SignalCooperation::new(None, "conv2");
    let complex = block_on(convolve_separable(
        &signal,
        &column,
        &row,
        Conv2Mode::Full,
        false,
        &mut cooperation,
    ))
    .expect("finite complex product");
    assert!(complex.data[0].im.is_finite());
    assert!((complex.data[0].im - 2.0).abs() <= 1e-12);

    let column = [Complex::new(1e30, 0.0)];
    let row = [Complex::new(1e-30, 0.0)];
    let single = block_on(convolve_separable(
        &signal,
        &column,
        &row,
        Conv2Mode::Full,
        true,
        &mut cooperation,
    ))
    .expect("finite single product");
    assert!(single.data[0].re.is_finite());
    assert!((single.data[0].re - 2.0).abs() <= 1e-6);
}

#[test]
fn unsafe_intermediate_uses_general_kernel_before_two_pass_work() {
    let signal = real_matrix(2, 1, &[1e308, 1e308]);
    let column = [Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
    let row = [Complex::new(1e-308, 0.0)];
    let mut cooperation = SignalCooperation::new(None, "conv2");
    let output = block_on(convolve_separable(
        &signal,
        &column,
        &row,
        Conv2Mode::Full,
        false,
        &mut cooperation,
    ))
    .expect("finite general-kernel output");
    assert_eq!((output.rows, output.cols), (3, 1));
    for (actual, expected) in output.data.iter().zip([1.0, 2.0, 1.0]) {
        assert!(actual.re.is_finite());
        assert!((actual.re - expected).abs() <= 1e-12);
    }

    let tiny = real_matrix(1, 1, &[f64::from_bits(1)]);
    let tiny_column = [Complex::new(1e-308, 0.0)];
    let huge_row = [Complex::new(1e308, 0.0)];
    let output = block_on(convolve_separable(
        &tiny,
        &tiny_column,
        &huge_row,
        Conv2Mode::Full,
        false,
        &mut cooperation,
    ))
    .expect("subnormal general-kernel output");
    assert_eq!(output.data[0].re.to_bits(), f64::from_bits(1).to_bits());
}
