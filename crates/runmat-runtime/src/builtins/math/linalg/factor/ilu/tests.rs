use super::*;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
};
use crate::execution::RuntimeExecutionService;
use futures::executor::block_on;
use std::rc::Rc;
struct Parallel(ManagedHostEffectLedger);
impl RuntimeParallelService for Parallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let s = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(s.free_bytes()),
            epoch: s.epoch,
            ..Default::default()
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}
fn runtime() -> RuntimeContext {
    runtime_with_ledger().0
}
fn runtime_with_ledger() -> (RuntimeContext, ManagedHostEffectLedger) {
    let ledger = ManagedHostEffectLedger::new(2_000_000);
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger.clone()))),
    );
    (runtime, ledger)
}
fn source() -> SparseTensor {
    SparseTensor::new(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![4., 1., 1., 4., 1., 1., 3.],
    )
    .unwrap()
}

#[test]
fn managed_sparse_publication_requires_exact_sequential_writes() {
    let (runtime, ledger) = runtime_with_ledger();
    let baseline = ledger.snapshot().live_claimed_bytes;
    let mut output = SparsePublication::new(
        &runtime,
        ExactHostSparseClass::Numeric(NumericDType::F64),
        1,
        1,
        1,
    )
    .unwrap();
    assert!(matches!(&output, SparsePublication::ManagedOwned(_)));
    assert!(matches!(
        output.pointer(1, 0),
        Err(SparsePublicationError::Invalid(_))
    ));
    output.pointer(0, 0).unwrap();
    assert!(matches!(
        block_on(output.finish(&runtime)),
        Err(SparsePublicationError::Invalid(_))
    ));
    assert_eq!(ledger.snapshot().live_claimed_bytes, baseline);

    let mut output = SparsePublication::new(
        &runtime,
        ExactHostSparseClass::Numeric(NumericDType::F64),
        1,
        1,
        1,
    )
    .unwrap();
    output.pointer(0, 0).unwrap();
    output.entry(0, 0, (1.0, 0.0)).unwrap();
    assert!(matches!(
        output.entry(0, 0, (2.0, 0.0)),
        Err(SparsePublicationError::Invalid(_))
    ));
    output.pointer(1, 1).unwrap();
    let value = block_on(output.finish(&runtime)).unwrap();
    assert!(ledger.snapshot().live_claimed_bytes > baseline);
    drop(value);
    assert_eq!(ledger.snapshot().live_claimed_bytes, baseline);
}
fn dense(s: &SparseTensor) -> Vec<f64> {
    let mut d = vec![0.; s.rows * s.cols];
    for c in 0..s.cols {
        for at in s.col_ptrs[c]..s.col_ptrs[c + 1] {
            d[s.row_indices[at] * s.cols + c] = s.as_f64_slice().unwrap()[at];
        }
    }
    d
}
fn multiply(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n * n];
    for row in 0..n {
        for column in 0..n {
            result[row * n + column] = (0..n)
                .map(|inner| a[row * n + inner] * b[inner * n + column])
                .sum();
        }
    }
    result
}
fn sparse_outputs(values: &[Value]) -> Vec<Vec<f64>> {
    values
        .iter()
        .map(|value| {
            let Value::SparseTensor(value) = value else {
                panic!("sparse output")
            };
            dense(value)
        })
        .collect()
}
#[test]
fn nofill_has_exact_source_pattern_residual_and_preconditions_rhs() {
    let r = runtime();
    let outputs = block_on(evaluate_typed::<f64>(source(), Options::default(), 2, &r)).unwrap();
    let (Value::SparseTensor(l), Value::SparseTensor(u)) = (&outputs[0], &outputs[1]) else {
        panic!("sparse factors")
    };
    let a = dense(&source());
    let l = dense(l);
    let u = dense(u);
    let mut product = [0.; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                product[i * 3 + j] += l[i * 3 + k] * u[k * 3 + j];
            }
        }
    }
    for c in 0..3 {
        for at in source().col_ptrs[c]..source().col_ptrs[c + 1] {
            let row = source().row_indices[at];
            assert!((a[row * 3 + c] - product[row * 3 + c]).abs() < 1e-10);
        }
    }
    let b = [1., 2., 3.];
    let mut y = [0.; 3];
    for i in 0..3 {
        y[i] = (b[i] - (0..i).map(|j| l[i * 3 + j] * y[j]).sum::<f64>()) / l[i * 3 + i];
    }
    let mut x = [0.; 3];
    for i in (0..3).rev() {
        x[i] = (y[i] - ((i + 1)..3).map(|j| u[i * 3 + j] * x[j]).sum::<f64>()) / u[i * 3 + i];
    }
    for i in 0..3 {
        let actual = (0..3).map(|j| a[i * 3 + j] * x[j]).sum::<f64>();
        assert!((actual - b[i]).abs() < 1e-10);
    }
}

#[test]
fn nofill_does_not_reuse_a_prior_rows_source_pattern() {
    let runtime = runtime();
    // Row 3 contains lower column 2. The old dense scan popped it without
    // clearing its source-pattern bit. Row 4 contains only lower column 0 and
    // its diagonal; eliminating column 0 produces a forbidden lower-column-2
    // candidate which must not inherit row 3's structural permission.
    let source = SparseTensor::new(
        5,
        5,
        vec![0, 2, 3, 6, 7, 8],
        vec![0, 4, 1, 0, 2, 3, 3, 4],
        vec![4.0, 1.0, 4.0, 1.0, 4.0, 1.0, 4.0, 4.0],
    )
    .unwrap();
    let original = dense(&source);
    let outputs = block_on(evaluate_typed::<f64>(
        source,
        Options::default(),
        2,
        &runtime,
    ))
    .unwrap();
    let outputs = sparse_outputs(&outputs);
    assert_eq!(outputs[0][4 * 5 + 2], 0.0);
    let product = multiply(&outputs[0], &outputs[1], 5);
    for row in 0..5 {
        for column in 0..5 {
            if original[row * 5 + column] != 0.0 {
                assert!((product[row * 5 + column] - original[row * 5 + column]).abs() < 1e-12);
            }
        }
    }
}
#[test]
fn crout_and_threshold_pivoting_publish_sparse_class_and_permutation() {
    let r = runtime();
    let a = SparseTensor::new_f32(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1e-4, 1., 1., 2.],
    )
    .unwrap();
    let o = Options {
        kind: IncompleteLuKind::ThresholdPivoting,
        droptol: 1e-6,
        thresh: 1.0,
        ..Options::default()
    };
    let values = block_on(evaluate_typed::<f32>(a, o, 3, &r)).unwrap();
    assert_eq!(values.len(), 3);
    for (index, value) in values.into_iter().enumerate() {
        let Value::SparseTensor(value) = value else {
            panic!("sparse")
        };
        assert_eq!((value.rows, value.cols), (2, 2));
        assert_eq!(
            value.numeric_dtype(),
            Some(if index == 2 {
                NumericDType::F64
            } else {
                NumericDType::F32
            })
        );
    }
}

#[test]
fn complex_single_and_double_retain_native_class_and_coefficients() {
    let r = runtime();
    let single = SparseTensor::new_complex_f32(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![(2.0, 1.0), (3.0, -2.0)],
    )
    .unwrap();
    let single = block_on(evaluate_typed::<faer::c32>(
        single,
        Options::default(),
        2,
        &r,
    ))
    .unwrap();
    let Value::SparseTensor(single_upper) = &single[1] else {
        panic!("complex single upper factor")
    };
    assert_eq!(single_upper.numeric_dtype(), Some(NumericDType::F32));
    let single_values = single_upper.as_complex_f32_slice().unwrap();
    assert_eq!((single_values[0].0, single_values[0].1), (2.0, 1.0));
    assert_eq!((single_values[1].0, single_values[1].1), (3.0, -2.0));

    let double = SparseTensor::new_complex(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![(2.0, 1.0), (3.0, -2.0)],
    )
    .unwrap();
    let double = block_on(evaluate_typed::<faer::c64>(
        double,
        Options::default(),
        2,
        &r,
    ))
    .unwrap();
    let Value::SparseTensor(double_upper) = &double[1] else {
        panic!("complex double upper factor")
    };
    assert_eq!(double_upper.numeric_dtype(), Some(NumericDType::F64));
    let double_values = double_upper.as_complex_f64_slice().unwrap();
    assert_eq!((double_values[0].0, double_values[0].1), (2.0, 1.0));
    assert_eq!((double_values[1].0, double_values[1].1), (3.0, -2.0));
}

#[test]
fn pivoted_row_and_column_modes_publish_and_fold_the_documented_permutation() {
    let r = runtime();
    let source = SparseTensor::new(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![1e-4, 2.0, 4.0, 1.0, 3.0, 5.0, 6.0],
    )
    .unwrap();
    let a = dense(&source);
    for (modified, row_permutation) in [
        (ModifiedIncompleteLu::Row, false),
        (ModifiedIncompleteLu::Column, true),
    ] {
        let options = Options {
            kind: IncompleteLuKind::ThresholdPivoting,
            droptol: 0.0,
            modified,
            udiag: false,
            thresh: 1.0,
        };
        let three = block_on(evaluate_typed::<f64>(source.clone(), options, 3, &r)).unwrap();
        let three = sparse_outputs(&three);
        let lhs = if row_permutation {
            multiply(&three[2], &a, 3)
        } else {
            multiply(&a, &three[2], 3)
        };
        let rhs = multiply(&three[0], &three[1], 3);
        for (actual, expected) in lhs.iter().zip(&rhs) {
            assert!((actual - expected).abs() < 1e-10);
        }

        let two = block_on(evaluate_typed::<f64>(source.clone(), options, 2, &r)).unwrap();
        let two = sparse_outputs(&two);
        let folded = multiply(&two[0], &two[1], 3);
        for (actual, expected) in a.iter().zip(&folded) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }
}

#[test]
fn modified_threshold_dropping_preserves_the_selected_sum_invariant() {
    let runtime = runtime();
    let source = SparseTensor::new(
        4,
        4,
        vec![0, 3, 6, 9, 12],
        vec![0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3],
        vec![4.0, 0.2, 0.1, 1.0, 5.0, 0.3, 1.0, 6.0, 0.4, 0.5, 1.0, 7.0],
    )
    .unwrap();
    let a = dense(&source);
    for modified in [ModifiedIncompleteLu::Row, ModifiedIncompleteLu::Column] {
        let options = Options {
            kind: IncompleteLuKind::ThresholdPivoting,
            droptol: 0.25,
            modified,
            udiag: false,
            // This witness isolates dropping compensation from pivoting.
            thresh: 0.0,
        };
        let factors =
            block_on(evaluate_typed::<f64>(source.clone(), options, 2, &runtime)).unwrap();
        let factors = sparse_outputs(&factors);
        let product = multiply(&factors[0], &factors[1], 4);
        for index in 0..4 {
            let (expected, actual) = if modified == ModifiedIncompleteLu::Row {
                (
                    (0..4).map(|column| a[index * 4 + column]).sum::<f64>(),
                    (0..4)
                        .map(|column| product[index * 4 + column])
                        .sum::<f64>(),
                )
            } else {
                (
                    (0..4).map(|row| a[row * 4 + index]).sum::<f64>(),
                    (0..4).map(|row| product[row * 4 + index]).sum::<f64>(),
                )
            };
            assert!((actual - expected).abs() < 1e-10);
        }
    }
}

#[test]
fn row_modified_upper_drop_compensates_the_diagonal_not_a_kept_upper_entry() {
    let runtime = runtime();
    let source = SparseTensor::new(
        3,
        3,
        vec![0, 1, 3, 5],
        vec![0, 0, 1, 0, 2],
        vec![4.0, 10.0, 4.0, 0.1, 4.0],
    )
    .unwrap();
    let options = Options {
        kind: IncompleteLuKind::Crout,
        droptol: 0.5,
        modified: ModifiedIncompleteLu::Row,
        udiag: false,
        thresh: 0.0,
    };
    let outputs = block_on(evaluate_typed::<f64>(source, options, 2, &runtime)).unwrap();
    let outputs = sparse_outputs(&outputs);
    let upper = &outputs[1];
    assert!((upper[0] - 4.1).abs() < 1e-12);
    assert!((upper[1] - 10.0).abs() < 1e-12);
}
