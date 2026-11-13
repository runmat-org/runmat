use super::*;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
    RuntimeServicePorts,
};
use crate::execution::RuntimeExecutionService;
use futures::executor::block_on;
use runmat_value::{NumericDType, SparseTensor};
use std::rc::Rc;

struct Parallel(ManagedHostEffectLedger);
impl RuntimeParallelService for Parallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let state = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(state.free_bytes()),
            epoch: state.epoch,
            ..Default::default()
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}
fn runtime() -> (RuntimeContext, ManagedHostEffectLedger) {
    let owner = ManagedHostEffectLedger::new(1 << 20);
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(owner.clone()))),
    );
    (runtime, owner)
}

fn matrix() -> SparseTensor {
    // [4 1 0; 1 3 1; 0 1 2]
    SparseTensor::new(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0],
    )
    .unwrap()
}

#[test]
fn sparse_cholesky_retains_exact_claimed_fill_and_solves() {
    let (runtime, owner) = runtime();
    let mut factor = block_on(ManagedSparseCholesky::<f64>::factor(
        &matrix(),
        false,
        true,
        &runtime,
    ))
    .unwrap();
    assert_eq!(factor.completed_columns(), 3);
    assert_eq!(factor.permutation(), &[0, 1, 2]);
    let mut output = [0.0; 3];
    block_on(
        factor.solve_with_output(&[1.0, 2.0, 3.0], 1, &runtime, Some, |index, value| {
            output[index] = value;
            Ok(())
        }),
    )
    .unwrap();
    for (actual, expected) in
        output
            .into_iter()
            .zip([0.222222222222, 0.111111111111, 1.444444444444])
    {
        assert!((actual - expected).abs() < 1e-9);
    }
    assert!(owner.snapshot().live_claimed_bytes > 0);
    drop(factor);
    assert_eq!(owner.snapshot().live_claimed_bytes, 0);
}

#[test]
fn sparse_cholesky_preserves_partial_factor_and_failure_index() {
    let (runtime, _) = runtime();
    let sparse = SparseTensor::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0, 2.0, 2.0, 1.0],
    )
    .unwrap();
    let factor = block_on(ManagedSparseCholesky::<f64>::factor(
        &sparse, false, true, &runtime,
    ))
    .unwrap();
    assert_eq!(factor.completed_columns(), 1);
    assert_eq!(factor.lower_col_ptrs(), &[0, 2, 2]);
    assert_eq!(factor.lower_rows(), &[0, 1]);
}

#[test]
fn sparse_cholesky_preserves_single_and_complex_native_classes() {
    let (runtime, _) = runtime();
    let single = SparseTensor::new_f32(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0, 2.0]).unwrap();
    let factor = block_on(ManagedSparseCholesky::<f32>::factor(
        &single, true, true, &runtime,
    ))
    .unwrap();
    assert_eq!(factor.completed_columns(), 2);
    let complex = SparseTensor::new_complex(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![(2.0, 0.0), (2.0, 0.0)],
    )
    .unwrap();
    let factor = block_on(ManagedSparseCholesky::<faer::c64>::factor(
        &complex, false, true, &runtime,
    ))
    .unwrap();
    assert_eq!(factor.completed_columns(), 2);
    assert_eq!(single.numeric_dtype(), Some(NumericDType::F32));
}

#[test]
fn incomplete_cholesky_validates_and_scales_only_the_selected_triangle() {
    let (runtime, _) = runtime();
    let options = IncompleteCholeskyOptions {
        kind: IncompleteCholeskyKind::Threshold,
        drop_tolerance: 0.1,
        modified: false,
        diagonal_compensation: 0.0,
    };

    // Lower selection ignores the stored upper NaN entirely.
    let lower = SparseTensor::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![4.0, 2.0, f64::NAN, 3.0],
    )
    .unwrap();
    let lower = block_on(ManagedSparseCholesky::<f64>::factor_incomplete(
        &lower, options, true, &runtime,
    ))
    .unwrap();
    assert_eq!(lower.completed_columns(), 2);

    // Upper selection factors the nonconjugating layout projection. The huge
    // ignored lower coefficient must not inflate its local tail norm and drop
    // the selected upper off-diagonal.
    let upper = SparseTensor::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![4.0, 1.0e12, 2.0, 3.0],
    )
    .unwrap();
    let upper = block_on(ManagedSparseCholesky::<f64>::factor_incomplete(
        &upper, options, false, &runtime,
    ))
    .unwrap();
    assert_eq!(upper.completed_columns(), 2);
    assert_eq!(upper.lower_rows(), &[0, 1, 1]);
}
