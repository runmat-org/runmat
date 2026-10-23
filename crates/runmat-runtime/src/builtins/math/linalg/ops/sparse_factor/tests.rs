use std::rc::Rc;

use faer::{c32, c64};
use runmat_value::SparseTensor;

use super::{ManagedSparseLu, ManagedSparseLuError};
use crate::context::RuntimeContext;
use crate::execution::RuntimeExecutionService;
use futures::executor::block_on;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn sparse_lu_pivots_and_preserves_native_precision_across_reused_rhs() {
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    // A = [0 2; 1 3], requiring a row pivot. The factor is reused for two
    // right-hand columns without a dense coefficient conversion.
    let real = SparseTensor::new(2, 2, vec![0, 1, 3], vec![1, 0, 1], vec![1.0, 2.0, 3.0]).unwrap();
    let mut factor = block_on(ManagedSparseLu::<f64>::factor(&real, &context)).unwrap();
    let mut rhs = [10.0, 19.0, -2.0, -1.0];
    let input = rhs;
    block_on(factor.solve_into(&input, &mut rhs, 2, &context)).unwrap();
    for (actual, expected) in rhs.into_iter().zip([4.0, 5.0, 2.0, -1.0]) {
        assert!((actual - expected).abs() < 1e-12);
    }

    let single = SparseTensor::new_f32(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![4.0, 2.0, 1.0, 3.0],
    )
    .unwrap();
    let mut factor = block_on(ManagedSparseLu::<f32>::factor(&single, &context)).unwrap();
    let mut rhs = [6.0_f32, 8.0, 7.0, 1.0];
    let input = rhs;
    block_on(factor.solve_into(&input, &mut rhs, 2, &context)).unwrap();
    for (actual, expected) in rhs.into_iter().zip([1.0_f32, 2.0, 2.0, -1.0]) {
        assert!((actual - expected).abs() < 1e-5);
    }

    let complex_single = SparseTensor::new_complex_f32(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![(2.0, 1.0), (1.0, -2.0)],
    )
    .unwrap();
    let mut factor = block_on(ManagedSparseLu::<c32>::factor(&complex_single, &context)).unwrap();
    let mut rhs = [c32::new(1.0, 3.0), c32::new(0.0, -5.0)];
    let input = rhs;
    block_on(factor.solve_into(&input, &mut rhs, 1, &context)).unwrap();
    for (actual, expected) in rhs
        .into_iter()
        .zip([c32::new(1.0, 1.0), c32::new(2.0, -1.0)])
    {
        assert!((actual - expected).norm() < 1e-5);
    }

    let complex_double = SparseTensor::new_complex(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![(2.0, 1.0), (1.0, -2.0)],
    )
    .unwrap();
    let mut factor = block_on(ManagedSparseLu::<c64>::factor(&complex_double, &context)).unwrap();
    let mut rhs = [c64::new(1.0, 3.0), c64::new(0.0, -5.0)];
    let input = rhs;
    block_on(factor.solve_into(&input, &mut rhs, 1, &context)).unwrap();
    for (actual, expected) in rhs
        .into_iter()
        .zip([c64::new(1.0, 1.0), c64::new(2.0, -1.0)])
    {
        assert!((actual - expected).norm() < 1e-12);
    }

    // Structural fill is present in both columns, but the second numeric
    // pivot vanishes after elimination. The public solve must not publish a
    // seemingly successful result with NaNs from pivot division.
    let singular = SparseTensor::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    assert!(matches!(
        block_on(ManagedSparseLu::<f64>::factor(&singular, &context)),
        Err(ManagedSparseLuError::Singular { .. })
    ));

    // Constructors validate topology, not numeric finiteness. Reject these
    // coefficients before numerical factorization.
    for invalid in [
        SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![f64::NAN]).unwrap(),
        SparseTensor::new_f32(1, 1, vec![0, 1], vec![0], vec![f32::INFINITY]).unwrap(),
        SparseTensor::new_complex(1, 1, vec![0, 1], vec![0], vec![(1.0, f64::NAN)]).unwrap(),
        SparseTensor::new_complex_f32(1, 1, vec![0, 1], vec![0], vec![(f32::NEG_INFINITY, 1.0)])
            .unwrap(),
    ] {
        let rejected = if invalid.as_f32_slice().is_some() {
            matches!(
                block_on(ManagedSparseLu::<f32>::factor(&invalid, &context)),
                Err(ManagedSparseLuError::NonFiniteInput)
            )
        } else if invalid.as_f64_slice().is_some() {
            matches!(
                block_on(ManagedSparseLu::<f64>::factor(&invalid, &context)),
                Err(ManagedSparseLuError::NonFiniteInput)
            )
        } else if invalid.as_complex_f32_slice().is_some() {
            matches!(
                block_on(ManagedSparseLu::<c32>::factor(&invalid, &context)),
                Err(ManagedSparseLuError::NonFiniteInput)
            )
        } else {
            matches!(
                block_on(ManagedSparseLu::<c64>::factor(&invalid, &context)),
                Err(ManagedSparseLuError::NonFiniteInput)
            )
        };
        assert!(rejected);
    }
}
