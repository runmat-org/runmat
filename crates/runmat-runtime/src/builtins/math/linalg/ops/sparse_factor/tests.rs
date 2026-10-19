use std::rc::Rc;

use faer::{c32, c64};
use runmat_value::SparseTensor;

use super::{SparseLuError, SparseLuFactor};
use crate::context::RuntimeContext;
use crate::execution::RuntimeExecutionService;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn sparse_lu_pivots_and_preserves_native_precision_across_reused_rhs() {
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    // A = [0 2; 1 3], requiring a row pivot. The factor is reused for two
    // right-hand columns without a dense coefficient conversion.
    let real = SparseTensor::new(2, 2, vec![0, 1, 3], vec![1, 0, 1], vec![1.0, 2.0, 3.0]).unwrap();
    let factor = SparseLuFactor::factor_unbounded(&real, &context).unwrap();
    let mut rhs = [10.0, 19.0, -2.0, -1.0];
    factor.solve_real_f64(&mut rhs, 2, &context).unwrap();
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
    let factor = SparseLuFactor::factor_unbounded(&single, &context).unwrap();
    let mut rhs = [6.0_f32, 8.0, 7.0, 1.0];
    factor.solve_real_f32(&mut rhs, 2, &context).unwrap();
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
    let factor = SparseLuFactor::factor_unbounded(&complex_single, &context).unwrap();
    let mut rhs = [c32::new(1.0, 3.0), c32::new(0.0, -5.0)];
    factor.solve_complex_f32(&mut rhs, 1, &context).unwrap();
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
    let factor = SparseLuFactor::factor_unbounded(&complex_double, &context).unwrap();
    let mut rhs = [c64::new(1.0, 3.0), c64::new(0.0, -5.0)];
    factor.solve_complex_f64(&mut rhs, 1, &context).unwrap();
    for (actual, expected) in rhs
        .into_iter()
        .zip([c64::new(1.0, 1.0), c64::new(2.0, -1.0)])
    {
        assert!((actual - expected).norm() < 1e-12);
    }

    // Structural fill is present in both columns, but the second numeric
    // pivot vanishes after elimination. The public solve must not publish a
    // seemingly successful result with NaNs from faer's pivot division.
    let singular = SparseTensor::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    match SparseLuFactor::factor_unbounded(&singular, &context) {
        Err(SparseLuError::Singular { .. } | SparseLuError::NumericalBreakdown) => {}
        Ok(factor) => {
            let mut rhs = [1.0, 1.0];
            assert!(matches!(
                factor.solve_real_f64(&mut rhs, 1, &context),
                Err(SparseLuError::NumericalBreakdown)
            ));
        }
        Err(error) => panic!("unexpected sparse LU failure: {error:?}"),
    }

    // Constructors validate topology, not numeric finiteness. Reject these
    // coefficients before they reach the opaque factorization kernel.
    for invalid in [
        SparseTensor::new(1, 1, vec![0, 1], vec![0], vec![f64::NAN]).unwrap(),
        SparseTensor::new_f32(1, 1, vec![0, 1], vec![0], vec![f32::INFINITY]).unwrap(),
        SparseTensor::new_complex(1, 1, vec![0, 1], vec![0], vec![(1.0, f64::NAN)]).unwrap(),
        SparseTensor::new_complex_f32(1, 1, vec![0, 1], vec![0], vec![(f32::NEG_INFINITY, 1.0)])
            .unwrap(),
    ] {
        assert!(matches!(
            SparseLuFactor::factor_unbounded(&invalid, &context),
            Err(SparseLuError::NonFiniteInput)
        ));
    }
}
