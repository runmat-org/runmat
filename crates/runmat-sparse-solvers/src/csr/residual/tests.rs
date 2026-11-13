use super::*;
use crate::scalar::{c32, c64};

fn check<T: Scalar>(values: [T; 3], x: [T; 3], rhs: T, expected: T) {
    let csr = CsrRef::try_new(1, 3, &[0, 3], &[0, 1, 2], &values).unwrap();
    let csc = CscRef::try_new(1, 3, &[0, 1, 2, 3], &[0, 0, 0], &values).unwrap();
    let mut row = [T::zero()];
    let mut column = [T::zero()];
    let mut scratch = [T::zero()];
    csr.residual(&x, &[rhs], &mut row);
    csc.residual(&x, &[rhs], &mut column, &mut scratch);
    assert_eq!(row, [expected], "CSR cancellation/product residual");
    assert_eq!(column, [expected], "CSC cancellation/product residual");
}

fn all_forms<T: Scalar>() {
    let scale = if T::IS_SINGLE {
        2.0f64.powi(25)
    } else {
        2.0f64.powi(54)
    };
    let delta = if T::IS_SINGLE {
        2.0f64.powi(-13)
    } else {
        2.0f64.powi(-27)
    };
    let small = T::from_parts(1.0, -2.0);
    check(
        [
            T::from_parts(scale, scale),
            small,
            T::from_parts(-scale, -scale),
        ],
        [T::one(); 3],
        small,
        T::zero(),
    );
    // The rounded product is one, but its exact defect is delta squared.
    check(
        [T::from_f64(1.0 + delta), T::zero(), T::zero()],
        [T::from_f64(1.0 - delta), T::zero(), T::zero()],
        T::one(),
        T::from_f64(delta * delta),
    );
    if T::IS_COMPLEX {
        // Cross-products cancel in the real component; retain the imaginary defect.
        check(
            [T::from_parts(1.0 + delta, delta), T::zero(), T::zero()],
            [T::from_parts(1.0 - delta, -delta), T::zero(), T::zero()],
            T::one(),
            T::from_parts(0.0, 2.0 * delta * delta),
        );
    }
}

#[test]
fn csr_and_csc_retain_small_residuals_in_every_scalar_class() {
    all_forms::<f32>();
    all_forms::<f64>();
    all_forms::<c32>();
    all_forms::<c64>();
}
