//! Ritz extraction and independent cached-product residuals. No callback is
//! replayed merely to test convergence; A*Q is the actual product transcript.
use super::{super::Sigma, scalar::Scalar, work::Work};
use crate::BuiltinResult;
use num_complex::Complex64;

pub(super) async fn project<T: Scalar>(
    q: &[T],
    products: &[T],
    matrix: &mut [Complex64],
    n: usize,
    operator: &super::operator::Operator<'_, T>,
    temporary: &mut [T],
    work: &mut Work<'_>,
) -> BuiltinResult<()> {
    let p = q.len() / n;
    for column in 0..p {
        let product = &products[column * n..(column + 1) * n];
        let (mass, scratch) = temporary.split_at_mut(n);
        let product = if operator.uses_metric() {
            operator.apply_mass(product, mass, scratch, work).await?;
            &*mass
        } else {
            product
        };
        for row in 0..p {
            let mut dot = Complex64::new(0.0, 0.0);
            for (&left, &right) in q[row * n..(row + 1) * n].iter().zip(product) {
                dot += left.complex().conj() * right.complex();
                work.tick().await?;
            }
            matrix[column * p + row] = dot;
        }
    }
    work.flush().await
}
pub(super) fn transformed(value: Complex64, sigma: Sigma) -> Complex64 {
    use crate::builtins::math::linalg::ops::linear_operator::scalar::LinearOperatorScalar;
    use runmat_value::ComplexElement;
    let inverse = || {
        <ComplexElement<f64> as LinearOperatorScalar>::divide_finite(
            ComplexElement(1.0, 0.0),
            ComplexElement(value.re, value.im),
        )
        .map_or_else(
            || {
                if value == Complex64::new(0.0, 0.0) {
                    Complex64::new(f64::INFINITY, 0.0)
                } else {
                    Complex64::new(f64::NAN, f64::NAN)
                }
            },
            |v| Complex64::new(v.0, v.1),
        )
    };
    match sigma {
        Sigma::SmallestAbs => inverse(),
        Sigma::Near(shift) => shift + inverse(),
        _ => value,
    }
}

pub(super) async fn extract<T: Scalar>(
    q: &[T],
    products: &[T],
    eigenvectors: &[Complex64],
    theta: &[Complex64],
    selected: &[usize],
    vectors: &mut [Complex64],
    flags: &mut [usize],
    n: usize,
    tolerance: f64,
    work: &mut Work<'_>,
) -> BuiltinResult<bool> {
    let p = theta.len();
    let mut all = true;
    for (slot, &selected) in selected.iter().enumerate() {
        let vector = &mut vectors[slot * n..(slot + 1) * n];
        let mut residual = super::residual_norm::Norm::new();
        let mut action_norm = super::residual_norm::Norm::new();
        let mut vector_norm = super::residual_norm::Norm::new();
        let mut norm = 0.0f64;
        for row in 0..n {
            let mut v = Complex64::new(0.0, 0.0);
            let mut av = v;
            for column in 0..p {
                let coefficient = eigenvectors[selected * p + column];
                v += q[column * n + row].complex() * coefficient;
                av += products[column * n + row].complex() * coefficient;
                work.tick().await?;
            }
            vector[row] = v;
            vector_norm.add(v);
            action_norm.add(av);
            residual.add(av - theta[selected] * v);
            norm = norm.hypot(v.norm());
        }
        let converged = norm > 0.0
            && super::residual_norm::converged(
                residual,
                action_norm,
                theta[selected],
                vector_norm,
                tolerance,
            );
        flags[slot] = usize::from(converged);
        all &= converged;
        if norm == 0.0 || !norm.is_finite() {
            return Err(super::invalid("projected eigenvector is nonfinite or zero"));
        }
        let mut largest = 0;
        for i in 1..n {
            if vector[i].norm() > vector[largest].norm() {
                largest = i;
            }
            work.tick().await?;
        }
        let phase = vector[largest].conj() / vector[largest].norm();
        for v in vector {
            *v = (*v / norm) * phase;
            work.tick().await?;
        }
    }
    work.flush().await?;
    Ok(all)
}
