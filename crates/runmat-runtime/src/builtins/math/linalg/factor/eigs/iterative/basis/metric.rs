//! Twice-reorthogonalized B-inner-product basis. B is applied through the
//! same retained matrix/supplied-Cholesky owner as the pencil residual.
use super::{Operator, Scalar, Work};
use crate::BuiltinResult;
use num_complex::Complex64;
pub(in crate::builtins::math::linalg::factor::eigs::iterative) enum InnerNorm {
    Positive(f64),
    Null,
    NonPositive,
}
async fn inner_norm(
    n: usize,
    left: impl Fn(usize) -> Complex64,
    right: impl Fn(usize) -> Complex64,
    epsilon: f64,
    work: &mut Work<'_>,
) -> BuiltinResult<InnerNorm> {
    let mut scale = 0.0f64;
    let mut bscale = 0.0f64;
    for index in 0..n {
        let v = left(index);
        let b = right(index);
        scale = scale.max(v.re.abs()).max(v.im.abs());
        bscale = bscale.max(b.re.abs()).max(b.im.abs());
        work.tick().await?;
    }
    if !scale.is_finite() || !bscale.is_finite() {
        return Err(super::super::invalid("nonfinite B-inner-product operand"));
    }
    if scale == 0.0 || bscale == 0.0 {
        return Ok(InnerNorm::Null);
    }
    let mut dot = Complex64::new(0.0, 0.0);
    for index in 0..n {
        dot += (left(index) / scale).conj() * (right(index) / bscale);
        work.tick().await?;
    }
    if !dot.re.is_finite() || !dot.im.is_finite() {
        return Err(super::super::invalid("nonfinite B-inner-product reduction"));
    }
    if dot.im.abs() > 128.0 * epsilon * dot.re.abs() || dot.re < 0.0 {
        return Ok(InnerNorm::NonPositive);
    }
    Ok(if dot.re > 0.0 {
        InnerNorm::Positive(0.5 * (scale.ln() + bscale.ln() + dot.re.ln()))
    } else {
        InnerNorm::Null
    })
}
pub(in crate::builtins::math::linalg::factor::eigs::iterative) async fn norm_complex(
    vector: &[Complex64],
    mass: &[Complex64],
    epsilon: f64,
    work: &mut Work<'_>,
) -> BuiltinResult<InnerNorm> {
    inner_norm(vector.len(), |i| vector[i], |i| mass[i], epsilon, work).await
}
async fn norm<T: Scalar>(
    vector: &[T],
    mass: &[T],
    work: &mut Work<'_>,
) -> BuiltinResult<Option<f64>> {
    let epsilon = if T::DTYPE == runmat_value::NumericDType::F32 {
        f32::EPSILON as f64
    } else {
        f64::EPSILON
    };
    match inner_norm(
        vector.len(),
        |i| vector[i].complex(),
        |i| mass[i].complex(),
        epsilon,
        work,
    )
    .await?
    {
        InnerNorm::Positive(log) => Ok(Some(log)),
        InnerNorm::Null => Ok(None),
        InnerNorm::NonPositive => Err(super::super::invalid(
            "B is not positive semidefinite in the selected recurrence",
        )),
    }
}
pub(in crate::builtins::math::linalg::factor::eigs::iterative) async fn orthogonalize<T: Scalar>(
    previous: &[T],
    vector: &mut [T],
    operator: &Operator<'_, T>,
    temporary: &mut [T],
    work: &mut Work<'_>,
) -> BuiltinResult<bool> {
    if !operator.uses_metric() {
        return super::orthogonalize(previous, vector, work).await;
    }
    let n = vector.len();
    let (mass, scratch) = temporary.split_at_mut(n);
    let mut scale = 0.0f64;
    for &v in vector.iter() {
        scale = scale.max(v.complex().re.abs()).max(v.complex().im.abs());
        work.tick().await?;
    }
    if scale == 0.0 {
        return Ok(false);
    }
    if !scale.is_finite() {
        return Err(super::super::invalid("nonfinite basis candidate"));
    }
    for v in vector.iter_mut() {
        *v = T::native(v.complex() / scale)
            .ok_or_else(|| super::super::invalid("basis scaling exceeds native range"))?;
        work.tick().await?;
    }
    operator.apply_mass(vector, mass, scratch, work).await?;
    let Some(original) = norm(vector, mass, work).await? else {
        return Ok(false);
    };
    for pass in 0..2 {
        if pass != 0 {
            operator.apply_mass(vector, mass, scratch, work).await?;
        }
        for q in previous.chunks_exact(n) {
            let mut dot = Complex64::new(0.0, 0.0);
            for (&q, &b) in q.iter().zip(mass.iter()) {
                dot += q.complex().conj() * b.complex();
                work.tick().await?;
            }
            for (v, &q) in vector.iter_mut().zip(q) {
                *v = T::native(v.complex() - dot * q.complex())
                    .ok_or_else(|| super::super::invalid("nonfinite B-orthogonalization"))?;
                work.tick().await?;
            }
        }
    }
    operator.apply_mass(vector, mass, scratch, work).await?;
    let Some(log_norm) = norm(vector, mass, work).await? else {
        return Ok(false);
    };
    let epsilon = if T::DTYPE == runmat_value::NumericDType::F32 {
        f32::EPSILON as f64
    } else {
        f64::EPSILON
    };
    if log_norm - original <= (64.0 * epsilon).ln() {
        return Ok(false);
    }
    let divisor = log_norm.exp();
    for v in vector.iter_mut() {
        *v = T::native(v.complex() / divisor)
            .ok_or_else(|| super::super::invalid("B-normalized vector exceeds native range"))?;
        work.tick().await?;
    }
    work.flush().await?;
    Ok(true)
}
