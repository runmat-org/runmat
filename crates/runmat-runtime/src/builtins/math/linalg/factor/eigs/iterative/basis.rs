//! Twice-reorthogonalized native Krylov basis and cached operator products.
use super::{operator::Operator, scalar::Scalar, work::Work};
use crate::BuiltinResult;
use num_complex::Complex64;
pub(super) mod metric;
pub(super) async fn range_seed<T: Scalar>(
    vector: &mut [T],
    operator: &mut Operator<'_, T>,
    temporary: &mut [T],
    work: &mut Work<'_>,
) -> BuiltinResult<()> {
    if !operator.needs_range_seed() {
        return Ok(());
    }
    let (output, scratch) = temporary.split_at_mut(vector.len());
    operator.apply(vector, output, scratch, work).await?;
    for (destination, &source) in vector.iter_mut().zip(output.iter()) {
        *destination = source;
        work.tick().await?;
    }
    work.flush().await
}

pub(super) fn element(value: &runmat_value::Value, index: usize) -> BuiltinResult<Complex64> {
    use runmat_value::Value;
    let real = match value {
        Value::Tensor(t) => {
            let scalar = t
                .numeric_value_at(index)
                .ok_or_else(|| super::invalid("StartVector element is absent"))?;
            if let Some(integer) = super::super::numeric_scalar_i128(scalar) {
                if !(-(1i128 << 53)..=(1i128 << 53)).contains(&integer) {
                    return Err(super::invalid(
                        "integer StartVector must be exactly representable as double",
                    ));
                }
            }
            scalar.materialize_f64()
        }
        Value::ComplexTensor(t) => {
            return Ok(crate::builtins::common::tensor::complex_tensor_value_complex64(t, index))
        }
        Value::Num(value) => *value,
        Value::Int(value) => {
            let integer = super::super::int_value_i128(value);
            if !(-(1i128 << 53)..=(1i128 << 53)).contains(&integer) {
                return Err(super::invalid(
                    "integer StartVector must be exactly representable as double",
                ));
            }
            value.to_f64()
        }
        Value::Complex(real, imag) => return Ok(Complex64::new(*real, *imag)),
        Value::Bool(value) => {
            if *value {
                1.0
            } else {
                0.0
            }
        }
        Value::LogicalArray(value) => {
            if value.data[index] != 0 {
                1.0
            } else {
                0.0
            }
        }
        _ => return Err(super::invalid("StartVector must be numeric")),
    };
    Ok(Complex64::new(real, 0.0))
}

pub(super) async fn orthogonalize<T: Scalar>(
    previous: &[T],
    vector: &mut [T],
    work: &mut Work<'_>,
) -> BuiltinResult<bool> {
    let n = vector.len();
    let mut original_scale = 0.0f64;
    for &value in vector.iter() {
        original_scale = original_scale
            .max(value.complex().re.abs())
            .max(value.complex().im.abs());
        work.tick().await?;
    }
    for _ in 0..2 {
        for column in previous.chunks_exact(n) {
            let mut dot = Complex64::new(0.0, 0.0);
            for (&q, &v) in column.iter().zip(vector.iter()) {
                dot += q.complex().conj() * v.complex();
                work.tick().await?;
            }
            for (v, &q) in vector.iter_mut().zip(column) {
                *v = T::native(v.complex() - dot * q.complex())
                    .ok_or_else(|| super::invalid("nonfinite native orthogonalization"))?;
                work.tick().await?;
            }
        }
    }
    let mut scale = 0.0f64;
    let mut squares = 1.0f64;
    for &value in vector.iter() {
        for component in [value.complex().re.abs(), value.complex().im.abs()] {
            if !component.is_finite() {
                return Err(super::invalid("operator returned a nonfinite vector"));
            }
            if component != 0.0 {
                if component > scale {
                    squares = 1.0 + squares * (scale / component).powi(2);
                    scale = component;
                } else {
                    squares += (component / scale).powi(2);
                }
            }
        }
        work.tick().await?;
    }
    let root = squares.sqrt();
    let eps = if T::DTYPE == runmat_value::NumericDType::F32 {
        f32::EPSILON as f64
    } else {
        f64::EPSILON
    };
    if scale == 0.0 || (scale / original_scale) * root <= 64.0 * eps {
        return Ok(false);
    }
    for value in vector {
        *value = T::native((value.complex() / scale) / root)
            .ok_or_else(|| super::invalid("nonfinite normalized vector"))?;
        work.tick().await?;
    }
    Ok(true)
}

pub(super) async fn initial<T: Scalar>(
    vector: &mut [T],
    start: Option<&runmat_value::Value>,
    work: &mut Work<'_>,
) -> BuiltinResult<()> {
    if let Some(start) = start {
        for (index, destination) in vector.iter_mut().enumerate() {
            *destination = T::native(element(start, index)?).ok_or_else(|| {
                super::invalid("StartVector must be finite and fit the selected class")
            })?;
            work.tick().await?;
        }
    } else {
        // Fixed private stream; neither ambient RNG, time nor pointer entropy.
        for (index, value) in vector.iter_mut().enumerate() {
            let mut bits = (index as u64).wrapping_add(0x9e3779b97f4a7c15);
            bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
            bits ^= bits >> 31;
            let real = ((bits >> 11) as f64) * (1.0 / 9007199254740992.0) - 0.5;
            *value = T::native(Complex64::new(real, 0.0))
                .ok_or_else(|| super::invalid("invalid private start vector"))?;
            work.tick().await?;
        }
    }
    if !orthogonalize::<T>(&[], vector, work).await? {
        return Err(super::invalid("StartVector must be finite and nonzero"));
    }
    Ok(())
}

pub(super) async fn expand<T: Scalar>(
    q: &mut [T],
    products: &mut [T],
    temporary: &mut [T],
    n: usize,
    mut available: usize,
    operator: &mut Operator<'_, T>,
    work: &mut Work<'_>,
) -> BuiltinResult<usize> {
    let p = q.len() / n;
    let mut unit = 0;
    for column in 0..p {
        if available == column {
            let (previous, tail) = q.split_at_mut(available * n);
            let vector = &mut tail[..n];
            loop {
                if unit == n {
                    work.flush().await?;
                    return Ok(column);
                }
                for (index, value) in vector.iter_mut().enumerate() {
                    *value = if index == unit {
                        T::native(Complex64::new(1.0, 0.0)).expect("native one")
                    } else {
                        T::zero()
                    };
                    work.tick().await?;
                }
                unit += 1;
                range_seed(vector, operator, temporary, work).await?;
                if metric::orthogonalize(previous, vector, operator, temporary, work).await? {
                    break;
                }
            }
            available += 1;
        }
        operator
            .apply(
                &q[column * n..(column + 1) * n],
                &mut products[column * n..(column + 1) * n],
                temporary,
                work,
            )
            .await?;
        if available < p {
            let (previous, tail) = q.split_at_mut(available * n);
            let vector = &mut tail[..n];
            for (destination, &source) in vector
                .iter_mut()
                .zip(&products[column * n..(column + 1) * n])
            {
                *destination = source;
                work.tick().await?;
            }
            if metric::orthogonalize(previous, vector, operator, temporary, work).await? {
                available += 1;
            }
        }
    }
    work.flush().await?;
    Ok(p)
}

pub(super) async fn restart<T: Scalar>(
    q: &mut [T],
    vectors: &[Complex64],
    n: usize,
    k: usize,
    operator: &mut Operator<'_, T>,
    temporary: &mut [T],
    work: &mut Work<'_>,
) -> BuiltinResult<usize> {
    let p = q.len() / n;
    let mut available = 0;
    for candidate in vectors.chunks_exact(n).take(k) {
        for imaginary in [false, true] {
            if imaginary && T::COMPLEX {
                continue;
            }
            if available == p - 1 {
                return Ok(available);
            }
            let (previous, tail) = q.split_at_mut(available * n);
            let vector = &mut tail[..n];
            for (destination, &source) in vector.iter_mut().zip(candidate) {
                let source = if T::COMPLEX {
                    source
                } else {
                    Complex64::new(if imaginary { source.im } else { source.re }, 0.0)
                };
                *destination = T::native(source)
                    .ok_or_else(|| super::invalid("Ritz vector exceeds native range"))?;
                work.tick().await?;
            }
            if metric::orthogonalize(previous, vector, operator, temporary, work).await? {
                available += 1;
            }
        }
    }
    if available == 0 {
        initial(&mut q[..n], None, work).await?;
        range_seed(&mut q[..n], operator, temporary, work).await?;
        if !metric::orthogonalize(&[], &mut q[..n], operator, temporary, work).await? {
            return Err(super::invalid("restart has zero B-norm"));
        }
        available = 1;
    }
    Ok(available)
}
