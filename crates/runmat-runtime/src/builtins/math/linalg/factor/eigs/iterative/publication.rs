//! Output arity and nonconvergence policy, with independently retained exact
//! payload/shape/list owners and a final original-authority fence.
use super::{request::Failure, work::Work};
use crate::{context::host_allocation::HostAllocationError, BuiltinResult};
use num_complex::Complex64;
use runmat_value::{
    ComplexTensor, ExactHostComplexTensorAllocation, ExactHostTensorAllocation, NumericDType,
    Tensor, Value,
};

enum Array {
    Real(ExactHostTensorAllocation),
    Complex(ExactHostComplexTensorAllocation),
    F32(Vec<f32>),
    F64(Vec<f64>),
    C32(Vec<(f32, f32)>),
    C64(Vec<(f64, f64)>),
}
impl Array {
    fn new(
        shape: [usize; 2],
        dtype: NumericDType,
        complex: bool,
        work: &Work<'_>,
    ) -> BuiltinResult<Self> {
        work.check()?;
        if let (Some(runtime), Some(receipt)) = (work.runtime, work.receipt) {
            let permit = receipt
                .output_permit(runtime)
                .map_err(|e| super::admission(work.runtime, e))?;
            return if complex {
                ExactHostComplexTensorAllocation::new(dtype, &shape, &permit).map(Self::Complex)
            } else {
                ExactHostTensorAllocation::new(dtype, &shape, &permit).map(Self::Real)
            }
            .map_err(|e| super::admission(work.runtime, HostAllocationError::Numeric(e)));
        }
        let len = shape[0]
            .checked_mul(shape[1])
            .ok_or_else(|| super::invalid("output shape overflow"))?;
        fn capacity<T>(len: usize) -> BuiltinResult<Vec<T>> {
            let mut v = Vec::new();
            v.try_reserve_exact(len)
                .map_err(|_| super::invalid("unable to reserve eigs output"))?;
            Ok(v)
        }
        Ok(match (dtype, complex) {
            (NumericDType::F32, false) => Self::F32(capacity(len)?),
            (NumericDType::F64, false) => Self::F64(capacity(len)?),
            (NumericDType::F32, true) => Self::C32(capacity(len)?),
            (NumericDType::F64, true) => Self::C64(capacity(len)?),
            _ => return Err(super::invalid("unsupported eigs output class")),
        })
    }
    fn set(&mut self, index: usize, value: Complex64) -> BuiltinResult<()> {
        match self {
            Self::Real(out) => {
                if let Some(values) = out.as_f64_slice_mut() {
                    values[index] = value.re;
                } else {
                    out.as_f32_slice_mut().expect("selected floating precision")[index] =
                        value.re as f32;
                }
            }
            Self::Complex(out) => out
                .set(index, value.re, value.im)
                .map_err(|e| super::admission(None, HostAllocationError::Numeric(e)))?,
            Self::F32(out) => out.push(value.re as f32),
            Self::F64(out) => out.push(value.re),
            Self::C32(out) => out.push((value.re as f32, value.im as f32)),
            Self::C64(out) => out.push((value.re, value.im)),
        }
        Ok(())
    }
    fn finish(self, shape: [usize; 2]) -> BuiltinResult<Value> {
        let exact = |e| super::admission(None, HostAllocationError::Numeric(e));
        match self {
            Self::Real(out) => out.into_tensor().map(Value::Tensor).map_err(exact),
            Self::Complex(out) => out.into_tensor().map(Value::ComplexTensor).map_err(exact),
            Self::F32(out) => Tensor::from_f32(out, shape.to_vec())
                .map(Value::Tensor)
                .map_err(|_| super::invalid("invalid real output geometry")),
            Self::F64(out) => Tensor::new(out, shape.to_vec())
                .map(Value::Tensor)
                .map_err(|_| super::invalid("invalid real output geometry")),
            Self::C32(out) => ComplexTensor::from_f32(out, shape.to_vec())
                .map(Value::ComplexTensor)
                .map_err(|_| super::invalid("invalid complex output geometry")),
            Self::C64(out) => ComplexTensor::new(out, shape.to_vec())
                .map(Value::ComplexTensor)
                .map_err(|_| super::invalid("invalid complex output geometry")),
        }
    }
}

pub(super) async fn publish(
    values: &[Complex64],
    vectors: &[Complex64],
    flags: &[usize],
    n: usize,
    dtype: NumericDType,
    failure: Failure,
    requested: Option<usize>,
    work: &mut Work<'_>,
) -> BuiltinResult<Value> {
    work.check()?;
    if requested == Some(0) {
        work.check()?;
        return Ok(Value::output_list(Vec::new()));
    }
    let mut converged_count = 0;
    for &flag in flags {
        converged_count += usize::from(flag != 0);
        work.tick().await?;
    }
    let count = if failure == Failure::Drop {
        converged_count
    } else {
        values.len()
    };
    let mut complex = false;
    for (column, value) in values.iter().enumerate() {
        if failure == Failure::Drop && flags[column] == 0 {
            continue;
        }
        complex |= value.im != 0.0;
        if requested.is_some_and(|n| n >= 2) {
            for v in &vectors[column * n..(column + 1) * n] {
                complex |= v.im != 0.0;
                work.tick().await?;
            }
        }
        work.tick().await?;
    }
    let shape = if requested.is_some_and(|n| n >= 2) {
        [count, count]
    } else {
        [count, 1]
    };
    let mut diagonal = Array::new(shape, dtype, complex, work)?;
    let mut matrix = if requested.is_some_and(|n| n >= 2) {
        Some(Array::new([n, count], dtype, complex, work)?)
    } else {
        None
    };
    let mut destination = 0;
    for (source, &value) in values.iter().enumerate() {
        if failure == Failure::Drop && flags[source] == 0 {
            continue;
        }
        let value = if failure == Failure::ReplaceNan && flags[source] == 0 {
            Complex64::new(f64::NAN, 0.0)
        } else {
            value
        };
        if let Some(matrix) = matrix.as_mut() {
            for row in 0..count {
                diagonal.set(
                    destination * count + row,
                    if row == destination {
                        value
                    } else {
                        Complex64::new(0.0, 0.0)
                    },
                )?;
                work.tick().await?;
            }
            for row in 0..n {
                matrix.set(destination * n + row, vectors[source * n + row])?;
                work.tick().await?;
            }
        } else {
            diagonal.set(destination, value)?;
            work.tick().await?;
        }
        destination += 1;
    }
    let diagonal = diagonal.finish(shape)?;
    let matrix = matrix.map(|matrix| matrix.finish([n, count])).transpose()?;
    let flag = Value::Num(if converged_count == flags.len() {
        0.0
    } else {
        1.0
    });
    let result = match requested {
        None => diagonal,
        Some(count) => {
            let values = if count == 1 {
                [Some(diagonal), None, None, None, None]
            } else {
                [matrix, Some(diagonal), Some(flag), None, None]
            };
            if let (Some(runtime), Some(receipt)) = (work.runtime, work.receipt) {
                receipt
                    .with_output_allocation(runtime, |permit| {
                        crate::builtins::math::linalg::factor::iterative::output::initialize_list(
                            count, values, permit,
                        )
                    })
                    .map_err(|e| super::admission(work.runtime, e))?
            } else {
                Value::output_list(values.into_iter().take(count).flatten().collect::<Vec<_>>())
            }
        }
    };
    work.flush().await?;
    Ok(result)
}
