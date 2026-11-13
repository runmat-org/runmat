//! Project borrowed complex64 computational results into one final public
//! native-class array. The read projection is pure and does not run callbacks.
use crate::{
    context::{
        host_allocation::{
            host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
            HostAllocationReceipt,
        },
        RuntimeContext,
    },
    RuntimeError,
};
use num_complex::Complex64;
use runmat_value::{
    ComplexTensor, ExactHostComplexTensorAllocation, ExactHostNumericAllocationError,
    ExactHostTensorAllocation, NumericDType, Tensor, Value,
};

/// Preserve the two existing public classification boundaries: full eig
/// ignores imaginary NaNs when deciding realness, while eigs subset factors
/// require every imaginary component to satisfy the finite tolerance.
#[derive(Clone, Copy)]
pub(crate) enum RealProjection {
    EigFactor,
    EigsSubset,
}

pub(crate) async fn matrix(
    runtime: Option<&RuntimeContext>,
    dtype: NumericDType,
    rows: usize,
    cols: usize,
    read: impl Fn(usize) -> Complex64,
) -> Result<Value, RuntimeError> {
    matrix_with_projection(runtime, dtype, rows, cols, RealProjection::EigFactor, read).await
}

pub(crate) async fn matrix_with_projection(
    runtime: Option<&RuntimeContext>,
    dtype: NumericDType,
    rows: usize,
    cols: usize,
    projection: RealProjection,
    read: impl Fn(usize) -> Complex64,
) -> Result<Value, RuntimeError> {
    let error = |cause| {
        host_allocation_runtime_error(
            runtime,
            "RunMat:eig:Allocation",
            "dense EVD output allocation failed",
            cause,
        )
    };
    if !matches!(dtype, NumericDType::F32 | NumericDType::F64) {
        return Err(error(HostAllocationError::Numeric(
            ExactHostNumericAllocationError::UnsupportedClass,
        )));
    }
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| error(HostAllocationError::Overflow))?;
    let mut cooperation = runtime.map(RuntimeContext::cpu_cooperation);
    let owner = HostAllocationReceipt::capture_owner(runtime).map_err(error)?;
    let check = || {
        if let Some(owner) = &owner {
            owner
                .check(runtime.expect("captured managed owner"))
                .map_err(error)
        } else if HostAllocationReceipt::policy(runtime).map_err(error)?
            != crate::context::host_allocation::HostAllocationPolicy::Unbounded
        {
            Err(error(HostAllocationError::StaleAuthority))
        } else {
            Ok(())
        }
    };
    check()?;
    let mut real = true;
    for index in 0..count {
        let value = read(index);
        real &= value.im.abs() <= super::REAL_EPS
            || (matches!(projection, RealProjection::EigFactor) && value.im.is_nan());
        if index % 4096 == 4095 {
            check()?;
            if let Some(cooperation) = &mut cooperation {
                cooperation
                    .checkpoint(4096)
                    .await
                    .map_err(|e| e.into_runtime_error_in_context(runtime))?;
            }
            check()?;
        }
    }
    check()?;
    let shape = [rows, cols];
    let retained = if real {
        HostAllocationPlan::retained_tensor_bytes(dtype, &shape)
    } else {
        HostAllocationPlan::retained_complex_tensor_bytes(dtype, &shape)
    }
    .map_err(error)?;
    let receipt = HostAllocationReceipt::admit(
        runtime,
        HostAllocationPlan::new(0, retained).map_err(error)?,
    )
    .map_err(error)?;
    if let Some(receipt) = &receipt {
        let context = runtime.expect("managed publication context");
        let permit = receipt.output_permit(context).map_err(error)?;
        if real {
            let mut output = ExactHostTensorAllocation::new(dtype, &shape, &permit)
                .map_err(|e| error(e.into()))?;
            for index in 0..count {
                if index % 4096 == 0 {
                    receipt.checkpoint(context).map_err(error)?;
                }
                let value = read(index).re;
                if let Some(values) = output.as_f64_slice_mut() {
                    values[index] = value;
                } else {
                    output.as_f32_slice_mut().expect("F32 output")[index] = value as f32;
                }
                if index % 4096 == 4095 {
                    if let Some(cooperation) = &mut cooperation {
                        cooperation
                            .checkpoint(4096)
                            .await
                            .map_err(|e| e.into_runtime_error_in_context(runtime))?;
                    }
                    receipt.checkpoint(context).map_err(error)?;
                    check()?;
                }
            }
            receipt.checkpoint(context).map_err(error)?;
            let value = output
                .into_tensor()
                .map(Value::Tensor)
                .map_err(|e| error(e.into()))?;
            receipt.checkpoint(context).map_err(error)?;
            check()?;
            return Ok(value);
        }
        let mut output = ExactHostComplexTensorAllocation::new(dtype, &shape, &permit)
            .map_err(|e| error(e.into()))?;
        for index in 0..count {
            if index % 4096 == 0 {
                receipt.checkpoint(context).map_err(error)?;
            }
            let value = read(index);
            output
                .set(index, value.re, value.im)
                .map_err(|e| error(e.into()))?;
            if index % 4096 == 4095 {
                if let Some(cooperation) = &mut cooperation {
                    cooperation
                        .checkpoint(4096)
                        .await
                        .map_err(|e| e.into_runtime_error_in_context(runtime))?;
                }
                receipt.checkpoint(context).map_err(error)?;
                check()?;
            }
        }
        receipt.checkpoint(context).map_err(error)?;
        let value = output
            .into_tensor()
            .map(Value::ComplexTensor)
            .map_err(|e| error(e.into()))?;
        receipt.checkpoint(context).map_err(error)?;
        check()?;
        return Ok(value);
    }
    // A None receipt is the explicitly captured ordinary policy. These are
    // final output vectors, never temporary source/payload mirrors.
    async fn collect<T>(
        count: usize,
        runtime: Option<&RuntimeContext>,
        mut value: impl FnMut(usize) -> T,
    ) -> Result<Vec<T>, RuntimeError> {
        let error = |source| {
            host_allocation_runtime_error(
                runtime,
                "RunMat:eig:Allocation",
                "dense EVD output allocation failed",
                source,
            )
        };
        let mut output = Vec::new();
        output.try_reserve_exact(count).map_err(|_| {
            error(HostAllocationError::Numeric(
                ExactHostNumericAllocationError::AllocationFailed,
            ))
        })?;
        let mut cooperation = runtime.map(RuntimeContext::cpu_cooperation);
        for index in 0..count {
            output.push(value(index));
            if index % 4096 == 4095 {
                if HostAllocationReceipt::policy(runtime).map_err(error)?
                    != crate::context::host_allocation::HostAllocationPolicy::Unbounded
                {
                    return Err(error(HostAllocationError::StaleAuthority));
                }
                if let Some(cooperation) = &mut cooperation {
                    cooperation
                        .checkpoint(4096)
                        .await
                        .map_err(|e| e.into_runtime_error_in_context(runtime))?;
                }
                if HostAllocationReceipt::policy(runtime).map_err(error)?
                    != crate::context::host_allocation::HostAllocationPolicy::Unbounded
                {
                    return Err(error(HostAllocationError::StaleAuthority));
                }
            }
        }
        if HostAllocationReceipt::policy(runtime).map_err(error)?
            != crate::context::host_allocation::HostAllocationPolicy::Unbounded
        {
            return Err(error(HostAllocationError::StaleAuthority));
        }
        Ok(output)
    }
    let shape_error = |source| {
        crate::runtime_error::semantic_error(
            "ShapeMismatch",
            format!("dense EVD output shape: {source}"),
        )
    };
    let output = match (dtype, real) {
        (NumericDType::F32, true) => Tensor::from_f32(
            collect(count, runtime, |index| read(index).re as f32).await?,
            shape.to_vec(),
        )
        .map(Value::Tensor)
        .map_err(shape_error),
        (NumericDType::F64, true) => Tensor::new(
            collect(count, runtime, |index| read(index).re).await?,
            shape.to_vec(),
        )
        .map(Value::Tensor)
        .map_err(shape_error),
        (NumericDType::F32, false) => ComplexTensor::from_f32(
            collect(count, runtime, |index| {
                let value = read(index);
                (value.re as f32, value.im as f32)
            })
            .await?,
            shape.to_vec(),
        )
        .map(Value::ComplexTensor)
        .map_err(shape_error),
        (NumericDType::F64, false) => ComplexTensor::new(
            collect(count, runtime, |index| {
                let value = read(index);
                (value.re, value.im)
            })
            .await?,
            shape.to_vec(),
        )
        .map(Value::ComplexTensor)
        .map_err(shape_error),
        _ => unreachable!("validated floating output"),
    }?;
    check()?;
    Ok(output)
}
