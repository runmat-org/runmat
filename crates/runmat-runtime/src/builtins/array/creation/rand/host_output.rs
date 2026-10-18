//! One native-class output for host-stream uniform construction. Finite
//! routes retain an exact claim; unbounded routes use the same draw/fill body.

use super::{builtin_error, random};
use crate::builtins::common::random::gpu_like::ExactHostRandomMode;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_value::{
    exact_capacity_vec_from_slice, exact_capacity_vec_with, ComplexElement, ComplexStorage,
    ComplexTensor, ExactHostComplexTensorAllocation, ExactHostTensorAllocation, HostComplexBuffer,
    NumericDType, Tensor,
};

fn unbounded_storage<T: Clone>(len: usize, zero: T) -> BuiltinResult<Vec<T>> {
    exact_capacity_vec_with(len, |_| zero.clone()).map_err(|source| {
        build_runtime_error("rand: cannot reserve native output storage")
            .with_builtin("rand")
            .with_source(source)
            .build()
    })
}

fn unbounded_shape(shape: &[usize]) -> BuiltinResult<Vec<usize>> {
    exact_capacity_vec_from_slice(shape).map_err(|source| {
        build_runtime_error("rand: cannot reserve output shape")
            .with_builtin("rand")
            .with_source(source)
            .build()
    })
}

fn allocation_error(source: HostAllocationError) -> RuntimeError {
    build_runtime_error("rand: host output allocation was not admitted")
        .with_builtin("rand")
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

fn receipt(
    turn: &random::RandomTurn,
    shape: &[usize],
    dtype: NumericDType,
    complex: bool,
    mode: ExactHostRandomMode,
) -> BuiltinResult<Option<(crate::context::RuntimeContext, HostAllocationReceipt)>> {
    if mode == ExactHostRandomMode::Unbounded {
        return Ok(None);
    }
    let retained_bytes = if complex {
        HostAllocationPlan::retained_complex_tensor_bytes(dtype, shape)
    } else {
        HostAllocationPlan::retained_tensor_bytes(dtype, shape)
    }
    .map_err(allocation_error)?;
    let Some(runtime) = turn.owner().cloned() else {
        return Err(builtin_error(
            "rand: selected exact host output lost its runtime owner",
        ));
    };
    let plan = HostAllocationPlan::new(0, retained_bytes).map_err(allocation_error)?;
    let receipt = HostAllocationReceipt::admit(Some(&runtime), plan)
        .map_err(allocation_error)?
        .ok_or_else(|| builtin_error("rand: selected exact host output lost managed admission"))?;
    Ok(Some((runtime, receipt)))
}

pub(super) async fn real(
    turn: &mut random::RandomTurn,
    shape: &[usize],
    dtype: NumericDType,
    mode: ExactHostRandomMode,
) -> BuiltinResult<Tensor> {
    let len = shape
        .iter()
        .try_fold(1_usize, |count, &dim| count.checked_mul(dim))
        .ok_or_else(|| builtin_error("rand: output dimensions overflow"))?;
    let Some((runtime, receipt)) = receipt(turn, shape, dtype, false, mode)? else {
        let output_shape = unbounded_shape(shape)?;
        let owner = turn.owner().cloned();
        return match dtype {
            NumericDType::F64 => {
                let mut values = unbounded_storage(len, 0.0_f64)?;
                random::fill_uniform_scaled(
                    turn,
                    len,
                    "rand",
                    owner.as_ref(),
                    0.0,
                    1.0,
                    |i, value| {
                        values[i] = value;
                        Ok(())
                    },
                )
                .await?;
                Tensor::new(values, output_shape).map_err(builtin_error)
            }
            NumericDType::F32 => {
                let mut values = unbounded_storage(len, 0.0_f32)?;
                random::fill_uniform_scaled(
                    turn,
                    len,
                    "rand",
                    owner.as_ref(),
                    0.0,
                    1.0,
                    |i, value| {
                        values[i] = random::single_open_unit(value) as f32;
                        Ok(())
                    },
                )
                .await?;
                Tensor::from_f32(values, output_shape).map_err(builtin_error)
            }
            _ => Err(builtin_error("rand: output must be single or double")),
        };
    };
    let mut output = receipt
        .with_output_allocation(&runtime, |permit| {
            ExactHostTensorAllocation::new(dtype, shape, permit)
        })
        .map_err(allocation_error)?;
    match dtype {
        NumericDType::F64 => {
            let values = output.as_f64_slice_mut().expect("admitted double output");
            random::fill_uniform_scaled(turn, len, "rand", Some(&runtime), 0.0, 1.0, |i, value| {
                values[i] = value;
                Ok(())
            })
            .await?;
        }
        NumericDType::F32 => {
            let values = output.as_f32_slice_mut().expect("admitted single output");
            random::fill_uniform_scaled(turn, len, "rand", Some(&runtime), 0.0, 1.0, |i, value| {
                values[i] = random::single_open_unit(value) as f32;
                Ok(())
            })
            .await?;
        }
        _ => return Err(builtin_error("rand: output must be single or double")),
    }
    receipt.checkpoint(&runtime).map_err(allocation_error)?;
    let output = output
        .into_tensor()
        .map_err(|source| builtin_error(format!("rand: {source}")))?;
    receipt.checkpoint(&runtime).map_err(allocation_error)?;
    Ok(output)
}
pub(super) async fn complex(
    turn: &mut random::RandomTurn,
    shape: &[usize],
    dtype: NumericDType,
    mode: ExactHostRandomMode,
) -> BuiltinResult<ComplexTensor> {
    let len = shape
        .iter()
        .try_fold(1_usize, |count, &dim| count.checked_mul(dim))
        .ok_or_else(|| builtin_error("rand: output dimensions overflow"))?;
    let draws = len
        .checked_mul(2)
        .ok_or_else(|| builtin_error("rand: complex draw count overflow"))?;
    let Some((runtime, receipt)) = receipt(turn, shape, dtype, true, mode)? else {
        let output_shape = unbounded_shape(shape)?;
        let owner = turn.owner().cloned();
        return match dtype {
            NumericDType::F64 => {
                let mut values = unbounded_storage(len, ComplexElement(0.0_f64, 0.0_f64))?;
                let mut real = 0.0;
                random::fill_uniform_scaled(
                    turn,
                    draws,
                    "rand",
                    owner.as_ref(),
                    0.0,
                    1.0,
                    |i, value| {
                        if i & 1 == 0 {
                            real = value;
                        } else {
                            values[i / 2] = ComplexElement(real, value);
                        }
                        Ok(())
                    },
                )
                .await?;
                ComplexTensor::from_complex_storage(
                    ComplexStorage::F64(HostComplexBuffer::from_elements(values)),
                    output_shape,
                )
                .map_err(builtin_error)
            }
            NumericDType::F32 => {
                let mut values = unbounded_storage(len, ComplexElement(0.0_f32, 0.0_f32))?;
                let mut real = 0.0;
                random::fill_uniform_scaled(
                    turn,
                    draws,
                    "rand",
                    owner.as_ref(),
                    0.0,
                    1.0,
                    |i, value| {
                        if i & 1 == 0 {
                            real = value;
                        } else {
                            values[i / 2] = ComplexElement(
                                random::single_open_unit(real) as f32,
                                random::single_open_unit(value) as f32,
                            );
                        }
                        Ok(())
                    },
                )
                .await?;
                ComplexTensor::from_complex_storage(
                    ComplexStorage::F32(HostComplexBuffer::from_elements(values)),
                    output_shape,
                )
                .map_err(builtin_error)
            }
            _ => Err(builtin_error(
                "rand: complex output must be single or double",
            )),
        };
    };
    let mut output = receipt
        .with_output_allocation(&runtime, |permit| {
            ExactHostComplexTensorAllocation::new(dtype, shape, permit)
        })
        .map_err(allocation_error)?;
    let mut real = 0.0;
    random::fill_uniform_scaled(turn, draws, "rand", Some(&runtime), 0.0, 1.0, |i, value| {
        if i & 1 == 0 {
            real = value;
        } else {
            let (real, imaginary) = if dtype == NumericDType::F32 {
                (
                    random::single_open_unit(real),
                    random::single_open_unit(value),
                )
            } else {
                (real, value)
            };
            output
                .set(i / 2, real, imaginary)
                .map_err(|source| builtin_error(format!("rand: {source}")))?;
        }
        Ok(())
    })
    .await?;
    receipt.checkpoint(&runtime).map_err(allocation_error)?;
    let output = output
        .into_tensor()
        .map_err(|source| builtin_error(format!("rand: {source}")))?;
    receipt.checkpoint(&runtime).map_err(allocation_error)?;
    Ok(output)
}
