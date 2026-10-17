//! Exact retained host outputs for finite random calls. The sampler fills the
//! claimed storage directly, so neither Single nor complex output needs a
//! second whole-result conversion buffer.

use super::{builtin_error, random};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::execution::CpuCooperationBudget;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_value::{
    ComplexElement, ComplexStorage, ComplexTensor, ExactHostComplexTensorAllocation,
    ExactHostTensorAllocation, HostComplexBuffer, NumericDType, Tensor,
};

fn allocation_error(error: HostAllocationError) -> RuntimeError {
    build_runtime_error("randn: host output allocation was not admitted")
        .with_builtin("randn")
        .with_source(error)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

fn output_size(shape: &[usize]) -> BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| builtin_error("randn: output dimensions overflow"))
    })
}

fn receipt(
    turn: &random::RandomTurn,
    requested_bytes: usize,
) -> BuiltinResult<Option<(crate::context::RuntimeContext, HostAllocationReceipt)>> {
    let Some(runtime) = turn.owner().cloned() else {
        return Ok(None);
    };
    let requested_bytes = u64::try_from(requested_bytes)
        .map_err(|_| builtin_error("randn: output allocation size overflow"))?;
    let plan = HostAllocationPlan::new(0, requested_bytes).map_err(allocation_error)?;
    Ok(HostAllocationReceipt::admit(Some(&runtime), plan)
        .map_err(allocation_error)?
        .map(|receipt| (runtime, receipt)))
}

fn native_storage<T: Clone>(len: usize, zero: T) -> BuiltinResult<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|source| {
        build_runtime_error("randn: cannot reserve normal output storage")
            .with_builtin("randn")
            .with_source(source)
            .build()
    })?;
    values.resize(len, zero);
    Ok(values)
}

async fn fill_values(
    turn: &mut random::RandomTurn,
    len: usize,
    write: impl FnMut(usize, f64) -> BuiltinResult<()>,
) -> BuiltinResult<()> {
    if let Some(runtime) = turn.owner().cloned() {
        let mut budget =
            CpuCooperationBudget::new(runtime.execution().clone(), runtime.cancellation());
        random::fill_normal_cooperative(turn, len, "randn", &runtime, &mut budget, write).await
    } else {
        random::fill_normal(turn, len, "randn", write)
    }
}

async fn fill_pairs(
    turn: &mut random::RandomTurn,
    len: usize,
    mut write: impl FnMut(usize, f64, f64) -> BuiltinResult<()>,
) -> BuiltinResult<()> {
    let draws = len
        .checked_mul(2)
        .ok_or_else(|| builtin_error("randn: complex draw count overflow"))?;
    let mut real = 0.0;
    fill_values(turn, draws, |index, value| {
        if index & 1 == 0 {
            real = value;
            Ok(())
        } else {
            write(
                index / 2,
                real * std::f64::consts::FRAC_1_SQRT_2,
                value * std::f64::consts::FRAC_1_SQRT_2,
            )
        }
    })
    .await
}

pub(super) async fn real(
    turn: &mut random::RandomTurn,
    shape: &[usize],
    dtype: NumericDType,
) -> BuiltinResult<Option<Tensor>> {
    let len = output_size(shape)?;
    let requested_bytes = ExactHostTensorAllocation::requested_bytes(dtype, shape)
        .map_err(|error| builtin_error(format!("randn: {error}")))?;
    let Some((runtime, receipt)) = receipt(turn, requested_bytes)? else {
        if dtype == NumericDType::F32 {
            let mut values = native_storage(len, 0.0_f32)?;
            fill_values(turn, len, |index, value| {
                values[index] = value as f32;
                Ok(())
            })
            .await?;
            return Tensor::from_f32(values, shape.to_vec())
                .map(Some)
                .map_err(builtin_error);
        }
        return Ok(None);
    };
    let mut output = receipt
        .with_output_allocation(&runtime, |permit| {
            ExactHostTensorAllocation::new(dtype, shape, permit)
        })
        .map_err(allocation_error)?;
    let mut budget = CpuCooperationBudget::new(runtime.execution().clone(), runtime.cancellation());
    match dtype {
        NumericDType::F64 => {
            let values = output.as_f64_slice_mut().expect("admitted double output");
            random::fill_normal_cooperative(
                turn,
                len,
                "randn",
                &runtime,
                &mut budget,
                |i, value| {
                    values[i] = value;
                    Ok(())
                },
            )
            .await?;
        }
        NumericDType::F32 => {
            let values = output.as_f32_slice_mut().expect("admitted single output");
            random::fill_normal_cooperative(
                turn,
                len,
                "randn",
                &runtime,
                &mut budget,
                |i, value| {
                    values[i] = value as f32;
                    Ok(())
                },
            )
            .await?;
        }
        _ => return Err(builtin_error("randn: output must be single or double")),
    }
    receipt.checkpoint(&runtime).map_err(allocation_error)?;
    let output = output
        .into_tensor()
        .map_err(|error| builtin_error(format!("randn: {error}")))?;
    receipt.checkpoint(&runtime).map_err(allocation_error)?;
    Ok(Some(output))
}

pub(super) async fn complex(
    turn: &mut random::RandomTurn,
    shape: &[usize],
    dtype: NumericDType,
) -> BuiltinResult<ComplexTensor> {
    let len = output_size(shape)?;
    let requested_bytes = ExactHostComplexTensorAllocation::requested_bytes(dtype, shape)
        .map_err(|error| builtin_error(format!("randn: {error}")))?;
    let Some((runtime, receipt)) = receipt(turn, requested_bytes)? else {
        return match dtype {
            NumericDType::F64 => {
                let mut values = native_storage(len, ComplexElement(0.0_f64, 0.0_f64))?;
                fill_pairs(turn, len, |index, real, imaginary| {
                    values[index] = ComplexElement(real, imaginary);
                    Ok(())
                })
                .await?;
                ComplexTensor::from_complex_storage(
                    ComplexStorage::F64(HostComplexBuffer::from_elements(values)),
                    shape.to_vec(),
                )
                .map_err(builtin_error)
            }
            NumericDType::F32 => {
                let mut values = native_storage(len, ComplexElement(0.0_f32, 0.0_f32))?;
                fill_pairs(turn, len, |index, real, imaginary| {
                    values[index] = ComplexElement(real as f32, imaginary as f32);
                    Ok(())
                })
                .await?;
                ComplexTensor::from_complex_storage(
                    ComplexStorage::F32(HostComplexBuffer::from_elements(values)),
                    shape.to_vec(),
                )
                .map_err(builtin_error)
            }
            _ => Err(builtin_error(
                "randn: complex output must be single or double",
            )),
        };
    };
    let mut output = receipt
        .with_output_allocation(&runtime, |permit| {
            ExactHostComplexTensorAllocation::new(dtype, shape, permit)
        })
        .map_err(allocation_error)?;
    fill_pairs(turn, len, |index, real, imaginary| {
        output
            .set(index, real, imaginary)
            .map_err(|error| builtin_error(format!("randn: {error}")))
    })
    .await?;
    receipt.checkpoint(&runtime).map_err(allocation_error)?;
    let output = output
        .into_tensor()
        .map_err(|error| builtin_error(format!("randn: {error}")))?;
    receipt.checkpoint(&runtime).map_err(allocation_error)?;
    Ok(output)
}
