//! One exact retained numeric output and one stage-backed offset workspace.

use super::{builtin_error, random, Bounds};
use crate::builtins::common::random::gpu_like::ExactHostRandomMode;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_value::{
    exact_capacity_vec_from_slice, exact_capacity_vec_with, ExactHostLogicalTensorAllocation,
    ExactHostTensorAllocation, LogicalArray, NumericDType, NumericScalar, NumericStorage, Tensor,
};

fn unbounded_offsets(len: usize) -> BuiltinResult<Vec<u64>> {
    exact_capacity_vec_with(len, |_| 0_u64).map_err(|source| {
        build_runtime_error("randi: cannot reserve native offset workspace")
            .with_builtin("randi")
            .with_source(source)
            .build()
    })
}

fn unbounded_shape(shape: &[usize]) -> BuiltinResult<Vec<usize>> {
    exact_capacity_vec_from_slice(shape).map_err(|source| {
        build_runtime_error("randi: cannot reserve output shape")
            .with_builtin("randi")
            .with_source(source)
            .build()
    })
}

fn selected_runtime(turn: &random::RandomTurn) -> BuiltinResult<crate::context::RuntimeContext> {
    turn.owner()
        .cloned()
        .ok_or_else(|| builtin_error("randi: selected exact host output lost its runtime owner"))
}

fn allocation_error(source: HostAllocationError) -> RuntimeError {
    build_runtime_error("randi: host sampling allocation was not admitted")
        .with_builtin("randi")
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

fn converted(value: i128, dtype: NumericDType) -> NumericScalar {
    match dtype {
        NumericDType::F64 => NumericScalar::F64(value as f64),
        NumericDType::F32 => NumericScalar::F32((value as f64) as f32),
        NumericDType::I8 => NumericScalar::I8(value.clamp(i8::MIN as i128, i8::MAX as i128) as i8),
        NumericDType::I16 => {
            NumericScalar::I16(value.clamp(i16::MIN as i128, i16::MAX as i128) as i16)
        }
        NumericDType::I32 => {
            NumericScalar::I32(value.clamp(i32::MIN as i128, i32::MAX as i128) as i32)
        }
        NumericDType::I64 => {
            NumericScalar::I64(value.clamp(i64::MIN as i128, i64::MAX as i128) as i64)
        }
        NumericDType::U8 => NumericScalar::U8(value.clamp(0, u8::MAX as i128) as u8),
        NumericDType::U16 => NumericScalar::U16(value.clamp(0, u16::MAX as i128) as u16),
        NumericDType::U32 => NumericScalar::U32(value.clamp(0, u32::MAX as i128) as u32),
        NumericDType::U64 => NumericScalar::U64(value.clamp(0, u64::MAX as i128) as u64),
    }
}

pub(super) fn numeric(
    turn: &mut random::RandomTurn,
    bounds: &Bounds,
    shape: &[usize],
    dtype: NumericDType,
    mode: ExactHostRandomMode,
) -> BuiltinResult<Tensor> {
    let len = shape
        .iter()
        .try_fold(1_usize, |count, &dim| count.checked_mul(dim))
        .ok_or_else(|| builtin_error("randi: output dimensions overflow"))?;
    if mode == ExactHostRandomMode::Unbounded {
        let output_shape = unbounded_shape(shape)?;
        let mut output = Tensor::from_numeric_storage(
            NumericStorage::zeros_exact(dtype, len).map_err(|source| {
                build_runtime_error("randi: cannot reserve native output")
                    .with_builtin("randi")
                    .with_source(source)
                    .build()
            })?,
            output_shape,
        )
        .map_err(builtin_error)?;
        let mut offsets = unbounded_offsets(len)?;
        random::fill_integer_offsets(turn, bounds.span, &mut offsets, "randi")?;
        for (index, offset) in offsets.into_iter().enumerate() {
            let value = bounds
                .lower
                .checked_add(i128::from(offset))
                .ok_or_else(|| builtin_error("randi: integer overflow while sampling"))?;
            output
                .set_numeric_assignment_at(index, converted(value, dtype))
                .map_err(builtin_error)?;
        }
        return Ok(output);
    }
    let runtime = selected_runtime(turn)?;
    let scratch = HostAllocationPlan::numeric_workspace_bytes(NumericDType::U64, len)
        .map_err(allocation_error)?;
    let retained =
        HostAllocationPlan::retained_tensor_bytes(dtype, shape).map_err(allocation_error)?;
    let plan = HostAllocationPlan::new(scratch, retained).map_err(allocation_error)?;
    let receipt = HostAllocationReceipt::admit(Some(&runtime), plan)
        .map_err(allocation_error)?
        .ok_or_else(|| builtin_error("randi: selected exact host output lost managed admission"))?;
    let mut output = receipt
        .with_output_allocation(&runtime, |permit| {
            ExactHostTensorAllocation::new(dtype, shape, permit)
        })
        .map_err(allocation_error)?;
    let mut offsets = receipt
        .temporary_numeric(&runtime, NumericDType::U64, len)
        .map_err(allocation_error)?;
    random::fill_integer_offsets(
        turn,
        bounds.span,
        offsets
            .as_u64_slice_mut()
            .expect("admitted offset workspace"),
        "randi",
    )?;
    receipt.checkpoint(&runtime).map_err(allocation_error)?;
    for (index, &offset) in offsets
        .as_u64_slice()
        .expect("admitted offset workspace")
        .iter()
        .enumerate()
    {
        let value = bounds
            .lower
            .checked_add(i128::from(offset))
            .ok_or_else(|| builtin_error("randi: integer overflow while sampling"))?;
        debug_assert!(value <= bounds.upper);
        output
            .set_value(index, converted(value, dtype))
            .map_err(|source| builtin_error(format!("randi: {source}")))?;
    }
    drop(offsets);
    receipt.checkpoint(&runtime).map_err(allocation_error)?;
    let output = output
        .into_tensor()
        .map_err(|source| builtin_error(format!("randi: {source}")))?;
    receipt.checkpoint(&runtime).map_err(allocation_error)?;
    Ok(output)
}

pub(super) fn logical(
    turn: &mut random::RandomTurn,
    bounds: &Bounds,
    shape: &[usize],
    mode: ExactHostRandomMode,
) -> BuiltinResult<LogicalArray> {
    let len = shape
        .iter()
        .try_fold(1_usize, |count, &dim| count.checked_mul(dim))
        .ok_or_else(|| builtin_error("randi: logical output dimensions overflow"))?;
    if mode == ExactHostRandomMode::Unbounded {
        let output_shape = unbounded_shape(shape)?;
        let mut output = exact_capacity_vec_with(len, |_| 0_u8).map_err(|source| {
            build_runtime_error("randi: cannot reserve native logical output")
                .with_builtin("randi")
                .with_source(source)
                .build()
        })?;
        let mut offsets = unbounded_offsets(len)?;
        random::fill_integer_offsets(turn, bounds.span, &mut offsets, "randi")?;
        for (index, offset) in offsets.into_iter().enumerate() {
            let value = bounds
                .lower
                .checked_add(i128::from(offset))
                .ok_or_else(|| builtin_error("randi: integer overflow while sampling"))?;
            output[index] = u8::from(value > 0);
        }
        return LogicalArray::new(output, output_shape).map_err(builtin_error);
    }
    let runtime = selected_runtime(turn)?;
    let scratch = HostAllocationPlan::numeric_workspace_bytes(NumericDType::U64, len)
        .map_err(allocation_error)?;
    let retained =
        HostAllocationPlan::retained_logical_tensor_bytes(shape).map_err(allocation_error)?;
    let plan = HostAllocationPlan::new(scratch, retained).map_err(allocation_error)?;
    let receipt = HostAllocationReceipt::admit(Some(&runtime), plan)
        .map_err(allocation_error)?
        .ok_or_else(|| builtin_error("randi: selected exact host output lost managed admission"))?;
    let mut output = receipt
        .with_output_allocation(&runtime, |permit| {
            ExactHostLogicalTensorAllocation::new(shape, permit)
        })
        .map_err(allocation_error)?;
    let mut offsets = receipt
        .temporary_numeric(&runtime, NumericDType::U64, len)
        .map_err(allocation_error)?;
    random::fill_integer_offsets(
        turn,
        bounds.span,
        offsets
            .as_u64_slice_mut()
            .expect("admitted offset workspace"),
        "randi",
    )?;
    receipt.checkpoint(&runtime).map_err(allocation_error)?;
    for (index, &offset) in offsets
        .as_u64_slice()
        .expect("admitted offset workspace")
        .iter()
        .enumerate()
    {
        let value = bounds
            .lower
            .checked_add(i128::from(offset))
            .ok_or_else(|| builtin_error("randi: integer overflow while sampling"))?;
        output.as_mut_slice()[index] = u8::from(value > 0);
    }
    drop(offsets);
    receipt.checkpoint(&runtime).map_err(allocation_error)?;
    let output = output
        .into_array()
        .map_err(|source| builtin_error(format!("randi: {source}")))?;
    receipt.checkpoint(&runtime).map_err(allocation_error)?;
    Ok(output)
}
