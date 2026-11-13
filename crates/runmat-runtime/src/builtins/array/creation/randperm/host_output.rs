//! Finite host permutation with stage-backed shuffle and exact retained result.

use super::{builtin_error, random, tensor};
use crate::builtins::common::random::gpu_like::ExactHostRandomMode;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_value::{
    exact_capacity_vec_from_slice, exact_capacity_vec_with, ExactHostTensorAllocation,
    NumericDType, Tensor, Value,
};

fn native_storage(len: usize) -> crate::BuiltinResult<Vec<f64>> {
    exact_capacity_vec_with(len, |_| 0.0_f64).map_err(|source| {
        build_runtime_error("randperm: cannot reserve native workspace")
            .with_builtin("randperm")
            .with_source(source)
            .build()
    })
}

fn native_offsets(len: usize) -> crate::BuiltinResult<Vec<u64>> {
    exact_capacity_vec_with(len, |_| 0_u64).map_err(|source| {
        build_runtime_error("randperm: cannot reserve native offsets")
            .with_builtin("randperm")
            .with_source(source)
            .build()
    })
}

fn apply_offsets(values: &mut [f64], offsets: &[u64]) -> crate::BuiltinResult<()> {
    for (index, &offset) in offsets.iter().enumerate() {
        let target = index
            .checked_add(
                usize::try_from(offset)
                    .map_err(|_| builtin_error("randperm: offset exceeds addressable extent"))?,
            )
            .ok_or_else(|| builtin_error("randperm: swap index overflow"))?;
        values.swap(index, target);
    }
    Ok(())
}

fn allocation_error(source: HostAllocationError) -> RuntimeError {
    build_runtime_error("randperm: host permutation allocation was not admitted")
        .with_builtin("randperm")
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

pub(super) fn double(
    turn: &mut random::RandomTurn,
    n: usize,
    k: usize,
    mode: ExactHostRandomMode,
) -> BuiltinResult<Value> {
    if mode == ExactHostRandomMode::Unbounded {
        let output_shape = exact_capacity_vec_from_slice(&[1, k]).map_err(|source| {
            build_runtime_error("randperm: cannot reserve output shape")
                .with_builtin("randperm")
                .with_source(source)
                .build()
        })?;
        let mut values = native_storage(n)?;
        let mut offsets = native_offsets(k)?;
        let mut output = native_storage(k)?;
        for (index, value) in values.iter_mut().enumerate() {
            *value = (index + 1) as f64;
        }
        random::fill_permutation_offsets(turn, n, &mut offsets)?;
        apply_offsets(&mut values, &offsets)?;
        output.copy_from_slice(&values[..k]);
        return Tensor::new(output, output_shape)
            .map(tensor::tensor_into_value)
            .map_err(builtin_error);
    }
    let runtime = turn.owner().cloned().ok_or_else(|| {
        builtin_error("randperm: selected exact host output lost its runtime owner")
    })?;
    let values_bytes = HostAllocationPlan::numeric_workspace_bytes(NumericDType::F64, n)
        .map_err(allocation_error)?;
    let offsets_bytes = HostAllocationPlan::numeric_workspace_bytes(NumericDType::U64, k)
        .map_err(allocation_error)?;
    let scratch = values_bytes
        .checked_add(offsets_bytes)
        .ok_or_else(|| builtin_error("randperm: workspace size overflow"))?;
    let shape = [1, k];
    let retained = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &shape)
        .map_err(allocation_error)?;
    let plan = HostAllocationPlan::new(scratch, retained).map_err(allocation_error)?;
    let receipt = HostAllocationReceipt::admit(Some(&runtime), plan)
        .map_err(allocation_error)?
        .ok_or_else(|| {
            builtin_error("randperm: selected exact host output lost managed admission")
        })?;
    let mut output = receipt
        .with_output_allocation(&runtime, |permit| {
            ExactHostTensorAllocation::new(NumericDType::F64, &shape, permit)
        })
        .map_err(allocation_error)?;
    let mut values = receipt
        .temporary_numeric(&runtime, NumericDType::F64, n)
        .map_err(allocation_error)?;
    let mut offsets = receipt
        .temporary_numeric(&runtime, NumericDType::U64, k)
        .map_err(allocation_error)?;
    for (index, value) in values
        .as_f64_slice_mut()
        .expect("admitted values workspace")
        .iter_mut()
        .enumerate()
    {
        *value = (index + 1) as f64;
    }
    random::fill_permutation_offsets(
        turn,
        n,
        offsets
            .as_u64_slice_mut()
            .expect("admitted offsets workspace"),
    )?;
    receipt.checkpoint(&runtime).map_err(allocation_error)?;
    let values_view = values
        .as_f64_slice_mut()
        .expect("admitted values workspace");
    apply_offsets(
        values_view,
        offsets.as_u64_slice().expect("admitted offsets workspace"),
    )?;
    output
        .as_f64_slice_mut()
        .expect("admitted double output")
        .copy_from_slice(&values_view[..k]);
    drop(offsets);
    drop(values);
    receipt.checkpoint(&runtime).map_err(allocation_error)?;
    let output = output
        .into_tensor()
        .map_err(|source| builtin_error(format!("randperm: {source}")))?;
    receipt.checkpoint(&runtime).map_err(allocation_error)?;
    Ok(tensor::tensor_into_value(output))
}
