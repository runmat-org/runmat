//! Direct native-class uniform output, with a retained finite allocation claim.

use super::{random, unifrnd_internal_error};
use crate::builtins::common::tensor;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_value::{ExactHostTensorAllocation, NumericDType, Tensor, Value};

fn resource_error(error: HostAllocationError) -> RuntimeError {
    build_runtime_error("unifrnd: host output allocation was not admitted")
        .with_builtin("unifrnd")
        .with_source(error)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

fn output_len(shape: &[usize]) -> BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| unifrnd_internal_error("unifrnd: output dimensions overflow"))
    })
}

pub(super) async fn generate(
    turn: &mut random::RandomTurn,
    a: f64,
    b: f64,
    shape: &[usize],
) -> BuiltinResult<Value> {
    let len = output_len(shape)?;
    let bytes = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, shape)
        .map_err(resource_error)?;
    if let Some(runtime) = turn.owner().cloned() {
        let plan = HostAllocationPlan::new(0, bytes).map_err(resource_error)?;
        if let Some(receipt) =
            HostAllocationReceipt::admit(Some(&runtime), plan).map_err(resource_error)?
        {
            let mut output = receipt
                .with_output_allocation(&runtime, |permit| {
                    ExactHostTensorAllocation::new(NumericDType::F64, shape, permit)
                })
                .map_err(resource_error)?;
            let values = output.as_f64_slice_mut().expect("admitted double output");
            random::fill_uniform_scaled(
                turn,
                len,
                "unifrnd",
                Some(&runtime),
                a,
                b,
                |index, value| {
                    values[index] = value;
                    Ok(())
                },
            )
            .await?;
            receipt.checkpoint(&runtime).map_err(resource_error)?;
            let output = output
                .into_tensor()
                .map_err(|error| unifrnd_internal_error(format!("unifrnd: {error}")))?;
            receipt.checkpoint(&runtime).map_err(resource_error)?;
            return Ok(tensor::tensor_into_value(output));
        }
    }

    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        unifrnd_internal_error(format!("unifrnd: cannot reserve output: {error}"))
    })?;
    values.resize(len, 0.0_f64);
    random::fill_uniform_scaled(turn, len, "unifrnd", None, a, b, |index, value| {
        values[index] = value;
        Ok(())
    })
    .await?;
    Tensor::new(values, shape.to_vec())
        .map(tensor::tensor_into_value)
        .map_err(|error| unifrnd_internal_error(format!("unifrnd: {error}")))
}
