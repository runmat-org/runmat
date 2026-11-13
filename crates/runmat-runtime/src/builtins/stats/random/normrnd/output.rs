//! Native-class normal output without a second floating conversion payload.

use super::{normrnd_internal_error, random};
use crate::builtins::common::tensor;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::execution::CpuCooperationBudget;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_value::{ExactHostTensorAllocation, NumericDType, Tensor, Value};

fn resource_error(error: HostAllocationError) -> RuntimeError {
    build_runtime_error("normrnd: host output allocation was not admitted")
        .with_builtin("normrnd")
        .with_source(error)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

fn output_len(shape: &[usize]) -> BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| normrnd_internal_error("normrnd: output dimensions overflow"))
    })
}

async fn fill(
    turn: &mut random::RandomTurn,
    len: usize,
    mu: f64,
    sigma: f64,
    mut write: impl FnMut(usize, f64) -> BuiltinResult<()>,
) -> BuiltinResult<()> {
    if let Some(runtime) = turn.owner().cloned() {
        let mut budget =
            CpuCooperationBudget::new(runtime.execution().clone(), runtime.cancellation());
        random::fill_normal_cooperative(
            turn,
            len,
            "normrnd",
            &runtime,
            &mut budget,
            |index, value| write(index, mu + sigma * value),
        )
        .await
    } else {
        random::fill_normal(turn, len, "normrnd", |index, value| {
            write(index, mu + sigma * value)
        })
    }
}

pub(super) async fn generate(
    turn: &mut random::RandomTurn,
    mu: f64,
    sigma: f64,
    shape: &[usize],
    single: bool,
) -> BuiltinResult<Value> {
    let len = output_len(shape)?;
    let dtype = if single {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    let bytes = HostAllocationPlan::retained_tensor_bytes(dtype, shape).map_err(resource_error)?;
    let receipt = if let Some(runtime) = turn.owner().cloned() {
        let plan = HostAllocationPlan::new(0, bytes).map_err(resource_error)?;
        HostAllocationReceipt::admit(Some(&runtime), plan)
            .map_err(resource_error)?
            .map(|receipt| (runtime, receipt))
    } else {
        None
    };
    if let Some((runtime, receipt)) = receipt {
        let mut output = receipt
            .with_output_allocation(&runtime, |permit| {
                ExactHostTensorAllocation::new(dtype, shape, permit)
            })
            .map_err(resource_error)?;
        if single {
            let values = output.as_f32_slice_mut().expect("admitted single output");
            fill(turn, len, mu, sigma, |index, value| {
                values[index] = value as f32;
                Ok(())
            })
            .await?;
        } else {
            let values = output.as_f64_slice_mut().expect("admitted double output");
            fill(turn, len, mu, sigma, |index, value| {
                values[index] = value;
                Ok(())
            })
            .await?;
        }
        receipt.checkpoint(&runtime).map_err(resource_error)?;
        let output = output
            .into_tensor()
            .map_err(|error| normrnd_internal_error(format!("normrnd: {error}")))?;
        receipt.checkpoint(&runtime).map_err(resource_error)?;
        return Ok(if single {
            Value::Tensor(output)
        } else {
            tensor::tensor_into_value(output)
        });
    }

    if single {
        let mut values = Vec::new();
        values.try_reserve_exact(len).map_err(|error| {
            normrnd_internal_error(format!("normrnd: cannot reserve single output: {error}"))
        })?;
        values.resize(len, 0.0_f32);
        fill(turn, len, mu, sigma, |index, value| {
            values[index] = value as f32;
            Ok(())
        })
        .await?;
        Tensor::from_f32(values, shape.to_vec())
            .map(Value::Tensor)
            .map_err(|error| normrnd_internal_error(format!("normrnd: {error}")))
    } else {
        let mut values = Vec::new();
        values.try_reserve_exact(len).map_err(|error| {
            normrnd_internal_error(format!("normrnd: cannot reserve double output: {error}"))
        })?;
        values.resize(len, 0.0_f64);
        fill(turn, len, mu, sigma, |index, value| {
            values[index] = value;
            Ok(())
        })
        .await?;
        Tensor::new(values, shape.to_vec())
            .map(tensor::tensor_into_value)
            .map_err(|error| normrnd_internal_error(format!("normrnd: {error}")))
    }
}
