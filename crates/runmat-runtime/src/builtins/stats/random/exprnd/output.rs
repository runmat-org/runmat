//! Direct Single/Double exponential output with a retained finite claim.

use super::{exprnd_internal_error, random, OutputPrecision, ParsedExprnd, BUILTIN_NAME};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_value::{ExactHostTensorAllocation, NumericDType, Tensor};

fn resource_error(error: HostAllocationError) -> RuntimeError {
    build_runtime_error("exprnd: host output allocation was not admitted")
        .with_builtin(BUILTIN_NAME)
        .with_source(error)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

pub(super) async fn generate(
    turn: &mut random::RandomTurn,
    parsed: &ParsedExprnd,
    len: usize,
) -> BuiltinResult<Tensor> {
    let dtype = match parsed.output_precision {
        OutputPrecision::Double => NumericDType::F64,
        OutputPrecision::Single => NumericDType::F32,
    };
    let bytes =
        HostAllocationPlan::retained_tensor_bytes(dtype, &parsed.shape).map_err(resource_error)?;
    if let Some(runtime) = turn.owner().cloned() {
        let plan = HostAllocationPlan::new(0, bytes).map_err(resource_error)?;
        if let Some(receipt) =
            HostAllocationReceipt::admit(Some(&runtime), plan).map_err(resource_error)?
        {
            let mut output = receipt
                .with_output_allocation(&runtime, |permit| {
                    ExactHostTensorAllocation::new(dtype, &parsed.shape, permit)
                })
                .map_err(resource_error)?;
            if dtype == NumericDType::F32 {
                let values = output.as_f32_slice_mut().expect("admitted single output");
                random::fill_exponential_array(
                    turn,
                    &parsed.mu,
                    len,
                    BUILTIN_NAME,
                    |index, value| {
                        values[index] = value as f32;
                        Ok(())
                    },
                )
                .await?;
            } else {
                let values = output.as_f64_slice_mut().expect("admitted double output");
                random::fill_exponential_array(
                    turn,
                    &parsed.mu,
                    len,
                    BUILTIN_NAME,
                    |index, value| {
                        values[index] = value;
                        Ok(())
                    },
                )
                .await?;
            }
            receipt.checkpoint(&runtime).map_err(resource_error)?;
            let output = output
                .into_tensor()
                .map_err(|error| exprnd_internal_error(format!("exprnd: {error}")))?;
            receipt.checkpoint(&runtime).map_err(resource_error)?;
            return Ok(output);
        }
    }

    if dtype == NumericDType::F32 {
        let mut values = Vec::new();
        values.try_reserve_exact(len).map_err(|error| {
            exprnd_internal_error(format!("exprnd: cannot reserve single output: {error}"))
        })?;
        values.resize(len, 0.0_f32);
        random::fill_exponential_array(turn, &parsed.mu, len, BUILTIN_NAME, |index, value| {
            values[index] = value as f32;
            Ok(())
        })
        .await?;
        Tensor::from_f32(values, parsed.shape.clone())
            .map_err(|error| exprnd_internal_error(format!("exprnd: {error}")))
    } else {
        let mut values = Vec::new();
        values.try_reserve_exact(len).map_err(|error| {
            exprnd_internal_error(format!("exprnd: cannot reserve double output: {error}"))
        })?;
        values.resize(len, 0.0_f64);
        random::fill_exponential_array(turn, &parsed.mu, len, BUILTIN_NAME, |index, value| {
            values[index] = value;
            Ok(())
        })
        .await?;
        Tensor::new(values, parsed.shape.clone())
            .map_err(|error| exprnd_internal_error(format!("exprnd: {error}")))
    }
}
