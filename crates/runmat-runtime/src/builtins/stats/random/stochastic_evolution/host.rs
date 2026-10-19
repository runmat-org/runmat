//! Host arithmetic for one selected source iteration. Noise is already fully
//! drawn and published; this owner never draws, chooses placement, or writes
//! the VM workspace. Every intermediate follows the canonical class recipe.

use super::source_eligibility::StochasticTypedPattern;
use crate::builtins::common::random::gpu_like::ExactHostRandomMode;
use crate::builtins::common::tensor::tensor_into_value;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry, RuntimeError};
use runmat_accelerate_api::scalar_program::ScalarLiteral;
use runmat_types::{NumericClass, RealFloatingRecipe};
use runmat_value::{
    exact_capacity_vec_from_slice, exact_capacity_vec_with, ExactHostNumericAllocationError,
    ExactHostTensorAllocation, HostNumericBuffer, NumericDType, NumericScalar, Tensor, Value,
};

fn contract_error() -> RuntimeError {
    build_runtime_error("selected stochastic source operands no longer match their arithmetic plan")
        .with_identifier("RunMat:StochasticSource:InvalidOperands")
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}

fn allocation_error(context: &RuntimeContext, source: HostAllocationError) -> RuntimeError {
    build_runtime_error("stochastic state output allocation was not admitted")
        .with_identifier(format!(
            "{}:{}",
            context.error_namespace(),
            source.runtime_identifier_suffix()
        ))
        .with_source(source)
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}

fn dtype(class: NumericClass) -> BuiltinResult<NumericDType> {
    match class {
        NumericClass::Single => Ok(NumericDType::F32),
        NumericClass::Double => Ok(NumericDType::F64),
        _ => Err(contract_error()),
    }
}

fn literal(value: ScalarLiteral) -> f64 {
    match value {
        ScalarLiteral::F32Bits(bits) => f64::from(f32::from_bits(bits)),
        ScalarLiteral::F64Bits(bits) => f64::from_bits(bits),
    }
}

/// Retain only the canonical numeric allocation while the original Value is
/// moved into the source binding. No Tensor/shape clone or payload copy is
/// needed to read the just-published state or noise on the next phase.
pub(super) struct HostSourceValue<'a> {
    payload: HostSourcePayload,
    shape: &'a [usize],
}

enum HostSourcePayload {
    Scalar(f64),
    Dense(HostNumericBuffer),
}

impl<'a> HostSourceValue<'a> {
    pub(super) fn capture(
        value: &Value,
        class: NumericClass,
        shape: &'a [usize],
    ) -> BuiltinResult<Self> {
        let payload = match value {
            Value::Num(value)
                if class == NumericClass::Double
                    && !shape.is_empty()
                    && shape.iter().all(|extent| *extent == 1) =>
            {
                HostSourcePayload::Scalar(*value)
            }
            Value::Tensor(tensor)
                if tensor.numeric_dtype() == dtype(class)? && tensor.shape() == shape =>
            {
                HostSourcePayload::Dense(tensor.host_buffer().clone())
            }
            _ => return Err(contract_error()),
        };
        Ok(Self { payload, shape })
    }

    fn matches(&self, class: NumericClass, shape: &[usize]) -> BuiltinResult<bool> {
        Ok(self.shape == shape
            && match &self.payload {
                HostSourcePayload::Scalar(_) => class == NumericClass::Double,
                HostSourcePayload::Dense(buffer) => buffer.numeric_dtype() == dtype(class)?,
            })
    }

    fn element(&self, index: usize) -> BuiltinResult<f64> {
        match &self.payload {
            HostSourcePayload::Scalar(value) if index == 0 => Ok(*value),
            HostSourcePayload::Dense(buffer) => match buffer.value_at(index) {
                Some(NumericScalar::F32(value)) => Ok(f64::from(value)),
                Some(NumericScalar::F64(value)) => Ok(value),
                _ => Err(contract_error()),
            },
            _ => Err(contract_error()),
        }
    }
}

/// One element, with rounding after each represented source operation. In
/// particular Double noise does not become Single because the state is Single.
fn evolve_element(
    pattern: &StochasticTypedPattern,
    product: RealFloatingRecipe,
    state: f64,
    noise: f64,
) -> BuiltinResult<f64> {
    let scaled = pattern
        .scale_noise()
        .evaluate(literal(pattern.scale()), noise)
        .ok_or_else(contract_error)?;
    let exponent = pattern
        .drift_scaled()
        .evaluate(literal(pattern.drift()), scaled)
        .ok_or_else(contract_error)?;
    let factor = match pattern.exponential_class() {
        NumericClass::Single => f64::from((exponent as f32).exp()),
        NumericClass::Double => exponent.exp(),
        _ => return Err(contract_error()),
    };
    product.evaluate(state, factor).ok_or_else(contract_error)
}

/// Match the actual selected owner; scalar Double results have no heap payload.
pub(super) fn output_bytes(
    class: NumericClass,
    shape: &[usize],
    mode: ExactHostRandomMode,
) -> BuiltinResult<u64> {
    let count = shape
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
        .ok_or_else(contract_error)?;
    if count == 1 && class == NumericClass::Double {
        return Ok(0);
    }
    match mode {
        ExactHostRandomMode::Managed => {
            HostAllocationPlan::retained_tensor_bytes(dtype(class)?, shape)
                .map_err(|_| contract_error())
        }
        ExactHostRandomMode::Unbounded => {
            let bytes = ExactHostTensorAllocation::unmanaged_requested_bytes(dtype(class)?, shape)
                .map_err(|_| contract_error())?;
            u64::try_from(bytes).map_err(|_| contract_error())
        }
    }
}

async fn fill(
    context: &RuntimeContext,
    pattern: &StochasticTypedPattern,
    product: RealFloatingRecipe,
    state: &HostSourceValue<'_>,
    noise: &HostSourceValue<'_>,
    count: usize,
    mut write: impl FnMut(usize, f64) -> BuiltinResult<()>,
) -> BuiltinResult<()> {
    let mut cooperation = context.cpu_cooperation();
    cooperation
        .checkpoint(0)
        .await
        .map_err(|source| source.into_runtime_error())?;
    for index in 0..count {
        write(
            index,
            evolve_element(
                pattern,
                product,
                state.element(index)?,
                noise.element(index)?,
            )?,
        )?;
        cooperation
            .checkpoint(1)
            .await
            .map_err(|source| source.into_runtime_error())?;
    }
    Ok(())
}

fn native_output_error(source: ExactHostNumericAllocationError) -> RuntimeError {
    build_runtime_error("cannot reserve stochastic state output")
        .with_identifier("RunMat:StochasticSource:AllocationFailed")
        .with_source(source)
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}

fn native_storage<T: Clone>(count: usize, zero: T) -> BuiltinResult<Vec<T>> {
    exact_capacity_vec_with(count, |_| zero.clone()).map_err(native_output_error)
}

pub(super) async fn compute_state(
    context: &RuntimeContext,
    state: &HostSourceValue<'_>,
    noise: &HostSourceValue<'_>,
    pattern: &StochasticTypedPattern,
    step: u32,
) -> BuiltinResult<Value> {
    if step >= pattern.steps() {
        return Err(contract_error());
    }
    let state_class = if step == 0 {
        pattern.state_class()
    } else {
        pattern.result_class()
    };
    let product = if step == 0 {
        pattern.first_state_factor()
    } else {
        pattern.continuing_state_factor()
    };
    if !state.matches(state_class, pattern.shape())?
        || !noise.matches(pattern.noise_class(), pattern.shape())?
    {
        return Err(contract_error());
    }
    let count = pattern
        .shape()
        .iter()
        .try_fold(1usize, |n, dim| n.checked_mul(*dim))
        .ok_or_else(contract_error)?;
    let output_dtype = dtype(pattern.result_class())?;
    if count == 1 && output_dtype == NumericDType::F64 {
        let mut result = 0.0;
        fill(
            context,
            pattern,
            product,
            state,
            noise,
            count,
            |_, value| {
                result = value;
                Ok(())
            },
        )
        .await?;
        return Ok(Value::Num(result));
    }
    let plan = HostAllocationPlan::new(
        0,
        output_bytes(
            pattern.result_class(),
            pattern.shape(),
            ExactHostRandomMode::Managed,
        )?,
    )
    .map_err(|source| allocation_error(context, source))?;
    let receipt = HostAllocationReceipt::admit(Some(context), plan)
        .map_err(|source| allocation_error(context, source))?;
    let output = if let Some(receipt) = receipt.as_ref() {
        let mut output = receipt
            .exact_output_tensor(context, output_dtype, pattern.shape())
            .map_err(|source| allocation_error(context, source))?;
        fill(
            context,
            pattern,
            product,
            state,
            noise,
            count,
            |index, value| {
                let scalar = if output_dtype == NumericDType::F32 {
                    NumericScalar::F32(value as f32)
                } else {
                    NumericScalar::F64(value)
                };
                output
                    .set_value(index, scalar)
                    .map_err(|_| contract_error())
            },
        )
        .await?;
        receipt
            .checkpoint(context)
            .map_err(|source| allocation_error(context, source))?;
        output.into_tensor().map_err(|_| contract_error())?
    } else if output_dtype == NumericDType::F32 {
        let mut values = native_storage(count, 0.0_f32)?;
        fill(
            context,
            pattern,
            product,
            state,
            noise,
            count,
            |index, value| {
                values[index] = value as f32;
                Ok(())
            },
        )
        .await?;
        let shape = exact_capacity_vec_from_slice(pattern.shape()).map_err(native_output_error)?;
        Tensor::from_f32(values, shape).map_err(|_| contract_error())?
    } else {
        let mut values = native_storage(count, 0.0_f64)?;
        fill(
            context,
            pattern,
            product,
            state,
            noise,
            count,
            |index, value| {
                values[index] = value;
                Ok(())
            },
        )
        .await?;
        let shape = exact_capacity_vec_from_slice(pattern.shape()).map_err(native_output_error)?;
        Tensor::new(values, shape).map_err(|_| contract_error())?
    };
    Ok(tensor_into_value(output))
}
