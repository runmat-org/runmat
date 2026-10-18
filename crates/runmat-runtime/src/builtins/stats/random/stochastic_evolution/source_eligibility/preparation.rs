//! Exact request-local preparation for the selected stochastic source loop.
//! No source control is evaluated or copied before finite host admission.

use super::{
    positive_stochastic_source_steps, recipe_facts, scalar_literal, state_description,
    StochasticSourceControls, StochasticTypedPattern, ValueControlRef,
};
use crate::builtins::array::creation::randn::source_controls::describe_borrowed_source;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry, RuntimeError};
use runmat_accelerate_api::scalar_program::ScalarLiteral;
use runmat_types::{NumericClass, StorageFact};
use runmat_value::{NumericDType, Value};

/// Holds the request's COW state, exact arithmetic description, and finite
/// metadata reservation until the selected execution route has retired.
pub struct StochasticPreparedSource {
    state: Value,
    pattern: StochasticTypedPattern,
    receipt: Option<HostAllocationReceipt>,
}

impl StochasticPreparedSource {
    pub fn state(&self) -> &Value {
        &self.state
    }

    pub fn pattern(&self) -> &StochasticTypedPattern {
        &self.pattern
    }

    pub fn receipt(&self) -> Option<&HostAllocationReceipt> {
        self.receipt.as_ref()
    }

    pub fn parts_mut(
        &mut self,
    ) -> (
        &mut Value,
        &StochasticTypedPattern,
        Option<&HostAllocationReceipt>,
    ) {
        (&mut self.state, &self.pattern, self.receipt.as_ref())
    }

    pub(crate) fn checkpoint_receipt(
        receipt: Option<&HostAllocationReceipt>,
        context: &RuntimeContext,
    ) -> BuiltinResult<()> {
        if let Some(receipt) = receipt {
            receipt
                .checkpoint(context)
                .map_err(|error| preparation_error(context, error))?;
        } else if context.is_cancelled() {
            return Err(cancellation_error(context));
        }
        Ok(())
    }
}

fn preparation_error(context: &RuntimeContext, error: HostAllocationError) -> RuntimeError {
    let identifier = format!(
        "{}:{}",
        context.error_namespace(),
        error.runtime_identifier_suffix()
    );
    build_runtime_error(format!("stochastic source preparation: {error}"))
        .with_builtin("stochastic_evolution")
        .with_identifier(identifier)
        .with_source(error)
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}

fn cancellation_error(context: &RuntimeContext) -> RuntimeError {
    preparation_error(context, HostAllocationError::Cancelled)
}

fn step_count(control: ValueControlRef<'_>) -> Option<u32> {
    match control {
        ValueControlRef::Borrowed(value) => positive_stochastic_source_steps(value),
        ValueControlRef::Scalar(ScalarLiteral::F64Bits(bits)) => {
            let value = f64::from_bits(bits);
            (value.is_finite()
                && value > 0.0
                && value.fract() == 0.0
                && value <= f64::from(u32::MAX))
            .then_some(value as u32)
        }
        _ => None,
    }
}

fn scalar(control: ValueControlRef<'_>) -> Option<ScalarLiteral> {
    match control {
        ValueControlRef::Borrowed(value) => scalar_literal(value),
        ValueControlRef::Scalar(value) => Some(value),
        _ => None,
    }
}

fn checked_shape_bytes(rank: usize) -> Result<u64, HostAllocationError> {
    rank.checked_mul(std::mem::size_of::<usize>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(HostAllocationError::Overflow)
}

/// This is synchronous: constructing the preparation future cannot allocate
/// an unquoted boxed parser state before the first poll. Unsupported source
/// forms decline to the original For before any RNG or provider effect.
pub fn prepare_stochastic_source(
    context: &RuntimeContext,
    controls: StochasticSourceControls<'_>,
) -> BuiltinResult<Option<StochasticPreparedSource>> {
    if context.is_cancelled() {
        return Err(cancellation_error(context));
    }
    let Some(steps) = step_count(controls.steps) else {
        return Ok(None);
    };
    let Some((state_shape, state_class, state_storage)) = state_description(controls.state) else {
        return Ok(None);
    };
    if state_shape.is_empty() {
        return Ok(None);
    }
    let Some(drift) = scalar(controls.drift) else {
        return Ok(None);
    };
    let Some(scale) = scalar(controls.scale) else {
        return Ok(None);
    };
    let noise = describe_borrowed_source(
        context,
        controls.randn_arg_count,
        controls.randn_arg,
        state_shape,
    );
    if context.is_cancelled() {
        return Err(cancellation_error(context));
    }
    let Some(noise) = noise else {
        return Ok(None);
    };
    if noise.complex {
        return Ok(None);
    }
    let noise_class = match noise.dtype {
        NumericDType::F32 => NumericClass::Single,
        NumericDType::F64 => NumericClass::Double,
        _ => return Ok(None),
    };
    let Some(recipes) = recipe_facts(state_class, noise_class, drift, scale) else {
        return Ok(None);
    };
    let Some(len) = state_shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
    else {
        return Ok(None);
    };
    if context.is_cancelled() {
        return Err(cancellation_error(context));
    }

    // The parser streams controls without temporary ownership. The semantic
    // plan owns one rank-sized Vec; a dense Tensor COW clone owns another.
    // Scalar state and shared GPU handles clone without a heap allocation.
    let shape_bytes = checked_shape_bytes(state_shape.len())
        .map_err(|error| preparation_error(context, error))?;
    let state_shape_bytes = match controls.state {
        Value::Tensor(_) => shape_bytes,
        _ => 0,
    };
    let retained = shape_bytes
        .checked_add(state_shape_bytes)
        .ok_or(HostAllocationError::Overflow)
        .map_err(|error| preparation_error(context, error))?;
    let plan =
        HostAllocationPlan::new(0, retained).map_err(|error| preparation_error(context, error))?;
    let receipt = match HostAllocationReceipt::admit(Some(context), plan) {
        Ok(receipt) => receipt,
        Err(error) if error.is_capacity_denial() => return Ok(None),
        Err(error) => return Err(preparation_error(context, error)),
    };
    if let Some(receipt) = receipt.as_ref() {
        receipt
            .checkpoint(context)
            .map_err(|error| preparation_error(context, error))?;
    }

    let state = controls.state.clone();
    let shape = state_shape.to_vec();
    let noise_storage = if noise_class == NumericClass::Double && len == 1 {
        StorageFact::Scalar
    } else {
        StorageFact::Dense
    };
    let pattern = StochasticTypedPattern {
        shape,
        state_class,
        state_storage,
        noise_class,
        noise_storage,
        drift,
        scale,
        scale_noise: recipes.scale_noise,
        drift_scaled: recipes.drift_scaled,
        first_state_factor: recipes.first_state_factor,
        continuing_state_factor: recipes.continuing_state_factor,
        steps,
        last_counter: f64::from(steps),
    };
    if let Some(receipt) = receipt.as_ref() {
        receipt
            .checkpoint(context)
            .map_err(|error| preparation_error(context, error))?;
    } else if context.is_cancelled() {
        return Err(cancellation_error(context));
    }
    Ok(Some(StochasticPreparedSource {
        state,
        pattern,
        receipt,
    }))
}
