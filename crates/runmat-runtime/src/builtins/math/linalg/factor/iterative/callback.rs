//! One independently owned language input for each iterative callback.
//!
//! A callback may retain an argument after returning, so recurrence scratch
//! cannot become its Value. Each invocation admits exactly one native vector;
//! its physical claim follows the Value if user code retains it.

use runmat_value::{NumericDType, Value};

use super::value::IterativeValue;
use crate::builtins::math::linalg::ops::callable_linear_operator::{
    RepeatedCallableVector, VectorClass, VectorContract,
};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::RuntimeError;

#[derive(Clone, Copy)]
pub(in crate::builtins::math::linalg::factor) struct CallbackErrorPolicy {
    pub name: &'static str,
    pub invalid: fn(&str) -> RuntimeError,
}

pub(in crate::builtins::math::linalg::factor) fn contract<T: IterativeValue>(
    len: usize,
    policy: CallbackErrorPolicy,
) -> Result<VectorContract, RuntimeError> {
    let class = match (T::DTYPE, T::LANES) {
        (NumericDType::F32, 1) => VectorClass::RealSingle,
        (NumericDType::F64, 1) => VectorClass::RealDouble,
        (NumericDType::F32, 2) => VectorClass::ComplexSingle,
        (NumericDType::F64, 2) => VectorClass::ComplexDouble,
        _ => return Err((policy.invalid)("unsupported callback vector class")),
    };
    Ok(VectorContract { len, class })
}

fn admission(
    runtime: &RuntimeContext,
    source: HostAllocationError,
    policy: CallbackErrorPolicy,
) -> RuntimeError {
    crate::builtin::builtin_control_error(
        Some(&runtime.cancellation_context()),
        policy.name,
        source,
    )
}

async fn owned_input<T: IterativeValue>(
    runtime: &RuntimeContext,
    source: &[T],
    policy: CallbackErrorPolicy,
) -> Result<Value, RuntimeError> {
    let retained =
        T::output_bytes(source.len()).map_err(|error| admission(runtime, error, policy))?;
    let plan =
        HostAllocationPlan::new(0, retained).map_err(|error| admission(runtime, error, policy))?;
    if let Some(receipt) = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|error| admission(runtime, error, policy))?
    {
        super::copy::admitted::<T>(&receipt, runtime, source, policy.name).await
    } else {
        super::copy::unbounded::<T>(Some(runtime), source, policy.name).await
    }
}

pub(in crate::builtins::math::linalg::factor) async fn apply<T: IterativeValue>(
    selected: &RepeatedCallableVector,
    runtime: &RuntimeContext,
    input: &[T],
    output: &mut [T],
    policy: CallbackErrorPolicy,
) -> Result<(), RuntimeError> {
    if input.len() != output.len() {
        return Err((policy.invalid)("callback vector dimensions differ"));
    }
    let argument = owned_input(runtime, input, policy).await?;
    let result = selected.apply(runtime, argument).await?;
    let [value] = result.value().as_slice() else {
        return Err((policy.invalid)("callback must return one vector"));
    };
    let values = T::input(value)
        .ok_or_else(|| (policy.invalid)("callback returned the wrong native class"))?;
    let values = values.as_slice();
    if values.len() != output.len() {
        return Err((policy.invalid)(
            "callback returned the wrong vector length",
        ));
    }
    let mut budget = runtime.cpu_cooperation();
    for (index, (destination, source)) in output.iter_mut().zip(values).enumerate() {
        if index % 1024 == 0 {
            budget
                .checkpoint((values.len() - index).min(1024) as u64)
                .await
                .map_err(|e| e.into_runtime_error_in_context(Some(runtime)))?;
            HostAllocationReceipt::policy(Some(runtime))
                .map_err(|error| admission(runtime, error, policy))?;
        }
        *destination = *source;
    }
    Ok(())
}
