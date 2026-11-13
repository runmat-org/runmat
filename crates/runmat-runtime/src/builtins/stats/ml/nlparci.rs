//! Confidence intervals for nonlinear least-squares parameters.

use runmat_builtins::{
    NLPARCI_ERROR_INVALID_ARGUMENT, NLPARCI_ERROR_NUMERICAL, NLPARCI_RESIDENT_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub use runmat_builtins::NLPARCI_DESCRIPTOR;

mod admission;
mod allocation;
mod compute;
mod input;
#[cfg(test)]
mod tests;

use admission::admit;
use compute::{intervals, Cooperation};
use input::{host_input, HostInput};

const NAME: &str = "nlparci";

fn error(
    message: impl Into<String>,
    descriptor: &'static runmat_builtins::BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    error(message, &NLPARCI_ERROR_INVALID_ARGUMENT)
}

fn numerical(message: impl Into<String>) -> RuntimeError {
    error(message, &NLPARCI_ERROR_NUMERICAL)
}

#[runtime_builtin(
    name = "nlparci",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::ml::nlparci"
)]
pub(crate) async fn nlparci_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    beta: Value,
    residual: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    nlparci_in_runtime(runtime.as_ref(), beta, residual, rest).await
}

#[cfg(test)]
async fn nlparci_builtin(beta: Value, residual: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    nlparci_in_runtime(
        None,
        beta,
        residual,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
}

async fn nlparci_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    beta: Value,
    residual: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let outputs = crate::output_count::current_output_count();
    if outputs.is_some_and(|count| count > 1) {
        return Err(invalid("nlparci returns one confidence-interval matrix"));
    }
    let admitted = admit(&beta, &residual, rest)?;
    if matches!(&beta, Value::GpuTensor(_))
        || matches!(&residual, Value::GpuTensor(_))
        || matches!(&admitted.source, Value::GpuTensor(_))
        || admitted
            .alpha
            .as_ref()
            .is_some_and(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &NLPARCI_RESIDENT_INPUT_EXTENSION,
            NAME,
        )?;
    }
    // All operand shapes and classes are known before the first download.
    // A finite authority reserves the overlapping host envelope now; an
    // unbounded runtime continues through the same numerical path.
    let plan = allocation::plan(
        &beta,
        &residual,
        &admitted.source,
        admitted.alpha.as_ref(),
        admitted.method,
        outputs != Some(0),
    )?;
    let receipt = crate::context::host_allocation::HostAllocationReceipt::admit(runtime, plan)
        .map_err(allocation_error)?;
    let mut cooperation = Cooperation::new(runtime, receipt.as_ref());
    cooperation.check()?;
    let alpha = match admitted.alpha {
        Some(alpha) => alpha_value(host_input(alpha).await?)?,
        None => 0.05,
    };
    cooperation.check()?;
    let beta = host_input(beta).await?;
    cooperation.check()?;
    if floating_values(&beta).any(|value| !value.is_finite()) {
        return Err(invalid("beta must contain finite values"));
    }
    let residual = host_input(residual).await?;
    cooperation.check()?;
    if floating_values(&residual).any(|value| value.is_infinite()) {
        return Err(invalid("residuals must not contain Inf"));
    }
    if floating_values(&residual)
        .filter(|value| !value.is_nan())
        .count()
        <= beta.len()
    {
        return Err(numerical(
            "the number of complete residuals must exceed the number of coefficients",
        ));
    }
    let source = host_input(admitted.source).await?;
    cooperation.check()?;
    let result = intervals(
        &mut cooperation,
        admitted.method,
        &beta,
        &residual,
        &source,
        alpha,
        outputs != Some(0),
    )
    .await?;
    match (outputs, result) {
        (Some(0), None) => Ok(Value::output_list(Vec::new())),
        (Some(1), Some(result)) => Ok(Value::output_list(vec![result])),
        (None, Some(result)) => Ok(result),
        _ => Err(numerical(
            "interval result does not match the requested output count",
        )),
    }
}

fn allocation_error(source: crate::context::host_allocation::HostAllocationError) -> RuntimeError {
    use crate::context::host_allocation::HostAllocationError;
    let mut error = match source {
        HostAllocationError::Cancelled => {
            crate::execution::ExecutionServiceError::Cancelled.into_runtime_error()
        }
        HostAllocationError::StaleAuthority | HostAllocationError::MissingManagedOwner => {
            crate::execution::ExecutionServiceError::Infrastructure(
                "managed host allocation authority changed".into(),
            )
            .into_runtime_error()
        }
        _ => build_runtime_error(format!("nlparci: host allocation failed: {source}"))
            .with_builtin(NAME)
            .with_identifier("RunMat:nlparci:HostAllocation")
            .build(),
    };
    error.source = Some(Box::new(source));
    crate::builtins::common::map_control_flow_with_builtin(error, NAME)
}

fn floating_values(input: &HostInput) -> impl Iterator<Item = f64> + '_ {
    (0..input.len()).map(|index| input.value_f64(index))
}

fn alpha_value(input: HostInput) -> BuiltinResult<f64> {
    let alpha = input.value_f64(0);
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(invalid(
            "Alpha must be a finite scalar strictly between zero and one",
        ));
    }
    Ok(alpha)
}
