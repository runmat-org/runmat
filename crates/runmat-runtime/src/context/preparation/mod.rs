//! Shared preparation boundary for calls already admitted by their domain.
//!
//! No options are parsed here, no builtin is redispatched, and no backend policy
//! or timing model lives in the runtime. Controls remain in the caller's typed
//! state. An absent service preserves ordinary runtime/provider execution.
mod execution;
mod owned;
mod request_execution;
use execution::execute_prepared_resources;
use request_execution::execute_request_resources;
mod compound;
mod compound_resources;
mod host_numerical_output;
mod host_resources;
mod real_power;
mod real_power_scalar;
mod real_unary;
mod request;
pub use crate::builtins::common::real_unary_preparation::RealUnaryPublication;
pub use crate::dispatcher::direct_call_observation::{
    PreparedBindingAuthority, PreparedCompletionRight, PreparedHostExecution,
};
pub use compound::AdmittedBuiltinCompound;
pub use host_numerical_output::PreparedHostNumericalOutput;
pub use host_resources::AdmittedHostResources;
pub use owned::{
    BuiltinPreparationObservation, PreparedBuiltinOperands, PreparedCallCompletion,
    PreparedCallStatus,
};
pub use real_power::AdmittedRealPowerCompound;
pub use real_power_scalar::AdmittedRealPowerScalarCompound;
pub use real_unary::AdmittedRealUnaryCompound;
pub use request::{
    AdmittedBuiltinPreparation, AdmittedRealUnary, AdmittedSumReduction,
    BuiltinPreparationAdmission,
};

use super::RuntimeContext;
use crate::execution::RootedValue;
use crate::RuntimeError;
use runmat_value::Value;
use std::future::Future;

#[cfg(test)]
mod tests;

pub async fn prepare_admitted(
    context: &RuntimeContext,
    request: AdmittedBuiltinPreparation,
) -> Result<PreparedBuiltinOperands, RuntimeError> {
    let expected = request.operands().len();
    let prepared = match context.service_ports().acceleration() {
        Some(service) => {
            service
                .prepare_admitted(context.clone(), request.with_selected_call(context))
                .await?
        }
        None => request
            .into_operands()
            .with_unbounded_values(PreparedBuiltinOperands::passthrough)??,
    };
    if prepared.operands().len() != expected {
        let mut error = request::invalid("preparation service changed the declared operand count");
        if prepared.forbids_gpu_gather_retry() {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        }
        return Err(error);
    }
    Ok(prepared)
}

/// Keep ownership/observation alive across every poll of the admitted body.
/// Returned aliases are visible to the service's completion/adoption hook.
/// Dropping this future (including cancellation) records an abort, not success.
#[cfg(test)]
pub(crate) fn execute_prepared<F, Fut>(
    prepared: PreparedBuiltinOperands,
    execute: F,
) -> impl Future<Output = Result<RootedValue, RuntimeError>>
where
    F: FnOnce(Vec<Value>) -> Fut,
    Fut: Future<Output = Result<RootedValue, RuntimeError>>,
{
    execute_prepared_resources(prepared, |values, compound, host_cast_output| async move {
        if compound.is_some() {
            return Err(missing_compound_consumer());
        }
        if host_cast_output.is_some() {
            return Err(missing_host_numerical_consumer());
        }
        execute(values).await
    })
}

pub(crate) async fn execute_invocation<F, Fut>(
    context: Option<&RuntimeContext>,
    admission: BuiltinPreparationAdmission,
    operands: Vec<Value>,
    execute: F,
) -> Result<RootedValue, RuntimeError>
where
    F: FnOnce(Vec<Value>) -> Fut,
    Fut: Future<Output = Result<RootedValue, RuntimeError>>,
{
    let request = AdmittedBuiltinPreparation::new(admission, operands)?;
    execute_request(context, request, execute).await
}

/// An explicit absent context means ordinary runtime execution: never consult
/// an ambient session or silently borrow another call's preparation service.
pub(crate) async fn execute_request<F, Fut>(
    context: Option<&RuntimeContext>,
    request: AdmittedBuiltinPreparation,
    execute: F,
) -> Result<RootedValue, RuntimeError>
where
    F: FnOnce(Vec<Value>) -> Fut,
    Fut: Future<Output = Result<RootedValue, RuntimeError>>,
{
    execute_request_resources(
        context,
        request,
        false,
        |values, compound, host_cast_output| async move {
            if compound.is_some() {
                return Err(missing_compound_consumer());
            }
            if host_cast_output.is_some() {
                return Err(missing_host_numerical_consumer());
            }
            execute(values).await
        },
    )
    .await
}

pub(crate) fn execute_request_numerical<'a, F, Fut>(
    context: Option<&'a RuntimeContext>,
    request: AdmittedBuiltinPreparation,
    execute: F,
) -> impl Future<Output = Result<RootedValue, RuntimeError>> + 'a
where
    F: FnOnce(
            Vec<Value>,
            Option<AdmittedBuiltinCompound>,
            Option<PreparedHostNumericalOutput>,
        ) -> Fut
        + 'a,
    Fut: Future<Output = Result<RootedValue, RuntimeError>> + 'a,
{
    execute_request_resources(context, request, true, execute)
}

pub(crate) fn execute_request_host_numerical<'a, F, Fut>(
    context: Option<&'a RuntimeContext>,
    request: AdmittedBuiltinPreparation,
    execute: F,
) -> impl Future<Output = Result<RootedValue, RuntimeError>> + 'a
where
    F: FnOnce(Vec<Value>, Option<PreparedHostNumericalOutput>) -> Fut + 'a,
    Fut: Future<Output = Result<RootedValue, RuntimeError>> + 'a,
{
    execute_request_resources(
        context,
        request,
        true,
        move |values, compound, numerical_output| {
            if compound.is_some() {
                futures::future::Either::Left(std::future::ready(Err(missing_compound_consumer())))
            } else {
                futures::future::Either::Right(execute(values, numerical_output))
            }
        },
    )
}

fn missing_compound_consumer() -> RuntimeError {
    let mut error =
        request::invalid("selected compound body requires its admitted domain consumer");
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

fn missing_host_numerical_consumer() -> RuntimeError {
    let mut error =
        request::invalid("selected host numerical route requires its exact output domain consumer");
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
