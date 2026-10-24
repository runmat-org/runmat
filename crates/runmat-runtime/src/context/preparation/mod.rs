//! Shared preparation boundary for calls already admitted by their domain.
//!
//! No options are parsed here, no builtin is redispatched, and no backend policy
//! or timing model lives in the runtime. Controls remain in the caller's typed
//! state. An absent service preserves ordinary runtime/provider execution.
mod owned;
mod request;
mod host_resources;
mod host_numerical_output;
mod real_unary;
mod real_power;
mod real_power_scalar;
mod compound;
mod compound_resources;
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

fn execute_prepared_resources<F, Fut>(
    mut prepared: PreparedBuiltinOperands,
    execute: F,
) -> impl Future<Output = Result<RootedValue, RuntimeError>>
where
    F: FnOnce(
        Vec<Value>,
        Option<AdmittedBuiltinCompound>,
        Option<PreparedHostNumericalOutput>,
    ) -> Fut,
    Fut: Future<Output = Result<RootedValue, RuntimeError>>,
{
    let recorder = prepared.provider_recorder();
    let no_replay = prepared.forbids_gpu_gather_retry();
    runmat_accelerate_api::observation::observe_provider_future(recorder, async move {
        prepared.begin_execution();
        let compound = prepared.take_compound();
        let host_cast_output = prepared.take_host_numerical_output();
        let result = execute(prepared.take_operands(), compound, host_cast_output).await;
        let cleanup = prepared.finish(result.as_ref().map(|value| value.value()));
        let mut result = crate::runtime_error::complete_with_cleanup(result, cleanup);
        if no_replay {
            if let Err(error) = &mut result {
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            }
        }
        result
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

pub(crate) async fn execute_request_numerical<F, Fut>(
    context: Option<&RuntimeContext>,
    request: AdmittedBuiltinPreparation,
    execute: F,
) -> Result<RootedValue, RuntimeError>
where
    F: FnOnce(
        Vec<Value>,
        Option<AdmittedBuiltinCompound>,
        Option<PreparedHostNumericalOutput>,
    ) -> Fut,
    Fut: Future<Output = Result<RootedValue, RuntimeError>>,
{
    execute_request_resources(context, request, true, execute).await
}

pub(crate) async fn execute_request_host_numerical<F, Fut>(
    context: Option<&RuntimeContext>,
    request: AdmittedBuiltinPreparation,
    execute: F,
) -> Result<RootedValue, RuntimeError>
where
    F: FnOnce(Vec<Value>, Option<PreparedHostNumericalOutput>) -> Fut,
    Fut: Future<Output = Result<RootedValue, RuntimeError>>,
{
    execute_request_resources(
        context,
        request,
        true,
        |values, compound, numerical_output| async move {
            if compound.is_some() {
                return Err(missing_compound_consumer());
            }
            execute(values, numerical_output).await
        },
    )
    .await
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

async fn execute_request_resources<F, Fut>(
    context: Option<&RuntimeContext>,
    request: AdmittedBuiltinPreparation,
    consumes_host_numerical_output: bool,
    execute: F,
) -> Result<RootedValue, RuntimeError>
where
    F: FnOnce(
        Vec<Value>,
        Option<AdmittedBuiltinCompound>,
        Option<PreparedHostNumericalOutput>,
    ) -> Fut,
    Fut: Future<Output = Result<RootedValue, RuntimeError>>,
{
    match context {
        Some(context) => {
            // A finite host invocation with an actual numerical consumer
            // does not require an acceleration service. With no service,
            // admit the canonical numerical owner
            // before entering the ordinary prepared body, rather than letting
            // passthrough operands imply an unclaimed host result.
            //
            // Service-backed preparations retain their existing authority:
            // they alone choose/provide their output owner. Ordinary unbounded
            // calls keep their allocation-neutral passthrough path.
            let host_output = if consumes_host_numerical_output
                && context.service_ports().acceleration().is_none()
            {
                use super::host_allocation::{
                    host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt,
                };
                let error = |cause| {
                    host_allocation_runtime_error(
                        Some(context),
                        "RunMat:Placement:HostNumericalOutput",
                        "admitted host numerical output preparation failed",
                        cause,
                    )
                };
                match HostAllocationReceipt::policy(Some(context)).map_err(error)? {
                    HostAllocationPolicy::Managed => request
                        .prepare_host_numerical_output(context)
                        .map_err(error)?,
                    HostAllocationPolicy::Unbounded => None,
                }
            } else {
                None
            };
            execute_prepared_resources(
                prepare_admitted(context, request).await?,
                |values, compound, numerical_output| async move {
                    // host_output can exist only with no acceleration
                    // service; that passthrough has no observation/output.
                    // Refuse any contradictory delivery before body effects.
                    if host_output.is_some() && numerical_output.is_some() {
                        return Err(missing_host_numerical_consumer());
                    }
                    execute(values, compound, numerical_output.or(host_output)).await
                },
            )
            .await
        }
        None => {
            execute_prepared_resources(
                request
                    .into_operands()
                    .with_unbounded_values(PreparedBuiltinOperands::passthrough)??,
                execute,
            )
            .await
        }
    }
}
