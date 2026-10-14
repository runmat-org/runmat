//! Shared preparation boundary for calls already admitted by their domain.
//!
//! No options are parsed here, no builtin is redispatched, and no backend policy
//! or timing model lives in the runtime. Controls remain in the caller's typed
//! state. An absent service preserves ordinary runtime/provider execution.
mod owned;
mod request;
pub use crate::dispatcher::direct_call_observation::{
    PreparedBindingAuthority, PreparedCompletionRight,
};
pub use owned::{
    BuiltinPreparationObservation, PreparedBuiltinOperands, PreparedCallCompletion,
    PreparedCallStatus,
};
pub use request::{AdmittedBuiltinPreparation, AdmittedSumReduction, BuiltinPreparationAdmission};

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
        Some(service) => service.prepare_admitted(context.clone(), request).await?,
        None => request
            .into_operands()
            .with_owned_values(PreparedBuiltinOperands::passthrough)?,
    };
    if prepared.operands().len() != expected {
        return Err(request::invalid(
            "preparation service changed the declared operand count",
        ));
    }
    Ok(prepared)
}

/// Keep ownership/observation alive across every poll of the admitted body.
/// Returned aliases are visible to the service's completion/adoption hook.
/// Dropping this future (including cancellation) records an abort, not success.
pub(crate) fn execute_prepared<F, Fut>(
    mut prepared: PreparedBuiltinOperands,
    execute: F,
) -> impl Future<Output = Result<RootedValue, RuntimeError>>
where
    F: FnOnce(Vec<Value>) -> Fut,
    Fut: Future<Output = Result<RootedValue, RuntimeError>>,
{
    let recorder = prepared.provider_recorder();
    runmat_accelerate_api::observation::observe_provider_future(recorder, async move {
        prepared.begin_execution();
        let result = execute(prepared.take_operands()).await;
        let cleanup = prepared.finish(result.as_ref().map(|value| value.value()));
        crate::runtime_error::complete_with_cleanup(result, cleanup)
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
    match context {
        Some(context) => execute_prepared(prepare_admitted(context, request).await?, execute).await,
        None => {
            execute_prepared(
                request
                    .into_operands()
                    .with_owned_values(PreparedBuiltinOperands::passthrough)?,
                execute,
            )
            .await
        }
    }
}
