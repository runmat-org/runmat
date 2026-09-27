//! Shared preparation boundary for calls already admitted by their domain.
//!
//! No options are parsed here, no builtin is redispatched, and no backend policy
//! or timing model lives in the runtime. Controls remain in the caller's typed
//! state. An absent service preserves ordinary runtime/provider execution.
mod owned;
mod request;
pub use owned::{BuiltinPreparationObservation, PreparedBuiltinOperands};
pub use request::{AdmittedBuiltinPreparation, BuiltinPreparationAdmission};

use super::RuntimeContext;
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
        None => PreparedBuiltinOperands::passthrough(request.into_operands()),
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
pub(crate) async fn execute_prepared<F, Fut>(
    mut prepared: PreparedBuiltinOperands,
    execute: F,
) -> Result<Value, RuntimeError>
where
    F: FnOnce(Vec<Value>) -> Fut,
    Fut: Future<Output = Result<Value, RuntimeError>>,
{
    prepared.begin_execution();
    let result = execute(prepared.take_operands()).await;
    prepared.finish(&result)?;
    result
}

pub(crate) async fn execute_admitted<F, Fut>(
    context: &RuntimeContext,
    request: AdmittedBuiltinPreparation,
    execute: F,
) -> Result<Value, RuntimeError>
where
    F: FnOnce(Vec<Value>) -> Fut,
    Fut: Future<Output = Result<Value, RuntimeError>>,
{
    execute_prepared(prepare_admitted(context, request).await?, execute).await
}

/// Transitional adapter for the existing args-only builtin ABI. S2/L23 owns
/// eliminating this single ambient lookup when bindings carry explicit context.
/// No context or session is fabricated for direct runtime calls.
pub(crate) async fn execute_current_invocation<F, Fut>(
    admission: BuiltinPreparationAdmission,
    operands: Vec<Value>,
    execute: F,
) -> Result<Value, RuntimeError>
where
    F: FnOnce(Vec<Value>) -> Fut,
    Fut: Future<Output = Result<Value, RuntimeError>>,
{
    let request = AdmittedBuiltinPreparation::new(admission, operands)?;
    execute_current_request(request, execute).await
}

pub(crate) async fn execute_current_request<F, Fut>(
    request: AdmittedBuiltinPreparation,
    execute: F,
) -> Result<Value, RuntimeError>
where
    F: FnOnce(Vec<Value>) -> Fut,
    Fut: Future<Output = Result<Value, RuntimeError>>,
{
    match super::legacy::active() {
        Some(context) => execute_admitted(&context, request, execute).await,
        None => {
            execute_prepared(
                PreparedBuiltinOperands::passthrough(request.into_operands()),
                execute,
            )
            .await
        }
    }
}
