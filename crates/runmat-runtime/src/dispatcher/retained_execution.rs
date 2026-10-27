//! One selected retained-dispatch route in the existing erased future owner.
//! Ordinary calls do not poll inactive distribution/static/constructor phases.
use super::*;
use crate::{call::BuiltinArguments, context::RuntimeContext, execution::RootedValueSequence};
use std::{future::Future, pin::Pin, rc::Rc};
mod registered;
mod unregistered;

type Execution<'a> =
    Pin<Box<dyn Future<Output = Result<Option<RootedValueSequence>, RuntimeError>> + 'a>>;

struct Request<'a> {
    // Retire the output projection before the original input/context owners,
    // just as the old function-local guard did.
    _output_guard: crate::output_count::OutputCountGuard,
    observation: Option<Rc<direct_call_observation::CallObservation>>,
    context: Option<RuntimeContext>,
    args: BuiltinArguments,
    output_count: Option<usize>,
    name: &'a str,
    policy: RetainedDispatchPolicy,
}
pub(super) fn select<'a>(
    name: &'a str,
    args: BuiltinArguments,
    output_count: Option<usize>,
    context: Option<RuntimeContext>,
    observation: Option<Rc<direct_call_observation::CallObservation>>,
    policy: RetainedDispatchPolicy,
) -> Result<Execution<'a>, RuntimeError> {
    ensure_wasm_builtins_registered();
    let name = ResolvedBuiltinName::resolve(name).canonical;
    let request = Request {
        _output_guard: crate::output_count::push_output_count(output_count)?,
        observation,
        context,
        args,
        output_count,
        name,
        policy,
    };
    // This is the same concrete carrier check as the distributed owner. Do
    // not even construct/poll its large asynchronous alternatives otherwise.
    if policy == RetainedDispatchPolicy::Normal
        && request
            .args
            .iter()
            .any(|value| matches!(value, Value::Distributed(_)))
    {
        return Ok(Box::pin(distributed(request)));
    }
    let registered = selection::select_registered_builtin(name, request.context.as_ref())?;
    Ok(match registered {
        Some(selected) => Box::pin(registered::execute(request, selected)),
        None => Box::pin(unregistered::execute(request)),
    })
}

async fn distributed(request: Request<'_>) -> Result<Option<RootedValueSequence>, RuntimeError> {
    if let Some(result) = try_distributed_builtin(
        request.name,
        &request.args,
        request.output_count,
        request.context.as_ref(),
    )
    .await?
    {
        return Ok(Some(result));
    }
    // A handle-inspection contract can decline materialization. Resolve only
    // after that result, preserving the original distributed-first ordering.
    match selection::select_registered_builtin(request.name, request.context.as_ref())? {
        Some(selected) => registered::execute(request, selected).await,
        None => unregistered::execute(request).await,
    }
}
