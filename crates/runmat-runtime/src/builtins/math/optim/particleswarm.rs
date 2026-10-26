//! Adaptive-neighborhood particle swarm with original invocation authority.
use crate::builtins::math::optim::common::{self, ScalarCooperation};
use crate::{BuiltinResult, RuntimeError};
use runmat_macros::runtime_builtin;
mod creation;
mod dynamics;
mod evaluation;
mod fitness;
pub(crate) mod helpers;
mod hybrid;
mod inputs;
mod iteration;
mod limits;
mod observation;
mod options;
mod population;
mod publication;
mod random;
mod state;
#[cfg(test)]
mod tests;
const NAME: &str = "particleswarm";
const POLICY: common::population::Policy = common::population::Policy {
    identity: IDENTITY,
    error: &runmat_builtins::PARTICLESWARM_ERROR_INVALID_ARGUMENT,
};
const IDENTITY: runmat_builtins::BuiltinCatalogIdentity =
    runmat_builtins::BuiltinCatalogIdentity { name: NAME };
fn invalid(message: impl std::fmt::Display) -> RuntimeError {
    common::swarm_options::invalid(
        common::swarm_options::Owner::Solver,
        format!("particleswarm: {message}"),
    )
}
#[runtime_builtin(
    name = "particleswarm",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::particleswarm"
)]
async fn registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    execute(runtime.as_ref(), args).await.map_err(|mut error| {
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })
}
async fn execute(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    crate::context::host_allocation::require_unbounded_workspace(runtime,NAME,
        "finite host-memory execution is unsupported; population, callback, random and result workspaces have no complete finite resource bound")?;
    crate::output_count::validate_bounded_outputs_in(
        &runmat_builtins::PARTICLESWARM_DESCRIPTOR,
        &runmat_builtins::PARTICLESWARM_ERROR_INVALID_ARGUMENT,
        NAME,
        runtime,
    )?;
    let _original = args.fork();
    let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, runtime);
    cooperation.completed(0).await?;
    let mut problem = inputs::parse(args, &mut cooperation).await?;
    let options = options::Options::parse(
        std::mem::replace(&mut problem.options, inputs::empty()),
        problem.variables,
        &mut cooperation,
    )
    .await?;
    let outcome = iteration::minimize(problem, options, &mut cooperation).await?;
    publication::finalize(outcome, &cooperation)
}
