//! Single-chain simulated annealing, not a local-solver alias.
use crate::builtins::math::optim::common::{self, ScalarCooperation};
use crate::{BuiltinResult, RuntimeError};
use runmat_macros::runtime_builtin;
mod evaluation;
pub(crate) mod helpers;
mod hybrid;
mod inputs;
mod iteration;
mod observation;
mod operators;
mod options;
mod publication;
mod random;
mod reanneal;
mod state;
#[cfg(test)]
mod tests;
const NAME: &str = "simulannealbnd";
const IDENTITY: runmat_builtins::BuiltinCatalogIdentity =
    runmat_builtins::BuiltinCatalogIdentity { name: NAME };
fn invalid(message: impl std::fmt::Display) -> RuntimeError {
    common::anneal_options::invalid(common::anneal_options::Owner::Solver, message.to_string())
}
#[runtime_builtin(
    name = "simulannealbnd",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::simulannealbnd"
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
        "finite host-memory execution is unsupported; annealing, custom callback, RNG and result workspaces have no complete finite bound")?;
    crate::output_count::validate_bounded_outputs_in(
        &runmat_builtins::SIMULANNEALBND_DESCRIPTOR,
        &runmat_builtins::SIMULANNEALBND_ERROR_INVALID_ARGUMENT,
        NAME,
        runtime,
    )?;
    let _original = args.fork();
    let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, runtime);
    cooperation.completed(0).await?;
    let problem = inputs::parse(args, &mut cooperation).await?;
    inputs::preflight(&problem, &mut cooperation).await?;
    // Timing is a required invocation service, even when its numeric limit is
    // Inf. Check availability before any option readback or warning effects;
    // the one SolverTiming owner starts later at numerical entry.
    let _ = runtime
        .ok_or_else(common::solver_timing::SolverTiming::unavailable)?
        .service_ports()
        .timing()?;
    let options = options::Options::parse(problem.options.clone(), &mut cooperation).await?;
    let outcome = iteration::minimize(problem, options, &mut cooperation).await?;
    publication::finalize(outcome, &cooperation)
}
