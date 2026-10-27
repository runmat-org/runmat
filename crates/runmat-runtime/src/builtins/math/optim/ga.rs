//! Genetic optimization with retained population and original invocation authority.
use crate::builtins::math::optim::common::{self, ScalarCooperation};
use crate::RuntimeError;
use runmat_macros::runtime_builtin;

mod merit;
mod operators;
mod numeric;
mod inputs;
pub(in crate::builtins::math::optim) mod evaluation;
mod population;
mod random;
pub(in crate::builtins::math::optim) mod linear;
pub(crate) mod creation;
mod stall;
mod hybrid;
mod observation;
mod outcome;
mod history;
mod reproduction;
mod assessment;
mod lineage;
mod controller;
pub(in crate::builtins::math::optim) mod auxiliary;
mod control;
mod initialization;
pub(in crate::builtins::math::optim) mod callbacks;
mod score_value;
pub(crate) mod helpers;
pub(super) mod options;

const NAME: &str = "ga";
const IDENTITY: runmat_builtins::BuiltinCatalogIdentity =
    runmat_builtins::BuiltinCatalogIdentity { name: NAME };
const POLICY: common::population::Policy = common::population::Policy {
    identity: IDENTITY,
    error: &runmat_builtins::GA_ERROR_INVALID_ARGUMENT,
};
fn invalid(message: impl std::fmt::Display) -> RuntimeError {
    POLICY.invalid(message)
}

#[runtime_builtin(
    name = "ga",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::ga"
)]
async fn registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValueSequence> {
    execute(runtime.as_ref(), args).await.map_err(|mut error| {
        // A callback, RNG turn or accepted population may already have run.
        // A provider failure can never replay the complete search.
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })
}
async fn execute(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValueSequence> {
    crate::context::host_allocation::require_unbounded_workspace(runtime, NAME,
        "finite host-memory execution is unsupported; population, nonlinear, exact-feasibility, callback and result workspaces have no complete finite bound")?;
    crate::output_count::validate_bounded_outputs_in(
        &runmat_builtins::GA_DESCRIPTOR,
        &runmat_builtins::GA_ERROR_INVALID_ARGUMENT,
        NAME,
        runtime,
    )?;
    let _original_arguments = args.fork();
    let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, runtime);
    cooperation.completed(0).await?;
    let timing = common::solver_timing::SolverTiming::start(runtime)?;
    let raw = inputs::parse(args, &cooperation)?;
    let population = options::Prepared::population(&raw.options)?;
    inputs::preflight(&raw, population, &cooperation)?;
    let runtime = cooperation.runtime().cloned().ok_or_else(|| {
        let mut error = crate::context::RuntimeCapabilityError::new(
            crate::context::RuntimeCapability::Builtin,
            "genetic optimization requires its selected execution context",
        )
        .into_runtime_error();
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })?;
    crate::execution::admit_executable_sequence_with(&runtime, move || {
        prepare_and_solve(raw, population, timing, cooperation)
    })?
    .await
}

async fn prepare_and_solve(
    raw: inputs::raw::Raw,
    population: runmat_builtins::OptimGeneticPopulation,
    timing: common::solver_timing::SolverTiming,
    mut cooperation: ScalarCooperation,
) -> crate::BuiltinResult<crate::execution::RootedValueSequence> {
    let runtime = controller::continuation::original(&cooperation)?;
    let original = &mut cooperation;
    let ready = controller::continuation::run(&runtime, move || {
        initialization::prepare(raw, population, original)
    })?
    .await?;
    controller::solve(ready, timing, &mut cooperation).await
}
