//! Shared constrained numerical ownership and original-context callbacks.
//! Public scalar and typed attainment models retain distinct formulations.
use crate::builtins::math::optim::common::{solver_timing::SolverTiming, ScalarCooperation};
use crate::{BuiltinResult, RuntimeError};
pub use runmat_builtins::FMINCON_DESCRIPTOR;
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};
mod active_set;
mod bound_geometry;
pub(in crate::builtins::math::optim) mod compiled;
mod derivatives;
mod evaluation;
pub(super) mod feasibility;
mod hessian;
mod interior_point;
pub(in crate::builtins::math::optim) mod internal;
mod model;
pub(super) mod multiobjective;
mod reflective;
use super::common::constrained_numeric as numeric;
mod observation;
mod options;
mod outcome;
mod parse;
mod publication;
mod quadratic;
pub(super) mod semi_infinite;
mod sqp;
#[cfg(test)]
mod tests;
mod validation;
mod workspace;
const NAME: &str = "fmincon";
const IDENTITY: runmat_builtins::BuiltinCatalogIdentity =
    runmat_builtins::BuiltinCatalogIdentity { name: NAME };

#[runtime_builtin(
    name = "fmincon",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::fmincon"
)]
async fn registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    execute(runtime.as_ref(), args).await
}
async fn execute(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    crate::context::host_allocation::require_unbounded_workspace(runtime,NAME,
        "finite host-memory execution is unsupported; callback, Lagrangian model and sparse factorization workspace have no complete finite bound")?;
    crate::output_count::validate_bounded_outputs_in(
        &FMINCON_DESCRIPTOR,
        &runmat_builtins::FMINCON_ERROR_INVALID_ARGUMENT,
        NAME,
        runtime,
    )?;
    let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, runtime);
    cooperation.completed(0).await?;
    let timing =
        if crate::output_count::current_output_count_in(runtime).is_some_and(|count| count >= 4) {
            Some(SolverTiming::start(runtime)?)
        } else {
            None
        };
    // The canonical invocation retains all original callback/options roots
    // through parsing, asynchronous evaluation and successor publication.
    let _original = args.fork();
    let policy = numeric::Policy {
        owner: IDENTITY,
        invalid: None,
        numeric: Some(&runmat_builtins::FMINCON_NUMERIC_EXTENSION),
        single: Some(&runmat_builtins::FMINCON_SINGLE_EXTENSION),
        resident: Some(&runmat_builtins::FMINCON_RESIDENT_EXTENSION),
        sparse: None,
    };
    let problem = parse::problem(args, policy, &mut cooperation).await?;
    let parse::Problem {
        policy,
        objective,
        constraints,
        x,
        shape,
        scalar,
        geometry,
        controls,
    } = problem;
    controls.scaling.validate_variables(NAME, x.len())?;
    let limit = match controls.point_limit {
        Some(limit) => limit,
        None => x
            .len()
            .checked_mul(100)
            .ok_or_else(|| invalid("default function-evaluation limit overflow"))?,
    };
    let parameters = outcome::Parameters {
        function: None,
        max_iterations: controls.max_iterations,
        feasibility: controls.feasibility,
        optimality: controls.optimality,
        step: controls.step,
        differences: controls.differences,
        callback: controls.callback(),
        display: controls.display,
    };
    let reflective_options =
        if controls.algorithm == runmat_builtins::OptimAlgorithm::TrustRegionReflective {
            let mode = reflective::select(
                &geometry,
                constraints.is_some(),
                controls.objective_gradient,
            )?;
            Some((mode, controls.reflective(x.len(), mode)?))
        } else {
            None
        };
    let mut evaluator = evaluation::Evaluator {
        internal: None,
        attainment: None,
        semi_infinite: None,
        scaling: controls.scaling.clone(),
        cooperation: &mut cooperation,
        objective: evaluation::Objective::User(&objective),
        feasibility_limit: None,
        selected_constraints: None,
        constraints: constraints.as_ref(),
        policy,
        shape: shape.clone(),
        scalar,
        objective_gradient: controls.objective_gradient,
        constraint_gradient: controls.constraint_gradient,
        budget: evaluation::Budget::new(limit),
        constraint_extents: None,
    };
    let outcome = match controls.algorithm {
        runmat_builtins::OptimAlgorithm::Sqp => {
            sqp::minimize(x, &geometry, &parameters, &mut evaluator).await?
        }
        runmat_builtins::OptimAlgorithm::InteriorPoint => {
            interior_point::minimize(x, &geometry, &parameters, &mut evaluator).await?
        }
        runmat_builtins::OptimAlgorithm::ActiveSet => {
            sqp::minimize_with(
                x,
                &geometry,
                &parameters,
                &mut evaluator,
                sqp::Strategy::WorkingSet,
            )
            .await?
        }
        runmat_builtins::OptimAlgorithm::TrustRegionReflective => {
            let (mode, controls) = reflective_options
                .as_ref()
                .expect("selected reflective controls");
            let solved =
                reflective::minimize(x, &geometry, &parameters, controls, *mode, &mut evaluator)
                    .await?;
            return reflective::finalize(
                solved,
                &geometry,
                &shape,
                scalar,
                timing,
                &mut cooperation,
            )
            .await;
        }
        _ => unreachable!("selected implemented algorithm"),
    };
    publication::finalize(outcome, &geometry, &shape, scalar, timing, &mut cooperation).await
}
pub(super) fn invalid(detail: impl AsRef<str>) -> RuntimeError {
    crate::build_runtime_error(format!("fmincon: {}", detail.as_ref()))
        .with_builtin(NAME)
        .with_identifier("RunMat:fmincon:InvalidArgument")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn empty() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty shape"))
}
