//! Original-context public invocation into the dynamic semi-infinite SQP owner.
use super::super::{evaluation, numeric, outcome, sqp};
use super::{bernstein::invalid, inputs, options::Options, publication, Domain};
use crate::builtins::math::optim::common::{solver_timing::SolverTiming, ScalarCooperation};
use crate::BuiltinResult;
const IDENTITY: runmat_builtins::BuiltinCatalogIdentity =
    runmat_builtins::BuiltinCatalogIdentity { name: "fseminf" };
pub(in crate::builtins::math::optim) async fn execute(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    crate::context::host_allocation::require_unbounded_workspace(runtime,"fseminf",
        "finite host-memory execution is unsupported; adaptive interpolation, dynamic constraint and factorization workspaces have no complete finite bound")?;
    crate::output_count::validate_bounded_outputs_in(
        &runmat_builtins::FSEMINF_DESCRIPTOR,
        &runmat_builtins::FSEMINF_ERROR_INVALID_ARGUMENT,
        "fseminf",
        runtime,
    )?;
    let _original = args.fork();
    let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, runtime);
    cooperation.completed(0).await?;
    let timing = if crate::output_count::current_output_count_in(runtime).is_some_and(|n| n >= 4) {
        Some(SolverTiming::start(runtime)?)
    } else {
        None
    };
    let policy = numeric::Policy {
        owner: IDENTITY,
        invalid: None,
        numeric: Some(&runmat_builtins::FSEMINF_NUMERIC_EXTENSION),
        single: Some(&runmat_builtins::FSEMINF_SINGLE_EXTENSION),
        resident: Some(&runmat_builtins::FSEMINF_RESIDENT_EXTENSION),
        sparse: None,
    };
    let problem = inputs::parse(args, &policy, &mut cooperation).await?;
    let options = Options::parse(problem.options, &mut cooperation).await?;
    options
        .scaling
        .validate_variables("fseminf", problem.x.len())?;
    let limit = options.point_limit.unwrap_or(
        problem
            .x
            .len()
            .checked_mul(100)
            .ok_or_else(|| invalid("default function-evaluation limit overflow"))?,
    );
    let mut domain = Domain::new(problem.families, options.controls)?;
    let plots = options
        .plots()
        .map(|v| crate::execution::RootedValue::in_optional_context(v.clone(), runtime))
        .transpose()?;
    domain.configure_plots(plots)?;
    let parameters = outcome::Parameters {
        max_iterations: options.max_iterations,
        feasibility: options.controls.mesh_tolerance,
        optimality: options.optimality,
        function: Some(options.function),
        step: options.step,
        differences: options.differences,
        callback: options.callback(),
        display: options.display,
    };
    let mut evaluator = evaluation::Evaluator {
        internal: None,
        semi_infinite: Some(domain),
        attainment: None,
        scaling: options.scaling.clone(),
        cooperation: &mut cooperation,
        objective: evaluation::Objective::User(&problem.objective),
        feasibility_limit: None,
        selected_constraints: None,
        constraints: Some(&problem.seminfcon),
        policy,
        shape: problem.shape.clone(),
        scalar: problem.scalar,
        objective_gradient: options.gradient,
        constraint_gradient: false,
        budget: evaluation::Budget::new(limit),
        constraint_extents: None,
    };
    let out = if problem.inconsistent_bounds {
        outcome::Outcome {
            algorithm: runmat_builtins::OptimAlgorithm::ActiveSet,
            x: problem.x,
            point: None,
            constraints: None,
            multipliers: Vec::new(),
            multipliers_available: false,
            hessian: None,
            iterations: 0,
            counts: evaluator.budget.counts(),
            termination: outcome::Termination::NoFeasiblePoint,
            violation: None,
            optimality: None,
            step: 0.0,
            alpha: None,
        }
    } else {
        sqp::minimize_with(
            problem.x,
            &problem.geometry,
            &parameters,
            &mut evaluator,
            sqp::Strategy::WorkingSet,
        )
        .await?
    };
    publication::finalize(
        out,
        &problem.geometry,
        &problem.shape,
        problem.scalar,
        timing,
        &mut cooperation,
    )
    .await
}
