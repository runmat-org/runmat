//! Internal goal-attainment integration into the constrained numerical owner.
//! This never calls the public fmincon builtin or manufactures a user callback.
pub(super) mod merit;
pub(super) mod observation;
mod parse;
mod publication;
#[cfg(test)]
mod tests;
use super::{evaluation, numeric, outcome, sqp};
use crate::builtins::math::optim::common::{
    multiobjective::{inputs, Kind},
    ScalarCooperation,
};
use crate::BuiltinResult;

pub(in crate::builtins::math::optim) struct Definition {
    pub kind: Kind,
    pub descriptor: &'static runmat_builtins::BuiltinDescriptor,
    pub error: &'static runmat_builtins::BuiltinErrorDescriptor,
    pub numeric: &'static runmat_builtins::BuiltinExtensionDescriptor,
    pub single: &'static runmat_builtins::BuiltinExtensionDescriptor,
    pub resident: &'static runmat_builtins::BuiltinExtensionDescriptor,
    pub integer_option: &'static runmat_builtins::BuiltinExtensionDescriptor,
    pub sparse: &'static runmat_builtins::BuiltinExtensionDescriptor,
}
pub(in crate::builtins::math::optim) async fn execute(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
    definition: Definition,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let kind = definition.kind;
    crate::context::host_allocation::require_unbounded_workspace(runtime, kind.name(),
        "finite host-memory execution is unsupported; vector callback, Lagrangian and sparse factorization workspace have no complete finite bound")?;
    crate::output_count::validate_bounded_outputs_in(
        definition.descriptor,
        definition.error,
        kind.name(),
        runtime,
    )?;
    let _original = args.fork();
    let owner = runmat_builtins::BuiltinCatalogIdentity { name: kind.name() };
    let mut cooperation = ScalarCooperation::for_builtin(owner, runtime);
    cooperation.completed(0).await?;
    let policy = numeric::Policy {
        owner,
        invalid: Some(definition.error),
        numeric: Some(definition.numeric),
        single: Some(definition.single),
        resident: Some(definition.resident),
        sparse: Some(definition.sparse),
    };
    let parsed = parse::prepare(
        kind,
        inputs::parse(kind, args)?,
        &policy,
        definition.integer_option,
        &mut cooperation,
    )
    .await?;
    let variables = parsed.x.len() - 1;
    let limit = parsed
        .controls
        .point_limit
        .unwrap_or(variables.checked_mul(100).ok_or_else(|| {
            crate::builtins::math::optim::common::multiobjective::invalid(
                kind,
                "default function-evaluation extent overflow",
            )
        })?);
    parsed
        .controls
        .scaling
        .validate_variables(kind.name(), variables)?;
    let parameters = outcome::Parameters {
        max_iterations: parsed.controls.max_iterations,
        feasibility: parsed.controls.feasibility,
        optimality: parsed.controls.optimality,
        function: Some(parsed.controls.function),
        step: parsed.controls.step,
        differences: parsed.controls.differences,
        callback: parsed.controls.callback(),
        display: parsed.controls.display,
    };
    let mut evaluator = evaluation::Evaluator {
        internal: None,
        semi_infinite: None,
        scaling: parsed.controls.scaling.clone(),
        attainment: Some(evaluation::Attainment {
            binding: evaluation::Binding::Unbound {
                kind,
                specification: parsed.specification,
            },
            merit: parsed.controls.merit,
        }),
        cooperation: &mut cooperation,
        objective: evaluation::Objective::User(&parsed.objective),
        feasibility_limit: None,
        selected_constraints: None,
        constraints: parsed.constraints.as_ref(),
        policy,
        shape: parsed.shape.clone(),
        scalar: parsed.scalar,
        objective_gradient: parsed.controls.objective_gradient,
        constraint_gradient: parsed.controls.constraint_gradient,
        budget: evaluation::Budget::new(limit),
        constraint_extents: None,
    };
    let outcome = if parsed.inconsistent_bounds {
        outcome::Outcome {
            algorithm: runmat_builtins::OptimAlgorithm::Sqp,
            x: parsed.x,
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
        sqp::minimize(parsed.x, &parsed.geometry, &parameters, &mut evaluator).await?
    };
    let generated = match evaluator.attainment.as_ref().map(|value| &value.binding) {
        Some(evaluation::Binding::Bound(formulation)) => formulation.rows.len(),
        _ => 0,
    };
    publication::finalize(
        kind,
        outcome,
        &parsed.geometry,
        &parsed.shape,
        parsed.scalar,
        generated,
        parsed.inconsistent_bounds,
        evaluator.cooperation,
    )
    .await
}
