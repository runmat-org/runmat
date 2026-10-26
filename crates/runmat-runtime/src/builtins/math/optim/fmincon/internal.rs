//! Borrowed internal merit refinement through the existing SQP/KKT owner.
//! This path has no public builtin registration, callbacks or output projection.
pub(in crate::builtins::math::optim) use super::evaluation::{
    internal::Oracle, EvaluationStop, Probe, ProbeRole,
};
use super::{
    derivatives::Difference,
    evaluation::{Budget, Evaluator},
    model::{GeometryView, Role},
    numeric::{self, Policy},
    outcome::{Parameters, Termination},
    sqp,
};
use crate::builtins::math::optim::{
    common::{derivative_scaling::DerivativeScaling, ScalarCooperation},
    intlinprog::feasibility::Domain,
};
use crate::{execution::RootedValue, BuiltinResult};

#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim) struct Limits {
    pub iterations: usize,
    pub evaluations: usize,
    pub feasibility: f64,
    pub optimality: f64,
    pub step: f64,
}
pub(in crate::builtins::math::optim) struct Problem<'a> {
    pub domain: &'a Domain,
    pub lower: &'a [f64],
    pub upper: &'a [f64],
    pub source: &'a RootedValue,
}
pub(in crate::builtins::math::optim) enum Stop {
    Optimality,
    Step,
    Function,
    Limit,
    Time,
    NoFeasiblePoint,
    Numerical,
}
pub(in crate::builtins::math::optim) struct Refinement {
    pub point: Vec<f64>,
    /// This is the actual local model merit, NOT the expensive objective.
    pub merit: Option<f64>,
    pub violation: Option<f64>,
    pub constraints: Option<ConstraintSummary>,
    pub stop: Stop,
}
pub(in crate::builtins::math::optim) struct ConstraintSummary {
    pub linear_maximum: f64,
    pub nonlinear_violated: usize,
    pub nonlinear_maximum: Option<f64>,
}

pub(in crate::builtins::math::optim) async fn minimize(
    x: Vec<f64>,
    problem: Problem<'_>,
    oracle: &mut dyn Oracle,
    policy: Policy,
    limits: Limits,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Refinement> {
    let Problem {
        domain,
        lower,
        upper,
        source: _source,
    } = problem;
    if x.len() != lower.len()
        || !limits.feasibility.is_finite()
        || limits.feasibility <= 0.0
        || !limits.optimality.is_finite()
        || limits.optimality < 0.0
        || !limits.step.is_finite()
        || limits.step < 0.0
    {
        return Err(numeric::invalid(
            &policy,
            "internal SQP controls",
            "invalid local geometry or tolerances",
        ));
    }
    for value in &x {
        if !value.is_finite() {
            return Err(numeric::invalid(
                &policy,
                "internal SQP point",
                "nonfinite coordinate",
            ));
        }
        cooperation.completed(1).await?;
    }
    let geometry = GeometryView::original(domain, lower, upper, policy.owner, cooperation).await?;
    let parameters = Parameters {
        max_iterations: limits.iterations,
        feasibility: limits.feasibility,
        optimality: limits.optimality,
        step: limits.step,
        function: None,
        differences: Difference::Forward,
        callback: None,
        display: runmat_builtins::OptimDisplay::Off,
    };
    let mut evaluator = Evaluator {
        feasibility_limit: None,
        internal: Some(oracle),
        semi_infinite: None,
        attainment: None,
        scaling: DerivativeScaling::default(),
        cooperation,
        objective: super::evaluation::Objective::Internal,
        constraints: None,
        selected_constraints: None,
        policy,
        shape: vec![1, x.len()],
        scalar: false,
        objective_gradient: false,
        constraint_gradient: false,
        budget: Budget::new(limits.evaluations),
        constraint_extents: None,
    };
    let outcome = sqp::minimize_view(
        x,
        &geometry,
        &parameters,
        &mut evaluator,
        sqp::Strategy::Conic,
    )
    .await?;
    let stop = match outcome.termination {
        Termination::Optimality => Stop::Optimality,
        Termination::Step => Stop::Step,
        Termination::Function => Stop::Function,
        Termination::Limit => Stop::Limit,
        Termination::Time => Stop::Time,
        Termination::NoFeasiblePoint => Stop::NoFeasiblePoint,
        Termination::Numerical => Stop::Numerical,
        Termination::Observer => {
            return Err(numeric::invalid(
                &policy,
                "internal SQP",
                "a callback-free local solve cannot publish an observer termination",
            ))
        }
    };
    let constraints = if let Some(linearization) = &outcome.constraints {
        if linearization.semi_infinite.is_some()
            || linearization.roles.len() != linearization.b.len()
        {
            return Err(numeric::invalid(
                &policy,
                "internal SQP result",
                "accepted constraint generation does not match its internal domain",
            ));
        }
        let mut summary = ConstraintSummary {
            linear_maximum: 0.0,
            nonlinear_violated: 0,
            nonlinear_maximum: None,
        };
        for (&role, &rhs) in linearization.roles.iter().zip(&linearization.b) {
            let residual = match role {
                Role::EqualityLinear(_) | Role::EqualityNonlinear(_) => rhs.abs(),
                _ => -rhs,
            };
            match role {
                Role::EqualityNonlinear(_) | Role::InequalityNonlinear(_) => {
                    summary.nonlinear_violated += usize::from(residual > 0.0);
                    summary.nonlinear_maximum = Some(
                        summary
                            .nonlinear_maximum
                            .map_or(residual, |old| old.max(residual)),
                    );
                }
                _ => summary.linear_maximum = summary.linear_maximum.max(residual),
            }
            cooperation.completed(1).await?;
        }
        Some(summary)
    } else {
        None
    };
    Ok(Refinement {
        point: outcome.x,
        merit: outcome.point.map(|point| point.f),
        violation: outcome.violation,
        constraints,
        stop,
    })
}
