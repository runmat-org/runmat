//! Nonlinear models retain the original DAG and layout, not coefficient probes.
use super::*;
use crate::builtins::math::optim::fmincon::compiled;
use runmat_gc::GcHandle;
mod oracle;
mod invocation;
mod solver;
mod outcome;
mod constrained;
mod unconstrained;
mod residual;
use outcome::Outcome;
pub(super) use solver::select;
mod inputs;
mod publication;
pub(super) use publication::publish;

pub(in crate::builtins::math::optim::optimmodel) struct Model {
    pub(super) source: evaluation::Plan,
    pub(super) layout: layout::Layout,
    pub(super) squares: Option<objective::LeastSquares>,
    constraints: selection::Selection,
    geometries: Vec<constraint_geometry::Geometry>,
    pub(super) rows: Vec<assembly::Rows>,
    inequalities: usize,
    equalities: usize,
}
impl Model {
    pub(super) async fn prepare(
        source: evaluation::Plan,
        layout: layout::Layout,
        formulation: &formulation::Formulation,
        squares: Option<objective::LeastSquares>,
        policy: Policy,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        if layout.columns == 0 {
            return Err(
                policy.invalid("solve requires at least one original optimization variable")
            );
        }
        let mut roots = Vec::new();
        let mut geometries = runmat_value::exact_capacity_vec(formulation.constraints.len())
            .map_err(|e| policy.storage(e))?;
        let mut rows = runmat_value::exact_capacity_vec(formulation.constraints.len())
            .map_err(|e| policy.storage(e))?;
        let mut inequalities = 0usize;
        let mut equalities = 0usize;
        for constraint in &formulation.constraints {
            roots.try_reserve(2).map_err(|e| policy.reserve(e))?;
            roots.extend([constraint.left, constraint.right]);
            let geometry = constraint_geometry::Geometry::from_source(&source, constraint, policy)?;
            let count = geometry.plan.len();
            let equality = constraint.relation
                == runmat_value::optimization_model::OptimizationConstraintRelation::Equal;
            let start = if equality {
                &mut equalities
            } else {
                &mut inequalities
            };
            rows.push(assembly::Rows {
                equality,
                start: *start,
                count,
            });
            *start = start
                .checked_add(count)
                .ok_or_else(|| policy.invalid("nonlinear constraint row count overflow"))?;
            geometries.push(geometry);
            c.completed(1).await?;
        }
        let constraints = selection::Selection::prepare(&source, roots, policy, c).await?;
        Ok(Self {
            source,
            layout,
            squares,
            constraints,
            geometries,
            rows,
            inequalities,
            equalities,
        })
    }
}
pub(in crate::builtins::math::optim::optimmodel) struct Solved {
    pub(super) model: Model,
    outcome: Outcome,
    timing: Option<crate::builtins::math::optim::common::solver_timing::SolverTiming>,
}
pub(super) async fn solve(
    model: Model,
    formulation: &formulation::Formulation,
    solver: runmat_builtins::OptimSolver,
    starting: Option<&Value>,
    options: Option<&Value>,
    products: Products,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Solved> {
    let timing = if products == Products::Primal || solver == runmat_builtins::OptimSolver::Fminunc
    {
        None
    } else {
        Some(crate::builtins::math::optim::common::solver_timing::SolverTiming::start(c.runtime())?)
    };
    let outcome = invocation::Execution {
        model: &model,
        formulation,
        starting,
        options,
        products,
        policy,
        cooperation: c,
    }
    .select(solver)
    .await?;
    Ok(Solved {
        model,
        outcome,
        timing,
    })
}

#[cfg(test)]
pub(super) async fn check_probe_roles(
    model: &Model,
    formulation: &formulation::Formulation,
    policy: Policy,
    c: &mut ScalarCooperation,
) {
    use compiled::{EvaluationStop, Oracle, Probe, ProbeRole};
    let mut oracle = oracle::Original {
        model,
        formulation,
        policy,
    };
    assert!(
        matches!(oracle.evaluate(&[-1.], ProbeRole::ConstraintsOnly, c).await.unwrap(),
        Ok(Probe::ConstraintsOnly { inequality, equality }) if inequality == [0.75] && equality.is_empty())
    );
    assert!(matches!(
        oracle.evaluate(&[-1.], ProbeRole::Full, c).await.unwrap(),
        Err(EvaluationStop::Undefined)
    ));
}
