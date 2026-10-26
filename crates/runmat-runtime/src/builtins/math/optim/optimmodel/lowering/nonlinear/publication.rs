//! Actual nonlinear candidate outputs, distinct from certified affine points.
use super::outcome::ScalarOutcome;
use super::*;
use futures::future::Either;
use runmat_value::optimization_model::OptimizationObjectiveSense as Sense;
pub(in crate::builtins::math::optim::optimmodel::lowering) async fn publish(
    solved: Solved,
    formulation: &formulation::Formulation,
    requested: Option<usize>,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let Solved {
        model,
        outcome,
        timing,
    } = solved;
    match outcome.into_parts() {
        Either::Left(outcome) => {
            scalar(
                &model,
                outcome,
                formulation,
                requested,
                timing.as_ref(),
                policy,
                c,
            )
            .await
        }
        Either::Right(outcome) => {
            residual::publish(
                &model,
                outcome,
                formulation,
                requested,
                timing.as_ref(),
                policy,
                c,
            )
            .await
        }
    }
}
async fn scalar(
    model: &Model,
    outcome: ScalarOutcome,
    formulation: &formulation::Formulation,
    requested: Option<usize>,
    timing: Option<&crate::builtins::math::optim::common::solver_timing::SolverTiming>,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let count = requested.unwrap_or(1);
    if count > 5 {
        return Err(policy.invalid("solve returns at most five outputs"));
    }
    let flag = outcome.flag(policy)?;
    let mut roots = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
    if count > 0 {
        // A stopped nonlinear iterate can be infeasible. It is never inserted
        // into the LP/LSQ verified-feasible Point cache.
        let candidate = outcome.candidate();
        let point = numerical::solution(&model.layout, candidate, policy, c)
            .await?
            .ok_or_else(|| policy.invalid("nonlinear solver returned an invalid candidate"))?;
        roots.push(point);
    }
    if count > 1 {
        roots.push(match (outcome.objective(), &formulation.objective) {
            (Some(value), Some(objective)) => RootedValue::in_optional_context(
                Value::Num(if objective.sense == Sense::Maximize {
                    -value
                } else {
                    value
                }),
                c.runtime(),
            )?,
            _ => super::super::publication::empty(policy, c)?,
        });
    }
    if count > 2 {
        roots.push(super::super::super::exitflag::publish(
            flag,
            policy,
            c.runtime(),
        )?);
    }
    if count > 3 {
        let Value::Struct(fields) = outcome.information(timing)? else {
            return Err(policy.invalid("nonlinear solver information lost its record"));
        };
        let solver = Value::from(outcome.solver().name());
        let derivative = Value::from("finite-differences");
        roots.push(
            crate::execution::structure::construct_struct_from_fields_in_optional_context(
                c.runtime(),
                fields
                    .fields
                    .iter()
                    .chain(std::iter::once(("Solver", &solver)))
                    .chain(std::iter::once(("objectivederivative", &derivative))),
            )?,
        );
    }
    if count > 4 && outcome.solver() == runmat_builtins::OptimSolver::Fminunc {
        roots.push(super::super::publication::empty(policy, c)?);
    } else if count > 4 {
        let groups = outcome.multipliers(c).await?;
        let projected = groups.map(|groups| report::Multipliers {
            lower: groups.lower,
            upper: groups.upper,
            eqlin: groups.eqnonlin,
            ineqlin: groups.ineqnonlin,
        });
        roots.push(
            super::super::publication::multipliers::project(
                &model.layout,
                formulation,
                &model.rows,
                projected,
                policy,
                c,
            )
            .await?,
        );
    }
    super::super::publication::finish(roots, requested, policy, c).await
}
