//! Original objective and named model outputs, not the solver's weighted norm.
use super::*;
use runmat_builtins::OptimizationExitflag as Flag;
pub(in crate::builtins::math::optim::optimmodel::lowering::nonlinear) async fn publish(
    model: &Model,
    outcome: Outcome,
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
    let Outcome {
        result,
        options,
        bounds,
        point,
    } = outcome;
    let flag = match result.exitflag {
        1 => Flag::OptimalSolution,
        2 => Flag::StepSizeBelowTolerance,
        3 => Flag::FunctionChangeBelowTolerance,
        0 => Flag::SolverLimitExceeded,
        -2 => Flag::NoFeasiblePointFound,
        _ => return Err(policy.invalid("unsupported nonlinear residual solver termination")),
    };
    let mut roots = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
    let (solution, fval) = match point {
        super::super::super::residual::Candidate::Evaluated(point) => {
            let fval = if count > 1 {
                let objective = formulation
                    .objective
                    .as_ref()
                    .ok_or_else(|| policy.invalid("residual result lost original objective"))?;
                let value = point.at(objective.root, 0, policy)?;
                if !value.is_finite() {
                    return Err(policy.invalid("accepted original objective is not finite"));
                }
                Some(value)
            } else {
                None
            };
            (point.solution, fval)
        }
        super::super::super::residual::Candidate::Supplied(point) => {
            (point.into_solution(), (count > 1).then_some(f64::NAN))
        }
    };
    if count > 0 {
        roots.push(solution);
    }
    if let Some(value) = fval {
        roots.push(RootedValue::in_optional_context(
            Value::Num(value),
            c.runtime(),
        )?);
    }
    if count > 2 {
        roots.push(super::super::super::super::exitflag::publish(
            flag,
            policy,
            c.runtime(),
        )?);
    }
    if count > 3 {
        let Value::Struct(fields) = residual_solver::information(&result, &options) else {
            return Err(policy.invalid("residual information lost its record"));
        };
        let solver = Value::from("lsqnonlin");
        let derivative = Value::from(
            if result.point_evidence
                == crate::builtins::math::optim::least_squares::PointEvidence::Supplied
            {
                "not-evaluated"
            } else {
                "finite-differences"
            },
        );
        let elapsed = Value::Num(
            timing
                .ok_or_else(
                    crate::builtins::math::optim::common::solver_timing::SolverTiming::unavailable,
                )?
                .elapsed()
                .as_secs_f64(),
        );
        roots.push(
            crate::execution::structure::construct_struct_from_fields_in_optional_context(
                c.runtime(),
                fields
                    .fields
                    .iter()
                    .chain(std::iter::once(("Solver", &solver)))
                    .chain(std::iter::once(("objectivederivative", &derivative)))
                    .chain(std::iter::once(("wallclocktime", &elapsed))),
            )?,
        );
    }
    if count > 4 {
        let residual_solver::BoundMultipliers {
            mut lower,
            mut upper,
            finite_gradient,
        } = residual_solver::bound_multipliers(&result, &bounds, c).await?;
        let mut complete = finite_gradient;
        let multiplier = {
            // Raw residual bound projection uses J'r. Model objectives are
            // the full squared sum, so their matching gradient is 2J'r.
            for (lower, upper) in lower.iter_mut().zip(&mut upper) {
                *lower *= 2.;
                *upper *= 2.;
                complete &= lower.is_finite() && upper.is_finite();
                c.completed(1).await?;
            }
            complete.then_some(report::Multipliers {
                lower,
                upper,
                eqlin: Vec::new(),
                ineqlin: Vec::new(),
            })
        };
        roots.push(
            super::super::super::publication::multipliers::project(
                &model.layout,
                formulation,
                &model.rows,
                multiplier,
                policy,
                c,
            )
            .await?,
        );
    }
    super::super::super::publication::finish(roots, requested, policy, c).await
}
