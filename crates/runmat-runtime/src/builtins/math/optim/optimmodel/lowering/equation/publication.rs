//! Equation-shaped residual output reads the retained original numerical point.
use super::*;
use runmat_builtins::OptimizationExitflag as Flag;
use runmat_value::{NumericStorage, Tensor};
pub(in crate::builtins::math::optim::optimmodel) async fn publish(
    solved: run::Solved,
    requested: Option<usize>,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let count = requested.unwrap_or(1);
    if count > 4 {
        return Err(policy.invalid("equation solve returns at most four outputs"));
    }
    match solved {
        run::Solved::Linear {
            program,
            mut outcome,
        } => {
            let info = if count > 3 {
                let info =
                    super::super::publication::information::publish(&mut outcome.solver, policy, c)
                        .await?;
                if outcome.nonroot {
                    let message = Value::string(crate::call::argument_text::copy_str("The least-squares solver stopped at a nonzero equation residual; no equation solution was found.", c.runtime())?);
                    let Value::Struct(fields) = info.value() else {
                        return Err(policy.invalid("equation solver information is not a record"));
                    };
                    Some(crate::execution::structure::construct_struct_from_fields_in_optional_context(c.runtime(),
                        fields.fields.iter().filter(|(name,_)| *name != "message").chain(std::iter::once(("message", &message))))?)
                } else {
                    Some(info)
                }
            } else {
                None
            };
            let objective::ModelObjective::Equations(evidence) = &program.objective else {
                unreachable!("equation source evidence")
            };
            finish(
                &program.layout,
                evidence,
                outcome
                    .point
                    .map(super::super::residual::Candidate::Evaluated),
                outcome.flag,
                info,
                requested,
                policy,
                c,
            )
            .await
        }
        run::Solved::Nonlinear {
            model,
            outcome,
            timing,
        } => {
            let flag = nonlinear_flag(outcome.result.exitflag, policy)?;
            let info =
                if count > 3 {
                    let (value, solver) = match &outcome.options {
                        nonlinear::Options::System(_) => (
                            crate::builtins::math::optim::fsolve::compiled::information(
                                &outcome.result,
                            ),
                            "fsolve",
                        ),
                        nonlinear::Options::LeastSquares(options) => (
                            crate::builtins::math::optim::lsqnonlin::compiled::information(
                                &outcome.result,
                                options,
                            ),
                            "lsqnonlin",
                        ),
                    };
                    Some(information(
                    value,
                    solver,
                    Some(if outcome.result.point_evidence
                        == crate::builtins::math::optim::least_squares::PointEvidence::Supplied {
                        "not-evaluated"
                    } else { "finite-differences" }),
                    None,
                    timing.as_ref(),
                    policy,
                    c,
                )?)
                } else {
                    None
                };
            finish(
                &model.layout,
                &model.evidence,
                Some(outcome.point),
                flag,
                info,
                requested,
                policy,
                c,
            )
            .await
        }
        run::Solved::Scalar {
            model,
            outcome,
            timing,
        } => {
            let flag = if outcome.result.converged {
                Flag::EquationSolved
            } else {
                Flag::SolverLimitExceeded
            };
            let info = if count > 3 {
                Some(information(
                    crate::builtins::math::optim::fzero::compiled::information(&outcome.result),
                    "fzero",
                    None,
                    outcome
                        .result
                        .converged
                        .then_some("numerical-bracket-or-zero"),
                    timing.as_ref(),
                    policy,
                    c,
                )?)
            } else {
                None
            };
            finish(
                &model.layout,
                &model.evidence,
                Some(super::super::residual::Candidate::Evaluated(outcome.point)),
                flag,
                info,
                requested,
                policy,
                c,
            )
            .await
        }
    }
}
fn nonlinear_flag(flag: i32, policy: Policy) -> BuiltinResult<Flag> {
    Ok(match flag {
        1 => Flag::EquationSolved,
        2 => Flag::StepSizeBelowTolerance,
        3 => Flag::FunctionChangeBelowTolerance,
        0 => Flag::SolverLimitExceeded,
        -1 => Flag::OutputFcnStop,
        -2 => Flag::NoFeasiblePointFound,
        _ => return Err(policy.invalid("unsupported equation residual termination")),
    })
}
fn information(
    value: Value,
    solver: &'static str,
    derivative: Option<&'static str>,
    criterion: Option<&'static str>,
    timing: Option<&crate::builtins::math::optim::common::solver_timing::SolverTiming>,
    policy: Policy,
    c: &ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let value = RootedValue::in_optional_context(value, c.runtime())?;
    let Value::Struct(fields) = value.value() else {
        return Err(policy.invalid("equation solver information is not a record"));
    };
    let solver = Value::string(crate::call::argument_text::copy_str(solver, c.runtime())?);
    let derivative = derivative.map(Value::from);
    let criterion = criterion
        .map(|text| {
            crate::call::argument_text::copy_str(text, c.runtime())
                .map(|text| Value::String(text.into()))
        })
        .transpose()?;
    let time = Value::Num(
        timing
            .ok_or_else(
                crate::builtins::math::optim::common::solver_timing::SolverTiming::unavailable,
            )?
            .elapsed()
            .as_secs_f64(),
    );
    crate::execution::structure::construct_struct_from_fields_in_optional_context(
        c.runtime(),
        fields
            .fields
            .iter()
            .chain([("Solver", &solver), ("wallclocktime", &time)])
            .chain(
                derivative
                    .as_ref()
                    .map(|value| ("equationderivative", value)),
            )
            .chain(criterion.as_ref().map(|value| ("rootcriterion", value))),
    )
}
async fn finish(
    layout: &layout::Layout,
    evidence: &Evidence,
    point: Option<super::super::residual::Candidate>,
    flag: Flag,
    information: Option<RootedValue>,
    requested: Option<usize>,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    let count = requested.unwrap_or(1);
    let mut output = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
    if count > 0 {
        output.push(match &point {
            Some(super::super::residual::Candidate::Evaluated(point)) => {
                RootedValue::in_optional_context(point.solution.value().clone(), c.runtime())?
            }
            Some(super::super::residual::Candidate::Supplied(point)) => {
                RootedValue::in_optional_context(point.solution().value().clone(), c.runtime())?
            }
            None => super::super::publication::empty_solution(layout, policy, c).await?,
        });
    }
    if count > 1 {
        output.push(match &point {
            Some(super::super::residual::Candidate::Evaluated(point)) => {
                residual(evidence, point, policy, c).await?
            }
            Some(super::super::residual::Candidate::Supplied(_)) => {
                super::super::publication::empty(policy, c)?
            }
            None => super::super::publication::empty(policy, c)?,
        });
    }
    if count > 2 {
        output.push(super::super::super::exitflag::publish(
            flag,
            policy,
            c.runtime(),
        )?);
    }
    if count > 3 {
        output.push(
            information
                .ok_or_else(|| policy.invalid("equation solver information was not prepared"))?,
        );
    }
    super::super::publication::finish(output, requested, policy, c).await
}
pub(super) async fn residual(
    evidence: &Evidence,
    point: &numerical::Point,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    let mut values = runmat_value::exact_capacity_vec(evidence.formulation.entries.len())
        .map_err(|e| policy.storage(e))?;
    for (entry, geometry) in evidence
        .formulation
        .entries
        .iter()
        .zip(&evidence.geometries)
    {
        let mut data =
            runmat_value::exact_capacity_vec(geometry.plan.len()).map_err(|e| policy.storage(e))?;
        for row in 0..geometry.plan.len() {
            let (left, right) = geometry
                .indices(row)
                .ok_or_else(|| policy.invalid("equation output geometry changed"))?;
            let value =
                point.at(entry.left, left, policy)? - point.at(entry.right, right, policy)?;
            if !value.is_finite() {
                return Err(policy.invalid("equation output must be finite and real"));
            }
            data.push(value);
            c.completed(1).await?;
        }
        let value = Tensor::from_numeric_storage(NumericStorage::F64(data), entry.shape.clone())
            .map_err(|e| policy.invalid(e))?;
        values.push(RootedValue::in_optional_context(
            Value::Tensor(value),
            c.runtime(),
        )?);
    }
    // One named equation array still returns an array, not a one-field record.
    if values.len() == 1 {
        return Ok(values.remove(0));
    }
    let mut fields =
        runmat_value::exact_capacity_vec(values.len()).map_err(|e| policy.storage(e))?;
    for (entry, value) in evidence.formulation.entries.iter().zip(&values) {
        let name = entry
            .name
            .as_ref()
            .ok_or_else(|| policy.invalid("multiple equation entries require original names"))?;
        fields.push((name.as_str(), value.value()));
        c.completed(1).await?;
    }
    crate::execution::structure::construct_struct_from_fields_in_optional_context(
        c.runtime(),
        fields.into_iter(),
    )
}
