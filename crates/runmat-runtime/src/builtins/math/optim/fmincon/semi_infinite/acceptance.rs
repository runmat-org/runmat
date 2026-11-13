//! Refresh accepted semi-infinite rows before any nonlinear KKT publication.
//! The accepted objective and original x are retained; only a real requested
//! mesh callback is counted as an additional constraint-only evaluated point.
use super::super::{
    derivatives::{self, Completion},
    evaluation::Evaluator,
    model::{Geometry, Role},
    outcome::{Outcome, Parameters, Termination},
    workspace,
};
use super::{
    bernstein::invalid,
    generation::FiniteExtents,
    oracle::{self, Raw, Request},
    Domain, Payload,
};
use crate::BuiltinResult;
pub(in crate::builtins::math::optim::fmincon) enum Refresh {
    Unchanged,
    Rebuilt,
    Stopped(Termination),
}
fn recommended(
    domain: &Domain,
    next: &super::generation::Generation,
    allow_coarsen: bool,
) -> BuiltinResult<Vec<f64>> {
    let count = domain.state.family_count();
    let mut out = runmat_value::exact_capacity_vec(
        count
            .checked_mul(2)
            .ok_or_else(|| invalid("sampling interval extent overflow"))?,
    )
    .map_err(|source| {
        crate::build_runtime_error("fseminf: interval allocation failed")
            .with_source(source)
            .build()
    })?;
    for family in &next.families {
        let spacing =
            if family.recommendation == super::mesh::Recommendation::Coarsen && !allow_coarsen {
                family.samples.spacing
            } else {
                family.recommended
            };
        out.push(spacing.column);
    }
    for family in &next.families {
        let spacing =
            if family.recommendation == super::mesh::Recommendation::Coarsen && !allow_coarsen {
                family.samples.spacing
            } else {
                family.recommended
            };
        out.push(spacing.row.unwrap_or(0.0));
    }
    Ok(out)
}
fn same_mesh(
    a: &[std::sync::Arc<super::mesh::Samples>],
    b: &[std::sync::Arc<super::mesh::Samples>],
) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| {
            a.rows == b.rows
                && a.columns == b.columns
                && a.spacing.column == b.spacing.column
                && a.spacing.row == b.spacing.row
        })
}
pub(in crate::builtins::math::optim::fmincon) async fn refresh(
    out: &mut Outcome,
    geometry: &Geometry,
    parameters: &Parameters<'_>,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<Refresh> {
    if evaluator.semi_infinite.is_none() || out.point.is_none() {
        return Ok(Refresh::Unchanged);
    }
    // Invalidate the model/certificate before invoking any asynchronous
    // callback or numerical workspace owner. Errors cannot publish old KKT.
    let old_model = out.constraints.take();
    out.multipliers_available = false;
    out.optimality = None;
    out.violation = None;
    let mut domain = evaluator.semi_infinite.take().unwrap();
    let result =
        async {
            let point = out.point.as_mut().unwrap();
            let payload = point
                .semi_infinite
                .take()
                .ok_or_else(|| invalid("accepted point has no original sampling payload"))?;
            let old_rows = payload.rows.generation().rows.len();
            let mut old_duals = workspace::capacity(old_rows, evaluator.policy.owner)?;
            for _ in 0..old_rows {
                old_duals.push(0.0);
                evaluator.cooperation.completed(1).await?;
            }
            if let Some(model) = &old_model {
                if model.roles.len() != out.multipliers.len() {
                    return Err(invalid(
                        "accepted multipliers have a different original row layout",
                    ));
                }
                for (role, &dual) in model.roles.iter().zip(&out.multipliers) {
                    if let Role::InequalityNonlinear(row) = role {
                        let slot = old_duals.get_mut(*row).ok_or_else(|| {
                            invalid("semi-infinite source row is absent from its mesh generation")
                        })?;
                        if !dual.is_finite() {
                            return Err(invalid("source warm multiplier is not finite"));
                        }
                        *slot = dual.max(0.0);
                    }
                    evaluator.cooperation.completed(1).await?;
                }
            }
            let mut finite = payload.finite;
            let mut inequality = workspace::capacity(finite.inequality, evaluator.policy.owner)?;
            for &value in point.inequality.iter().take(finite.inequality) {
                inequality.push(value);
                evaluator.cooperation.completed(1).await?;
            }
            let mut equality = std::mem::take(&mut point.equality);
            let mut samples = payload.samples;
            let mut coarsening_attempted = false;
            let (proposal, stop) =
                loop {
                    let proposal = domain
                        .state
                        .propose(
                            samples.clone(),
                            finite,
                            &old_duals,
                            domain.controls.mesh_tolerance,
                            evaluator.cooperation,
                        )
                        .await?;
                    let refine =
                        proposal.next.families.iter().any(|family| {
                            family.recommendation == super::mesh::Recommendation::Refine
                        });
                    let coarsen = proposal.next.families.iter().any(|family| {
                        family.recommendation == super::mesh::Recommendation::Coarsen
                    }) && !coarsening_attempted;
                    if !refine && !coarsen {
                        break (proposal, None);
                    }
                    let intervals = recommended(&domain, &proposal.next, !coarsening_attempted)?;
                    coarsening_attempted |= coarsen;
                    let raw = oracle::call(Request {
                        x: &out.x,
                        shape: &evaluator.shape,
                        scalar: evaluator.scalar,
                        callback: evaluator
                            .constraints
                            .ok_or_else(|| invalid("missing original seminfcon callback"))?,
                        intervals,
                        families: domain.state.family_count(),
                        policy: &evaluator.policy,
                        cooperation: evaluator.cooperation,
                        budget: &mut evaluator.budget,
                        reserve_point: true,
                        check_values: domain.controls.check_values,
                    })
                    .await?;
                    let Raw {
                        inequality: next_inequality,
                        equality: next_equality,
                        samples: next_samples,
                    } = match raw {
                        Ok(raw) => raw,
                        Err(super::super::evaluation::EvaluationStop::Time) => {
                            break (proposal, Some(Termination::Time))
                        }
                        Err(super::super::evaluation::EvaluationStop::BudgetExhausted) => {
                            break (proposal, Some(Termination::Limit))
                        }
                        Err(super::super::evaluation::EvaluationStop::Undefined) => {
                            break (proposal, Some(Termination::Numerical))
                        }
                    };
                    let next_finite = FiniteExtents {
                        inequality: next_inequality.len(),
                        equality: next_equality.len(),
                    };
                    if next_finite != finite {
                        return Err(invalid(
                            "finite constraints changed extent during a same-x mesh refresh",
                        ));
                    }
                    let ignored = same_mesh(&samples, &next_samples);
                    inequality = next_inequality;
                    equality = next_equality;
                    samples = next_samples;
                    finite = next_finite;
                    if ignored {
                        // S is documented as a recommendation: callbacks may keep
                        // their own fixed sampling set. Still remaximize that actual
                        // returned interpolant; never replace its peaks by node max.
                        break (
                            domain
                                .state
                                .propose(
                                    samples.clone(),
                                    finite,
                                    &old_duals,
                                    domain.controls.mesh_tolerance,
                                    evaluator.cooperation,
                                )
                                .await?,
                            None,
                        );
                    }
                };
            let (rows, warm) = domain.state.commit(proposal)?;
            point.inequality = rows
                .evaluate(&samples, &inequality, evaluator.cooperation)
                .await?;
            point.equality = equality;
            point.inequality_jacobian = None;
            point.equality_jacobian = None;
            point.projected_constraints = false;
            point.semi_infinite = Some(Payload {
                rows,
                samples,
                finite,
            });
            Ok((warm, stop))
        }
        .await;
    // Restore the exact domain before derivative probes. No live context was
    // retained by Domain or sent through a transport artifact.
    evaluator.semi_infinite = Some(domain);
    let (warm, stop) = result?;
    if let Some(stop) = stop {
        return Ok(Refresh::Stopped(stop));
    }
    let point = out.point.as_mut().unwrap();
    match derivatives::complete(
        evaluator,
        &out.x,
        &geometry.lower,
        &geometry.upper,
        point,
        parameters.differences,
    )
    .await?
    {
        Completion::Time => return Ok(Refresh::Stopped(Termination::Time)),
        Completion::BudgetExhausted => return Ok(Refresh::Stopped(Termination::Limit)),
        Completion::Undefined => return Ok(Refresh::Stopped(Termination::Numerical)),
        Completion::Complete => {}
    }
    let next = geometry
        .linearize(&out.x, point, evaluator.policy.owner, evaluator.cooperation)
        .await?;
    let mut multipliers = workspace::capacity(next.roles.len(), evaluator.policy.owner)?;
    let mut old_cursor = 0usize;
    for role in &next.roles {
        let value = match role {
            Role::InequalityNonlinear(row) => *warm
                .values()
                .get(*row)
                .ok_or_else(|| invalid("warm nonlinear row is absent from the refreshed mesh"))?,
            _ => {
                if let Some(old) = &old_model {
                    while old
                        .roles
                        .get(old_cursor)
                        .is_some_and(|role| matches!(role, Role::InequalityNonlinear(_)))
                    {
                        old_cursor += 1;
                    }
                    if old.roles.get(old_cursor) != Some(role) {
                        return Err(invalid(
                            "original finite constraint identity changed during mesh refresh",
                        ));
                    }
                    let value = out.multipliers[old_cursor];
                    old_cursor += 1;
                    value
                } else {
                    0.0
                }
            }
        };
        multipliers.push(value);
        evaluator.cooperation.completed(1).await?;
    }
    out.constraints = Some(next);
    out.multipliers = multipliers;
    super::require_current(evaluator, point, out.constraints.as_ref().unwrap())?;
    out.violation = geometry
        .violation(&out.x, point, evaluator.policy.owner, evaluator.cooperation)
        .await?
        .map(|value| value.0);
    out.optimality = None;
    out.multipliers_available = false;
    Ok(Refresh::Rebuilt)
}
