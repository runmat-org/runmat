//! Retire phase-one/working-set numerical state before terminal publication.
use super::*;
pub(in crate::builtins::math::optim::quadprog) fn solve<'a, V: compiled::Validation + 'a>(
    problem: Problem,
    products: compiled::Products,
    c: &'a mut ScalarCooperation,
    validation: &'a mut V,
) -> impl std::future::Future<Output = BuiltinResult<Outcome>> + 'a {
    crate::execution::after_result(prepare(problem, products, c, validation), terminal::publish)
}
async fn prepare<'a, V: compiled::Validation>(
    mut problem: Problem,
    products: compiled::Products,
    c: &'a mut ScalarCooperation,
    validation: &'a mut V,
) -> BuiltinResult<terminal::Prepared<'a, V>> {
    problem.controls.require_active_clock()?;
    seed::prepare(&mut problem, c).await?;
    let (mut model, mut controls) = Model::new(problem, c).await?;
    controls.resolve_limit(model.f.len(), model.b.len())?;
    let mut point = std::mem::take(&mut model.seed);
    let mut state = terminal::State::Stopped {
        point: Vec::new(),
        reason: Reason::IterationLimit,
    };
    let mut active = None;
    let curvature_failure = match validation.restricted_convexity(c).await? {
        compiled::RestrictedCurvature::Nonconvex => Some(Reason::Nonconvex),
        compiled::RestrictedCurvature::Supplied => {
            (!curvature::certify(&model, c).await?).then_some(Reason::Nonconvex)
        }
        compiled::RestrictedCurvature::Convex => {
            (!curvature::certify(&model, c).await?).then_some(Reason::Unverified)
        }
    };
    if let Some(reason) = curvature_failure {
        state = terminal::State::Failure(reason);
    } else if controls.time_reached() {
        state = terminal::State::Stopped {
            point: Vec::new(),
            reason: Reason::TimeLimit,
        };
    } else if controls.remaining() != 0 {
        if !working_set::rows::feasible(view(&model), &point, controls.feasibility, OWNER, c)
            .await?
        {
            match feasibility::solve(&model, &point, &mut controls, validation, c).await? {
                conic::solve::Candidate::Converged { x, .. } => point = x,
                conic::solve::Candidate::CertifiedInfeasible => {
                    state = terminal::State::Failure(Reason::Infeasible);
                }
                conic::solve::Candidate::Stopped { kind, x } => {
                    if let Some(x) = x {
                        point = x;
                    }
                    let reason = match kind {
                        conic::solve::StopKind::TimeLimit => Reason::TimeLimit,
                        conic::solve::StopKind::IterationLimit => Reason::IterationLimit,
                        conic::solve::StopKind::InsufficientProgress => Reason::Numerical,
                    };
                    return Ok(terminal::Prepared {
                        model,
                        controls,
                        products,
                        active,
                        state: terminal::State::Stopped { point, reason },
                        validation,
                        c,
                    });
                }
                _ => {
                    return Ok(terminal::Prepared {
                        model,
                        controls,
                        products,
                        active,
                        state: terminal::State::Stopped {
                            point,
                            reason: Reason::Numerical,
                        },
                        validation,
                        c,
                    });
                }
            }
        }
        if !matches!(state, terminal::State::Failure(_)) && controls.remaining() != 0 {
            if !working_set::rows::feasible(view(&model), &point, controls.feasibility, OWNER, c)
                .await?
            {
                return Ok(terminal::Prepared {
                    model,
                    controls,
                    products,
                    active,
                    state: terminal::State::Stopped {
                        point,
                        reason: Reason::Numerical,
                    },
                    validation,
                    c,
                });
            }
            let limits = working_set::Limits {
                feasibility: controls.feasibility,
                dual: 0.,
                pivots: controls.remaining() as usize,
            };
            let mut target = target::Target {
                model: &model,
                controls: &mut controls,
                validation: &mut *validation,
                checked: None,
                active: None,
            };
            use working_set::Target;
            let result = if let Some(reason) = target.advanced(&point, c).await? {
                working_set::Outcome::Stopped { point, reason }
            } else {
                working_set::minimize(&mut target, view(&model), point, limits, OWNER, c).await?
            };
            active = target.active.take();
            let checked = target.checked.take();
            drop(target);
            state = match result {
                working_set::Outcome::Optimal(step) => terminal::State::Optimal {
                    step,
                    checked: checked
                        .ok_or_else(|| invalid("active optimum lost checked measurements"))?,
                },
                working_set::Outcome::Unbounded { point, direction } => {
                    terminal::State::Ray { point, direction }
                }
                working_set::Outcome::Numerical { point } => terminal::State::Stopped {
                    point,
                    reason: Reason::Numerical,
                },
                working_set::Outcome::Stopped { point, reason } => terminal::State::Stopped {
                    point,
                    reason: match reason {
                        working_set::Stop::IterationLimit => Reason::IterationLimit,
                        working_set::Stop::TimeLimit => Reason::TimeLimit,
                        working_set::Stop::ObjectiveLimit => Reason::ObjectiveLimit,
                        working_set::Stop::StepStagnation => Reason::StepStagnation,
                        working_set::Stop::Unverified => Reason::Unverified,
                    },
                },
            };
            return Ok(terminal::Prepared {
                model,
                controls,
                products,
                active,
                state,
                validation,
                c,
            });
        }
    }
    if let terminal::State::Stopped { point: stopped, .. } = &mut state {
        *stopped = point;
    }
    Ok(terminal::Prepared {
        model,
        controls,
        products,
        active,
        state,
        validation,
        c,
    })
}
mod seed;
