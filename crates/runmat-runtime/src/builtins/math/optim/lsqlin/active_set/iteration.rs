//! Retire feasibility/working numerical phases before original-source publication checks.
use super::super::model::ActiveTermination as Reason;
use super::*;
pub(in crate::builtins::math::optim::lsqlin) fn solve<'a, V: compiled::Validation + 'a>(
    problem: Problem,
    products: compiled::Products,
    units: objective::Units,
    c: &'a mut ScalarCooperation,
    validation: &'a mut V,
) -> impl std::future::Future<Output = BuiltinResult<Outcome>> + 'a {
    crate::execution::after_result(
        run(problem, products, units, c, validation),
        terminal::publish,
    )
}
async fn run<'a, V: compiled::Validation + 'a>(
    problem: Problem,
    products: compiled::Products,
    units: objective::Units,
    c: &'a mut ScalarCooperation,
    validation: &'a mut V,
) -> BuiltinResult<terminal::Prepared<'a, V>> {
    let (model, mut controls, mut seed) = Model::new(problem, units, c).await?;
    let mut state = terminal::State::Stopped {
        point: Vec::new(),
        reason: Reason::IterationLimit,
    };
    if controls.remaining() != 0 {
        if !working_set::rows::feasible(
            model.view(),
            &seed,
            controls.feasibility,
            runmat_builtins::LSQLIN_CATALOG_ENTRY.identity,
            c,
        )
        .await?
        {
            match feasibility::solve(&model, &seed, &mut controls, validation, c).await? {
                conic::solve::Candidate::Converged { x, .. } => seed = x,
                conic::solve::Candidate::CertifiedInfeasible => state = terminal::State::Infeasible,
                conic::solve::Candidate::Stopped { kind, x } => {
                    if let Some(x) = x {
                        seed = x;
                    }
                    let reason = if matches!(kind, conic::solve::StopKind::IterationLimit) {
                        Reason::IterationLimit
                    } else {
                        Reason::NoDirection
                    };
                    return Ok(terminal::Prepared {
                        active: None,
                        model,
                        controls,
                        products,
                        state: terminal::State::Stopped {
                            point: seed,
                            reason,
                        },
                        validation,
                        c,
                    });
                }
                _ => {
                    return Ok(terminal::Prepared {
                        active: None,
                        model,
                        controls,
                        products,
                        state: terminal::State::Stopped {
                            point: seed,
                            reason: Reason::NoDirection,
                        },
                        validation,
                        c,
                    })
                }
            }
        }
        if !matches!(&state, terminal::State::Infeasible) && controls.remaining() != 0 {
            if !working_set::rows::feasible(
                model.view(),
                &seed,
                controls.feasibility,
                runmat_builtins::LSQLIN_CATALOG_ENTRY.identity,
                c,
            )
            .await?
            {
                return Ok(terminal::Prepared {
                    active: None,
                    model,
                    controls,
                    products,
                    state: terminal::State::Stopped {
                        point: seed,
                        reason: Reason::NoDirection,
                    },
                    validation,
                    c,
                });
            }
            let Some(scale) = optimality::Scale::new(&model, c).await? else {
                return Ok(terminal::Prepared {
                    active: None,
                    model,
                    controls,
                    products,
                    state: terminal::State::Stopped {
                        point: seed,
                        reason: Reason::IllConditioned,
                    },
                    validation,
                    c,
                });
            };
            let pivots = controls.remaining() as usize;
            let tolerance = controls.feasibility;
            let mut target = target::Target {
                model: &model,
                controls: &mut controls,
                validation: &mut *validation,
                scale,
                metrics: None,
                retained_active: None,
            };
            use working_set::Target;
            let result = if let Some(reason) = target.advanced(&seed, c).await? {
                working_set::Outcome::Stopped {
                    point: seed,
                    reason,
                }
            } else {
                working_set::minimize(
                    &mut target,
                    model.view(),
                    seed,
                    working_set::Limits {
                        feasibility: tolerance,
                        dual: 0.,
                        pivots,
                    },
                    runmat_builtins::LSQLIN_CATALOG_ENTRY.identity,
                    c,
                )
                .await?
            };
            let metrics = target.metrics.take();
            let active = target.retained_active.take();
            drop(target);
            state = match result {
                working_set::Outcome::Optimal(step) => terminal::State::Optimal {
                    step,
                    metrics: metrics.ok_or_else(|| {
                        invalid("verified active target lost its measured KKT evidence")
                    })?,
                },
                working_set::Outcome::Stopped { point, reason } => terminal::State::Stopped {
                    point,
                    reason: match reason {
                        working_set::Stop::IterationLimit | working_set::Stop::TimeLimit => {
                            Reason::IterationLimit
                        }
                        working_set::Stop::ObjectiveLimit => Reason::ObjectiveLimit,
                        working_set::Stop::StepStagnation => Reason::IllConditioned,
                        working_set::Stop::Unverified => Reason::Unverified,
                    },
                },
                working_set::Outcome::Numerical { point }
                | working_set::Outcome::Unbounded { point, .. } => terminal::State::Stopped {
                    point,
                    reason: Reason::NoDirection,
                },
            };
            return Ok(terminal::Prepared {
                active,
                model,
                controls,
                products,
                state,
                validation,
                c,
            });
        }
    }
    if !matches!(&state, terminal::State::Infeasible) {
        state = terminal::State::Stopped {
            point: seed,
            reason: Reason::IterationLimit,
        };
    }
    Ok(terminal::Prepared {
        active: None,
        model,
        controls,
        products,
        state,
        validation,
        c,
    })
}
