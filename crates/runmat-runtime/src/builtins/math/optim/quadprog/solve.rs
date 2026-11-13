//! Public quadratic verification and presentation stay in the QP domain.
use super::*;
use compiled::{Products, Validation};
use conic::solve::{Candidate, Difficulty, StopKind};
use model::Model;

pub(super) async fn solve(
    problem: Problem,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Outcome> {
    let products = if crate::output_count::current_output_count().is_some_and(|count| count >= 5) {
        Products::WithMultipliers
    } else {
        Products::Primal
    };
    solve_validated(problem, products, cooperation, &mut compiled::SuppliedModel).await
}
pub(super) async fn solve_validated<V: Validation>(
    problem: Problem,
    products: Products,
    cooperation: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<Outcome> {
    let n = problem.f.len();
    for (column, (&lower, &upper)) in problem.lb.iter().zip(&problem.ub).enumerate() {
        if lower > upper || lower == f64::INFINITY || upper == f64::NEG_INFINITY {
            if !validation
                .certificate(
                    compiled::CertificateCandidate::InconsistentBounds {
                        column,
                        lower,
                        upper,
                    },
                    cooperation,
                )
                .await?
            {
                let mut out = Outcome::failure(
                    -8,
                    0,
                    "Original model did not verify the assembled bound conflict.",
                );
                if problem.controls.active.is_some() {
                    out.method = model::Method::ActiveSet(model::ActiveTermination::Unverified);
                }
                return Ok(out);
            }
            let mut out = Outcome::failure(
                -2,
                0,
                "Inconsistent bounds; no finite feasible point exists.",
            );
            if problem.controls.active.is_some() {
                out.method = model::Method::ActiveSet(model::ActiveTermination::Infeasible);
            }
            if !problem.x0.is_empty() {
                out.x = Some(problem.x0);
            }
            problem.controls.summary(&out, cooperation.runtime())?;
            return Ok(out);
        }
        cooperation.completed(1).await?;
    }
    if problem.controls.active.is_some() {
        // One selected lifecycle owner, constructed only after original checks.
        // The default interior-point route gains no continuation allocation.
        let runtime = cooperation.runtime().cloned();
        return crate::execution::admit_borrowed_operation_with_contextual_error(
            runtime.as_ref(),
            move || active_set::solve(problem, products, cooperation, validation),
        )?
        .await;
    }
    if !validation.original_convexity() || !psd::certify(&problem.p, cooperation).await? {
        let out = Outcome::failure(
            -6,
            0,
            "Nonconvex quadratic matrix; interior-point-convex requires positive semidefinite H.",
        );
        problem.controls.summary(&out, cooperation.runtime())?;
        return Ok(out);
    }
    let (model, mut controls) = Model::new(problem, cooperation).await?;
    debug_assert_eq!(model.f.len(), n);
    let out = if controls.remaining() == 0 {
        incumbent(
            &model,
            model.seed.clone(),
            0,
            "Iteration limit reached before solving.",
            &controls,
            cooperation,
            validation,
        )
        .await?
    } else {
        solve_model(&model, &mut controls, products, cooperation, validation).await?
    };
    cooperation.completed(0).await?;
    controls.summary(&out, cooperation.runtime())?;
    Ok(out)
}
async fn solve_model<V: Validation>(
    model: &Model,
    controls: &mut options::Controls,
    products: Products,
    cooperation: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<Outcome> {
    let candidate = {
        let mut terminal = compiled::Terminal {
            roles: &model.roles,
            validation: &mut *validation,
        };
        conic::solve::candidate_validated(
            &model.view(),
            &model.seed,
            controls,
            cooperation,
            &mut terminal,
        )
        .await?
    };
    match candidate {
        Candidate::Converged { x, z } => {
            let checked = validation::optimum(model, &x, &z, controls, cooperation).await?;
            if let Some(checked) = checked.filter(|value| value.valid) {
                if !validation
                    .solution(
                        &x,
                        &model.roles,
                        &z,
                        controls.feasibility,
                        controls.optimality,
                        cooperation,
                    )
                    .await?
                {
                    return Ok(failure(
                        controls,
                        "Numerical result failed the original model optimality checks.",
                    ));
                }
                let lambda = if products == Products::WithMultipliers {
                    Some(model.lambda(&z, cooperation).await?)
                } else {
                    None
                };
                Ok(Outcome {
                    method: model::Method::InteriorPoint,
                    x: Some(x),
                    fval: Some(checked.objective),
                    lambda,
                    flag: 1,
                    iterations: controls.iterations(),
                    violation: checked.violation,
                    optimality: checked.optimality,
                    gap: checked.gap,
                    message:
                        "Optimal solution satisfies original quadratic KKT and constraint checks.",
                })
            } else {
                Ok(failure(
                    controls,
                    "Numerical result failed the original quadratic optimality checks.",
                ))
            }
        }
        Candidate::CertifiedInfeasible => Ok(Outcome::failure(
            -2,
            controls.iterations(),
            "No feasible solution; original exact infeasibility certificate verified.",
        )),
        Candidate::CertifiedUnbounded {
            direction,
            feasible,
        } => {
            drop((direction, feasible));
            Ok(unbounded(controls))
        }
        Candidate::Stopped { kind, x } => {
            let (flag, message) = match kind {
                StopKind::IterationLimit => (0, "Iteration limit reached."),
                StopKind::TimeLimit => (0, "Time limit reached."),
                StopKind::InsufficientProgress => {
                    (-8, "Solver made insufficient numerical progress.")
                }
            };
            match x {
                Some(x) => {
                    incumbent(model, x, flag, message, controls, cooperation, validation).await
                }
                None => Ok(Outcome::failure(
                    flag,
                    controls.iterations(),
                    "Iteration limit reached before feasibility was certified.",
                )),
            }
        }
        Candidate::Difficulty(reason) => Ok(failure(
            controls,
            match reason {
                Difficulty::Numerical => "Unable to compute a reliable quadratic step.",
                Difficulty::UnverifiedInfeasibility => {
                    "Infeasibility certificate could not be verified exactly."
                }
                Difficulty::UnverifiedRay => {
                    "Quadratic recession certificate could not be verified exactly."
                }
                Difficulty::UnverifiedFeasibility => {
                    "Feasibility was not certified; no unbounded conclusion is published."
                }
            },
        )),
    }
}
async fn incumbent<V: Validation>(
    model: &Model,
    x: Vec<f64>,
    flag: i32,
    message: &'static str,
    controls: &options::Controls,
    cooperation: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<Outcome> {
    let Some(checked) = validation::primal(model, &x, cooperation)
        .await?
        .filter(|v| v.violation <= controls.feasibility)
    else {
        return Ok(Outcome::failure(flag, controls.iterations(), message));
    };
    if !validation
        .incumbent(&x, controls.feasibility, cooperation)
        .await?
    {
        return Ok(Outcome::failure(flag, controls.iterations(), message));
    }
    Ok(Outcome {
        method: model::Method::InteriorPoint,
        x: Some(x),
        fval: Some(checked.objective),
        lambda: None,
        flag,
        iterations: controls.iterations(),
        violation: checked.violation,
        optimality: f64::NAN,
        gap: f64::NAN,
        message,
    })
}
fn failure(controls: &options::Controls, message: &'static str) -> Outcome {
    Outcome::failure(-8, controls.iterations(), message)
}
fn unbounded(controls: &options::Controls) -> Outcome {
    Outcome::failure(-3,controls.iterations(),"Feasible quadratic program is unbounded; exact recession and feasibility certificates verified.")
}
