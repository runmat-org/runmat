//! Original interior-point body; controls, units and terminal checks are conserved.
use super::*;
pub(super) async fn solve<V: compiled::Validation>(
    problem: Problem,
    products: compiled::Products,
    units: objective::Units,
    cooperation: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<Outcome> {
    let (model, mut controls) = Model::new(problem, cooperation).await?;
    let out = if controls.remaining() == 0 {
        // A seed is only a stopped iterate, never an optimality assertion.
        incumbent(
            &model,
            model.seed[..model.c.n].to_vec(),
            0,
            "Iteration limit reached before solving.",
            &controls,
            cooperation,
            validation,
        )
        .await?
    } else {
        match conic::solve::candidate_validated(
            &model.view(),
            &model.seed,
            &mut controls,
            cooperation,
            &mut compiled::Terminal {
                model: &model,
                validation,
            },
        )
        .await?
        {
            Candidate::Converged { mut x, mut z } => {
                if x.len() != model.f.len() {
                    failure(&controls, "Backend returned invalid lifted geometry.")
                } else {
                    x.truncate(model.c.n);
                    match validation::optimum(&model, &x, &z, &controls, cooperation)
                        .await?
                        .filter(|v| v.valid)
                    {
                        Some(checked) => {
                            // Canonical KKT above uses the kernel's half-squared
                            // objective. Original model multipliers use its
                            // full squared objective, without changing C or P.
                            if matches!(units, objective::Units::OriginalSquared) {
                                for dual in z.iter_mut().skip(model.c.m) {
                                    *dual *= units.multiplier();
                                    cooperation.completed(1).await?;
                                }
                            }
                            if z.iter().any(|value| !value.is_finite())
                                || !validation
                                    .solution(
                                        &x,
                                        compiled::Rows::new(&model),
                                        &z,
                                        controls.feasibility,
                                        controls.optimality,
                                        cooperation,
                                    )
                                    .await?
                            {
                                failure(
                                    &controls,
                                    "Candidate failed original model objective/KKT validation.",
                                )
                            } else {
                                let lambda = if products == compiled::Products::WithMultipliers {
                                    Some(model.lambda(&z, cooperation).await?)
                                } else {
                                    None
                                };
                                Outcome{method:model::Method::InteriorPoint,x:Some(x),residual:Some(checked.residual),resnorm:Some(checked.resnorm),
                                lambda,flag:1,iterations:controls.iterations(),violation:checked.violation,
                                optimality:checked.optimality,gap:checked.gap,
                                message:"Optimal solution satisfies original residual, constraints and KKT checks."}
                            }
                        }
                        None => failure(
                            &controls,
                            "Numerical candidate failed original least-squares checks.",
                        ),
                    }
                }
            }
            Candidate::CertifiedInfeasible => Outcome::failure(
                -2,
                controls.iterations(),
                "No feasible solution; exact original constraint certificate verified.",
            ),
            Candidate::CertifiedUnbounded {
                direction,
                feasible,
            } => {
                drop((direction, feasible));
                failure(
                    &controls,
                    "An inconsistent recession claim cannot establish unbounded least squares.",
                )
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
                    Some(mut x) if x.len() == model.f.len() => {
                        x.truncate(model.c.n);
                        incumbent(&model, x, flag, message, &controls, cooperation, validation)
                            .await?
                    }
                    _ => Outcome::failure(flag, controls.iterations(), message),
                }
            }
            Candidate::Difficulty(_) => failure(
                &controls,
                "Unable to certify a reliable least-squares result.",
            ),
        }
    };
    cooperation.completed(0).await?;
    controls.summary(&out, cooperation.runtime())?;
    Ok(out)
}
async fn incumbent<V: compiled::Validation>(
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
        residual: Some(checked.residual),
        resnorm: Some(checked.resnorm),
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
