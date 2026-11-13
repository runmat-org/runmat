//! Domain outcomes from one shared cooperative sparse numerical substrate.
use super::*;
use clarabel::solver::{DefaultInfo, DefaultSettingsBuilder, DefaultSolver, SolverStatus};
use model::ConicModel;

struct Progress<'a> {
    cooperation: &'a mut ScalarCooperation,
    controls: &'a mut options::Controls,
    previous: u32,
    work: u64,
    scale: f64,
}
async fn run(
    model: &ConicModel,
    feasibility: bool,
    controls: &mut options::Controls,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<DefaultSolver<f64>> {
    cooperation.completed(0).await?;
    controls.begin_phase(feasibility, cooperation.runtime())?;
    let tol = controls.tol.min(1e-9);
    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .direct_solve_method("qdldl".into())
        .max_iter(controls.limit())
        .tol_gap_abs(tol)
        .tol_gap_rel(tol)
        .tol_feas(tol)
        .tol_infeas_abs(tol)
        .tol_infeas_rel(tol)
        .build()
        .map_err(|source| {
            build_runtime_error("coneprog: invalid backend settings")
                .with_builtin(CONEPROG)
                .with_source(source)
                .build()
        })?;
    let original = model.view();
    let mut progress = Progress {
        cooperation,
        controls,
        previous: 0,
        work: original.work_units().max(1),
        scale: if feasibility {
            1.0
        } else {
            model.objective_scale
        },
    };
    conic::backend::run(
        &original,
        feasibility,
        settings,
        &mut progress,
        async |progress: &mut Progress<'_>, info: &DefaultInfo<f64>| {
            progress.cooperation.completed(0).await?;
            progress.controls.record(
                &mut progress.previous,
                info,
                progress.scale,
                progress.cooperation.runtime(),
            )?;
            progress.cooperation.completed(progress.work).await
        },
    )
    .await
}
pub(super) async fn solve_coneprog(
    problem: ConeProgram,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ConeOutcome> {
    cooperation.completed(0).await?;
    for (&lower, &upper) in problem.lb.iter().zip(&problem.ub) {
        if lower > upper || lower == f64::INFINITY || upper == f64::NEG_INFINITY {
            let outcome = ConeOutcome::failure(
                -2,
                0,
                "No finite feasible point exists within the supplied bounds.",
            );
            problem.options.summary(&outcome, cooperation.runtime())?;
            return Ok(outcome);
        }
    }
    let (model, mut controls) = ConicModel::new(problem, cooperation).await?;
    let outcome = solve_model(&model, &mut controls, cooperation).await?;
    controls.summary(&outcome, cooperation.runtime())?;
    Ok(outcome)
}
async fn solve_model(
    model: &ConicModel,
    controls: &mut options::Controls,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ConeOutcome> {
    let mut solver = run(model, false, controls, cooperation).await?;
    match solver.solution.status {
        SolverStatus::Solved | SolverStatus::AlmostSolved => {
            let mut z = std::mem::take(&mut solver.solution.z);
            for chunk in z.chunks_mut(1024) {
                for value in chunk.iter_mut() {
                    *value *= model.objective_scale;
                }
                cooperation.completed(chunk.len() as u64).await?;
            }
            let checked =
                validation::optimum(model, &solver.solution.x, &z, controls.tol, cooperation)
                    .await?;
            if let Some(checked) = checked.filter(|value| value.valid) {
                let lambda = if crate::output_count::current_output_count()
                    .is_some_and(|count| count >= 5)
                {
                    Some(model.multipliers(&z, cooperation).await?)
                } else {
                    None
                };
                let x = std::mem::take(&mut solver.solution.x);
                drop(solver);
                Ok(ConeOutcome{x:Some(x),fval:Some(checked.point.objective),lambda,exitflag:1,iterations:controls.iterations(),
                    constrviolation:checked.point.violation,primalfeasibility:checked.point.normalized,dualfeasibility:checked.dual,dualitygap:checked.gap,
                    message:"Optimal solution found; original primal, dual and cone complementarity checks passed.".into()})
            } else {
                Ok(ConeOutcome::failure(
                    -10,
                    controls.iterations(),
                    "Numerical result did not satisfy original conic optimality checks.",
                ))
            }
        }
        SolverStatus::PrimalInfeasible | SolverStatus::AlmostPrimalInfeasible => {
            if conic::terminal::primal_infeasible(&model.view(), &solver.solution.z, cooperation)
                .await?
            {
                Ok(ConeOutcome::failure(
                    -2,
                    controls.iterations(),
                    "Problem is infeasible; original conic certificate verified.",
                ))
            } else {
                Ok(ConeOutcome::failure(
                    -10,
                    controls.iterations(),
                    "Infeasibility certificate could not be verified against the original cones.",
                ))
            }
        }
        SolverStatus::DualInfeasible | SolverStatus::AlmostDualInfeasible => {
            if !conic::terminal::descent_ray(&model.view(), &solver.solution.x, cooperation).await?
            {
                return Ok(ConeOutcome::failure(
                    -10,
                    controls.iterations(),
                    "Descent certificate could not be verified against the original cones.",
                ));
            }
            if conic::terminal::feasible_point(&model.view(), &model.seed, cooperation).await? {
                return Ok(ConeOutcome::failure(
                    -3,
                    controls.iterations(),
                    "Problem is feasible and unbounded below.",
                ));
            }
            // No overlapping kernels: the auxiliary problem proves genuine
            // feasibility, sharing the same total iteration/context budget.
            drop(solver);
            if controls.remaining() == 0 {
                return Ok(ConeOutcome::failure(
                    0,
                    controls.iterations(),
                    "Iteration limit reached before primal feasibility was established.",
                ));
            }
            let auxiliary = run(model, true, controls, cooperation).await?;
            if conic::terminal::feasible_point(&model.view(), &auxiliary.solution.x, cooperation)
                .await?
            {
                Ok(ConeOutcome::failure(
                    -3,
                    controls.iterations(),
                    "Problem is feasible and unbounded below.",
                ))
            } else if matches!(
                auxiliary.solution.status,
                SolverStatus::PrimalInfeasible | SolverStatus::AlmostPrimalInfeasible
            ) && conic::terminal::primal_infeasible(
                &model.view(),
                &auxiliary.solution.z,
                cooperation,
            )
            .await?
            {
                Ok(ConeOutcome::failure(
                    -2,
                    controls.iterations(),
                    "Problem is infeasible; auxiliary original-cone certificate verified.",
                ))
            } else {
                let stopped = matches!(
                    auxiliary.solution.status,
                    SolverStatus::MaxIterations
                        | SolverStatus::MaxTime
                        | SolverStatus::CallbackTerminated
                );
                Ok(ConeOutcome::failure(
                    if stopped { 0 } else { -10 },
                    controls.iterations(),
                    "Primal feasibility was not established; no unbounded conclusion is published.",
                ))
            }
        }
        SolverStatus::MaxIterations | SolverStatus::MaxTime | SolverStatus::CallbackTerminated => {
            incumbent(
                model,
                std::mem::take(&mut solver.solution.x),
                0,
                controls,
                "Iteration limit reached before conic convergence.",
                cooperation,
            )
            .await
        }
        SolverStatus::InsufficientProgress => {
            incumbent(
                model,
                std::mem::take(&mut solver.solution.x),
                -7,
                controls,
                "Search direction became too small for further progress.",
                cooperation,
            )
            .await
        }
        SolverStatus::Unsolved | SolverStatus::NumericalError => Ok(ConeOutcome::failure(
            -10,
            controls.iterations(),
            "Numerical instability prevented a verified conic result.",
        )),
    }
}
async fn incumbent(
    model: &ConicModel,
    x: Vec<f64>,
    flag: i32,
    controls: &options::Controls,
    message: &str,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ConeOutcome> {
    let mut outcome = ConeOutcome::failure(flag, controls.iterations(), message);
    if let Some(point) = validation::primal(model, &x, cooperation)
        .await?
        .filter(|point| point.normalized <= controls.tol)
    {
        outcome.x = Some(x);
        outcome.fval = Some(point.objective);
        outcome.constrviolation = point.violation;
        outcome.primalfeasibility = point.normalized;
        outcome.message.push_str(" Returned x is feasible to the requested tolerance; no optimality or multiplier claim is made.");
    } else {
        outcome
            .message
            .push_str(" No tolerance-feasible incumbent is available.");
    }
    Ok(outcome)
}
