//! One quadratic candidate/terminal lifecycle, never public optimality authority.
use super::super::ScalarCooperation;
use super::{backend, terminal, Model};
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult};
mod validation;
use clarabel::solver::{DefaultInfo, DefaultSettingsBuilder, DefaultSolver, SolverStatus};
pub(crate) use validation::{SuppliedModel, TerminalValidation};

pub(crate) trait IterationControl {
    fn accuracy(&self) -> f64;
    fn remaining(&self) -> u32;
    /// Selected-invocation semantic deadlines, never an ambient physical clock.
    fn progress_control(&mut self) -> clarabel::solver::ProgressControl {
        clarabel::solver::ProgressControl::Continue
    }
    fn record(
        &mut self,
        previous: &mut u32,
        info: &DefaultInfo<f64>,
        scale: f64,
        runtime: Option<&RuntimeContext>,
    ) -> BuiltinResult<()>;
}
pub(crate) enum StopKind {
    IterationLimit,
    TimeLimit,
    InsufficientProgress,
}
pub(crate) enum Difficulty {
    Numerical,
    UnverifiedInfeasibility,
    UnverifiedRay,
    UnverifiedFeasibility,
}
pub(crate) enum Candidate {
    Converged {
        x: Vec<f64>,
        z: Vec<f64>,
    },
    CertifiedInfeasible,
    /// Exact LP/QP terminal authority only. Consumers needing integral-domain
    /// authority retain and combine the actual rational evidence themselves.
    CertifiedUnbounded {
        direction: terminal::VerifiedRay,
        feasible: terminal::VerifiedPoint,
    },
    Stopped {
        kind: StopKind,
        x: Option<Vec<f64>>,
    },
    Difficulty(Difficulty),
}
struct Progress<'a, C> {
    control: &'a mut C,
    cooperation: &'a mut ScalarCooperation,
    previous: u32,
    work: u64,
    scale: f64,
}
async fn run<C: IterationControl>(
    model: &Model<'_>,
    feasibility: bool,
    control: &mut C,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<DefaultSolver<f64>> {
    cooperation.completed(0).await?;
    let accuracy = control.accuracy().min(1e-9);
    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .direct_solve_method("qdldl".into())
        .max_iter(control.remaining())
        .tol_gap_abs(accuracy)
        .tol_gap_rel(accuracy)
        .tol_feas(accuracy)
        .tol_infeas_abs(accuracy)
        .tol_infeas_rel(accuracy)
        .build()
        .map_err(|source| {
            build_runtime_error(format!("{}: invalid backend settings", model.identity.name))
                .with_builtin(model.identity.name)
                .with_source(source)
                .build()
        })?;
    let mut progress = Progress {
        control,
        cooperation,
        previous: 0,
        work: model.work_units().max(1),
        scale: if feasibility {
            1.0
        } else {
            model.objective_scale
        },
    };
    backend::run(
        model,
        feasibility,
        settings,
        &mut progress,
        async |progress: &mut Progress<'_, C>, info: &DefaultInfo<f64>| {
            progress.cooperation.completed(0).await?;
            progress.control.record(
                &mut progress.previous,
                info,
                progress.scale,
                progress.cooperation.runtime(),
            )?;
            progress.cooperation.completed(progress.work).await?;
            Ok(progress.control.progress_control())
        },
    )
    .await
}
pub(crate) async fn candidate<C: IterationControl>(
    model: &Model<'_>,
    seed: &[f64],
    control: &mut C,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Candidate> {
    candidate_validated(model, seed, control, cooperation, &mut SuppliedModel).await
}
pub(crate) async fn candidate_validated<C: IterationControl, V: TerminalValidation>(
    model: &Model<'_>,
    seed: &[f64],
    control: &mut C,
    cooperation: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<Candidate> {
    let mut solver = run(model, false, control, cooperation).await?;
    match solver.solution.status {
        SolverStatus::Solved | SolverStatus::AlmostSolved => {
            let mut z = std::mem::take(&mut solver.solution.z);
            for chunk in z.chunks_mut(1024) {
                for value in chunk.iter_mut() {
                    *value *= model.objective_scale;
                }
                cooperation.completed(chunk.len() as u64).await?;
            }
            Ok(Candidate::Converged {
                x: std::mem::take(&mut solver.solution.x),
                z,
            })
        }
        SolverStatus::PrimalInfeasible | SolverStatus::AlmostPrimalInfeasible => Ok(
            if validation::infeasible(model, &solver.solution.z, cooperation, validation).await? {
                Candidate::CertifiedInfeasible
            } else {
                Candidate::Difficulty(Difficulty::UnverifiedInfeasibility)
            },
        ),
        SolverStatus::DualInfeasible | SolverStatus::AlmostDualInfeasible => {
            let Some(direction) =
                validation::ray(model, &solver.solution.x, cooperation, validation).await?
            else {
                return Ok(Candidate::Difficulty(Difficulty::UnverifiedRay));
            };
            if let Some(feasible) =
                validation::feasible(model, seed, cooperation, validation).await?
            {
                return Ok(Candidate::CertifiedUnbounded {
                    direction,
                    feasible,
                });
            }
            // The original kernel is retired before any auxiliary allocation.
            drop(solver);
            if control.remaining() == 0 {
                return Ok(Candidate::Stopped {
                    kind: StopKind::IterationLimit,
                    x: None,
                });
            }
            let auxiliary = run(model, true, control, cooperation).await?;
            if let Some(feasible) =
                validation::feasible(model, &auxiliary.solution.x, cooperation, validation).await?
            {
                return Ok(Candidate::CertifiedUnbounded {
                    direction,
                    feasible,
                });
            }
            if matches!(
                auxiliary.solution.status,
                SolverStatus::PrimalInfeasible | SolverStatus::AlmostPrimalInfeasible
            ) && validation::infeasible(model, &auxiliary.solution.z, cooperation, validation)
                .await?
            {
                return Ok(Candidate::CertifiedInfeasible);
            }
            Ok(
                if matches!(
                    auxiliary.solution.status,
                    SolverStatus::MaxIterations
                        | SolverStatus::MaxTime
                        | SolverStatus::CallbackTerminated
                ) {
                    Candidate::Stopped {
                        kind: if auxiliary.solution.status == SolverStatus::MaxTime {
                            StopKind::TimeLimit
                        } else {
                            StopKind::IterationLimit
                        },
                        x: None,
                    }
                } else {
                    Candidate::Difficulty(Difficulty::UnverifiedFeasibility)
                },
            )
        }
        SolverStatus::MaxIterations | SolverStatus::MaxTime | SolverStatus::CallbackTerminated => {
            Ok(Candidate::Stopped {
                kind: if solver.solution.status == SolverStatus::MaxTime {
                    StopKind::TimeLimit
                } else {
                    StopKind::IterationLimit
                },
                x: Some(std::mem::take(&mut solver.solution.x)),
            })
        }
        SolverStatus::InsufficientProgress => Ok(Candidate::Stopped {
            kind: StopKind::InsufficientProgress,
            x: Some(std::mem::take(&mut solver.solution.x)),
        }),
        _ => Ok(Candidate::Difficulty(Difficulty::Numerical)),
    }
}
