//! Solve-phase progress from the shared sparse solvers.
//!
//! [`report_solver_progress`] is the `progress` hook of every FEA solve control. It runs on the
//! thread that drives the solve, which carries the FEA progress context. Iterative solves report
//! the iteration count and relative residual; direct solves report when numeric factorization
//! starts and finishes. Events leave at most every [`MINIMUM_EVENT_INTERVAL`], except that a
//! reported factorization start is always followed by its finish.
use std::cell::Cell;
use std::collections::BTreeMap;

use runmat_sparse_solvers::control::{Progress, ProgressPhase};
use runmat_time::Instant;

use super::rate::{MINIMUM_EVENT_INTERVAL, UNIT_METADATA};
use super::{emit_progress, FeaProgressEvent, FeaProgressPhase, FeaProgressStatus};

const OPERATION: &str = "fea.solve";

#[derive(Clone, Copy, Default)]
struct SolverProgressState {
    emitted_at: Option<Instant>,
    factorization_started: Option<Instant>,
}

thread_local! {
    static STATE: Cell<SolverProgressState> = const { Cell::new(SolverProgressState {
        emitted_at: None,
        factorization_started: None,
    }) };
}

/// Solver iterations between progress callbacks.
pub const SOLVER_PROGRESS_STRIDE: usize = 16;

pub fn report_solver_progress(progress: Progress) {
    let now = Instant::now();
    let mut state = STATE.with(Cell::get);
    let due = state
        .emitted_at
        .is_none_or(|at| now.saturating_duration_since(at) >= MINIMUM_EVENT_INTERVAL);
    let event = match progress.phase {
        ProgressPhase::NumericFactorization if due => {
            state.factorization_started = Some(now);
            Some(event(
                "factorization",
                "Factorizing the system matrix".into(),
                Some(0),
                Some(1),
                BTreeMap::new(),
            ))
        }
        ProgressPhase::FactorizationComplete => state.factorization_started.take().map(|at| {
            let seconds = now.saturating_duration_since(at).as_secs_f64();
            event(
                "factorization",
                format!("Factorization finished in {seconds:.1} s"),
                Some(1),
                Some(1),
                BTreeMap::from([("factorizationMs".into(), format!("{:.0}", seconds * 1e3))]),
            )
        }),
        ProgressPhase::Krylov if due => {
            let mut metadata =
                BTreeMap::from([("iteration".to_string(), progress.iteration.to_string())]);
            let message = if progress.residual.is_finite() {
                metadata.insert("residual".into(), format!("{:e}", progress.residual));
                format!(
                    "Iteration {}, relative residual {:.2e}",
                    progress.iteration, progress.residual
                )
            } else {
                format!("Iteration {}", progress.iteration)
            };
            Some(event(
                "iterations",
                message,
                Some(progress.iteration as u64),
                None,
                metadata,
            ))
        }
        ProgressPhase::Eigen if due => Some(event(
            "eigenpairs",
            format!(
                "{} eigenpairs converged after {} iterations",
                progress.converged, progress.iteration
            ),
            Some(progress.converged as u64),
            None,
            BTreeMap::from([("iteration".to_string(), progress.iteration.to_string())]),
        )),
        _ => None,
    };
    if let Some(event) = event {
        state.emitted_at = Some(now);
        STATE.with(|slot| slot.set(state));
        emit_progress(event);
    } else {
        STATE.with(|slot| slot.set(state));
    }
}

fn event(
    unit: &str,
    message: String,
    current: Option<u64>,
    total: Option<u64>,
    mut metadata: BTreeMap<String, String>,
) -> FeaProgressEvent {
    metadata.insert(UNIT_METADATA.into(), unit.into());
    let fraction = match (current, total) {
        (Some(current), Some(total)) if total > 0 => Some((current as f64 / total as f64).min(1.0)),
        _ => None,
    };
    FeaProgressEvent {
        operation: OPERATION.into(),
        phase: FeaProgressPhase::Solve,
        status: FeaProgressStatus::Advanced,
        message,
        current,
        total,
        fraction,
        metadata,
    }
}

#[cfg(test)]
#[path = "solver_tests.rs"]
mod tests;
