//! Bounded solver telemetry; it is separate from physical acceptance evidence.
use crate::progress::{emit_phase, FeaProgressPhase, FeaProgressStatus};

pub(super) fn report(iteration: usize, relative_residual: f64, status: FeaProgressStatus) {
    emit_phase(
        "Conjugate gradient",
        FeaProgressPhase::Solve,
        status,
        format!("iteration={iteration} recursive_relative_residual={relative_residual:.8e}"),
        Some(iteration as u64),
        None,
    );
}
