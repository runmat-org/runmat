//! Trace spans of solver work, without a `step` field: they are the detail beneath a study's
//! Solve step in the Trace pane.
//!
//! - `solver.factorize`: one numeric factorization (`kind`, `dimension`).
//! - `solver.iterations`: one batch of [`ITERATIONS_PER_BATCH`] Krylov iterations (`first`,
//!   and `last` and `residual` when the batch ends). A solve opens one span per batch rather than
//!   one per iteration.
use tracing::field::Empty;
use tracing::Span;

use crate::direct::FactorKind;

/// Krylov iterations traced by one `solver.iterations` span.
pub const ITERATIONS_PER_BATCH: usize = 256;

/// Opens the span of one numeric factorization; it closes when the returned guard drops.
pub(crate) fn factorization_span(kind: FactorKind, dimension: usize) -> tracing::span::EnteredSpan {
    tracing::info_span!("solver.factorize", kind = ?kind, dimension).entered()
}

/// Batches of one Krylov solve's iterations. The open batch closes when the solve's
/// operations drop.
#[derive(Default)]
pub(crate) struct IterationBatches {
    open: Option<(Span, usize)>,
    last: usize,
    residual: f64,
}

impl IterationBatches {
    /// Records that the solve reached `iteration` with `residual` (`NaN` when unknown).
    pub(crate) fn observe(&mut self, iteration: usize, residual: f64) {
        if let Some((_, first)) = &self.open {
            if iteration >= *first && iteration - *first < ITERATIONS_PER_BATCH {
                self.last = iteration;
                self.residual = residual;
                return;
            }
            self.close();
        }
        self.open = Some((
            tracing::info_span!(
                "solver.iterations",
                first = iteration,
                last = Empty,
                residual = Empty
            ),
            iteration,
        ));
        self.last = iteration;
        self.residual = residual;
    }

    fn close(&mut self) {
        if let Some((span, _)) = self.open.take() {
            span.record("last", self.last);
            if self.residual.is_finite() {
                span.record("residual", self.residual);
            }
        }
    }
}

impl Drop for IterationBatches {
    fn drop(&mut self) {
        self.close();
    }
}
