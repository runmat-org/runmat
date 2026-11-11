//! Admitted Ritz-search controls shared by workload planning and execution.
use crate::contracts::FeaRunError;
use runmat_analysis_core::AlgebraicAccuracyMeasure;
use serde::{Deserialize, Serialize};

/// The modal qualification: every retained mode's `algebraic_measure` must lie at or below
/// `algebraic_tolerance`. Each mode publishes both its normwise backward error and its relative
/// residual; the measure selects which one gates.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalConvergencePolicy {
    algebraic_measure: AlgebraicAccuracyMeasure,
    algebraic_tolerance: f64,
    maximum_subspace_iterations: usize,
    /// Search ceiling for Ritz vectors, including boundary probes. A request admits the smaller
    /// of this ceiling and the subspace it works with (see `admitted_subspace_dimension`).
    /// Reaching the admitted bound never establishes a complete clustered eigenspace.
    maximum_subspace_dimension: usize,
}

impl ModalConvergencePolicy {
    /// The backward error at [`AlgebraicAccuracyMeasure::BACKWARD_ERROR_TOLERANCE`].
    pub const fn production() -> Self {
        Self {
            algebraic_measure: AlgebraicAccuracyMeasure::BackwardError,
            algebraic_tolerance: AlgebraicAccuracyMeasure::BACKWARD_ERROR_TOLERANCE,
            maximum_subspace_iterations: 256,
            maximum_subspace_dimension: 256,
        }
    }

    /// A preview qualification: the backward error at 1e-8.
    pub const fn exploratory() -> Self {
        Self {
            algebraic_measure: AlgebraicAccuracyMeasure::BackwardError,
            algebraic_tolerance: 1.0e-8,
            maximum_subspace_iterations: 64,
            maximum_subspace_dimension: 64,
        }
    }

    /// Gates the modes on `measure` at `tolerance`.
    pub const fn with_algebraic_accuracy(
        mut self,
        measure: AlgebraicAccuracyMeasure,
        tolerance: f64,
    ) -> Self {
        self.algebraic_measure = measure;
        self.algebraic_tolerance = tolerance;
        self
    }

    /// Keeps the measure and changes its tolerance.
    pub const fn with_algebraic_tolerance(mut self, tolerance: f64) -> Self {
        self.algebraic_tolerance = tolerance;
        self
    }

    pub const fn with_maximum_subspace_iterations(mut self, iterations: usize) -> Self {
        self.maximum_subspace_iterations = iterations;
        self
    }

    /// Sets the admitted search capacity without changing residual or cluster tolerances.
    pub const fn with_maximum_subspace_dimension(mut self, dimension: usize) -> Self {
        self.maximum_subspace_dimension = dimension;
        self
    }

    pub const fn maximum_subspace_dimension(self) -> usize {
        self.maximum_subspace_dimension
    }

    /// Shared planning/execution bound: the largest subspace the solver searches for this
    /// request, and the size the resource plan reserves.
    ///
    /// The working subspace is the requested modes, plus room for a cluster that shares the
    /// last requested eigenvalue (`MODAL_BOUNDARY_CLUSTER_ALLOWANCE`), plus the boundary probes
    /// that prove separation (`MODAL_BOUNDARY_PROBE_WIDTH`). It is capped by the search ceiling
    /// and the reduced dimension. A partial spectrum needs at least one additional vector to
    /// prove separation from the retained boundary.
    pub(crate) fn admitted_subspace_dimension(
        self,
        reduced_dimension: usize,
        requested_modes: usize,
    ) -> Result<usize, FeaRunError> {
        self.validate()?;
        let requested = requested_modes.min(reduced_dimension);
        let working = requested
            .saturating_add(crate::solve::modal::MODAL_BOUNDARY_CLUSTER_ALLOWANCE)
            .saturating_add(crate::solve::modal::MODAL_BOUNDARY_PROBE_WIDTH);
        let capacity = self
            .maximum_subspace_dimension
            .min(working)
            .min(reduced_dimension);
        if requested == 0
            || capacity < requested
            || (requested < reduced_dimension && capacity == requested)
        {
            return Err(FeaRunError::InvalidModel(
                "modal subspace capacity must cover requested modes and a truncation-boundary probe".into(),
            ));
        }
        Ok(capacity)
    }

    pub const fn algebraic_measure(self) -> AlgebraicAccuracyMeasure {
        self.algebraic_measure
    }

    pub const fn algebraic_tolerance(self) -> f64 {
        self.algebraic_tolerance
    }

    /// The figure of a mode this policy gates: its backward error or its relative residual.
    pub const fn gated_residual(self, relative_residual: f64, backward_error: f64) -> f64 {
        match self.algebraic_measure {
            AlgebraicAccuracyMeasure::BackwardError => backward_error,
            AlgebraicAccuracyMeasure::RelativeResidual => relative_residual,
        }
    }

    /// The same measure in the sparse eigensolvers' terms.
    pub(crate) const fn solver_measure(self) -> runmat_sparse_solvers::eigen::ResidualMeasure {
        match self.algebraic_measure {
            AlgebraicAccuracyMeasure::BackwardError => {
                runmat_sparse_solvers::eigen::ResidualMeasure::BackwardError
            }
            AlgebraicAccuracyMeasure::RelativeResidual => {
                runmat_sparse_solvers::eigen::ResidualMeasure::RelativeResidual
            }
        }
    }

    pub const fn maximum_subspace_iterations(self) -> usize {
        self.maximum_subspace_iterations
    }

    pub(crate) fn validate(self) -> Result<(), FeaRunError> {
        if !self.algebraic_tolerance.is_finite()
            || self.algebraic_tolerance <= 0.0
            || self.algebraic_tolerance >= 1.0
        {
            return Err(FeaRunError::InvalidModel(
                "modal algebraic tolerance must be finite and lie strictly between zero and one"
                    .into(),
            ));
        }
        if self.maximum_subspace_iterations == 0 || self.maximum_subspace_dimension == 0 {
            return Err(FeaRunError::InvalidModel(
                "modal convergence requires nonempty subspace capacity and iteration controls"
                    .into(),
            ));
        }
        Ok(())
    }
}

impl Default for ModalConvergencePolicy {
    fn default() -> Self {
        Self::production()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admitted_subspace_is_sized_by_the_request_not_the_search_ceiling() {
        use crate::solve::modal::{
            MODAL_BOUNDARY_CLUSTER_ALLOWANCE, MODAL_BOUNDARY_PROBE_WIDTH, MODAL_LANCZOS_BLOCK_SIZE,
        };
        let production = ModalConvergencePolicy::production();
        assert_eq!(
            MODAL_BOUNDARY_CLUSTER_ALLOWANCE,
            MODAL_LANCZOS_BLOCK_SIZE - 1
        );
        // The drone-arm modal study: 4 requested modes on 1,135,827 unconstrained DOFs.
        let working = 4 + MODAL_BOUNDARY_CLUSTER_ALLOWANCE + MODAL_BOUNDARY_PROBE_WIDTH;
        assert_eq!(working, 13);
        assert_eq!(
            production
                .admitted_subspace_dimension(1_135_827, 4)
                .unwrap(),
            working
        );
        // The search ceiling still caps a large request, and the reduced dimension a small model.
        assert_eq!(
            production
                .admitted_subspace_dimension(1_000_000, 250)
                .unwrap(),
            production.maximum_subspace_dimension()
        );
        assert_eq!(production.admitted_subspace_dimension(10, 4).unwrap(), 10);
    }

    #[test]
    fn modal_capacity_requires_a_real_boundary_probe_and_explicit_cold_controls() {
        let bounded = ModalConvergencePolicy::production().with_maximum_subspace_dimension(4);
        assert_eq!(bounded.admitted_subspace_dimension(1000, 2).unwrap(), 4);
        assert_eq!(bounded.admitted_subspace_dimension(3, 3).unwrap(), 3);
        assert!(bounded.admitted_subspace_dimension(1000, 4).is_err());
        assert!(bounded
            .with_maximum_subspace_dimension(0)
            .validate()
            .is_err());
        let bytes = serde_json::to_vec(&bounded).unwrap();
        assert_eq!(
            serde_json::from_slice::<ModalConvergencePolicy>(&bytes).unwrap(),
            bounded
        );
        let mut missing = serde_json::to_value(bounded).unwrap();
        missing
            .as_object_mut()
            .unwrap()
            .remove("maximum_subspace_dimension");
        assert!(serde_json::from_value::<ModalConvergencePolicy>(missing).is_err());
    }
}
