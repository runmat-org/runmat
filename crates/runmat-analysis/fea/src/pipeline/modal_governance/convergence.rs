//! Admitted Ritz-search controls shared by workload planning and execution.
use crate::contracts::FeaRunError;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModalConvergencePolicy {
    relative_residual_tolerance: f64,
    maximum_subspace_iterations: usize,
    /// Search ceiling for Ritz vectors, including boundary probes. A request admits the smaller
    /// of this ceiling and the subspace it works with (see `admitted_subspace_dimension`).
    /// Reaching the admitted bound never establishes a complete clustered eigenspace.
    maximum_subspace_dimension: usize,
}

impl ModalConvergencePolicy {
    pub const fn production() -> Self {
        Self {
            relative_residual_tolerance: 1.0e-8,
            maximum_subspace_iterations: 256,
            maximum_subspace_dimension: 256,
        }
    }

    pub const fn exploratory() -> Self {
        Self {
            relative_residual_tolerance: 5.0e-4,
            maximum_subspace_iterations: 64,
            maximum_subspace_dimension: 64,
        }
    }

    pub const fn with_relative_residual_tolerance(mut self, tolerance: f64) -> Self {
        self.relative_residual_tolerance = tolerance;
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

    pub const fn relative_residual_tolerance(self) -> f64 {
        self.relative_residual_tolerance
    }

    pub const fn maximum_subspace_iterations(self) -> usize {
        self.maximum_subspace_iterations
    }

    pub(crate) fn validate(self) -> Result<(), FeaRunError> {
        if !self.relative_residual_tolerance.is_finite()
            || self.relative_residual_tolerance <= 0.0
            || self.relative_residual_tolerance >= 1.0
        {
            return Err(FeaRunError::InvalidModel(
                "modal relative residual tolerance must be finite and lie strictly between zero and one"
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
