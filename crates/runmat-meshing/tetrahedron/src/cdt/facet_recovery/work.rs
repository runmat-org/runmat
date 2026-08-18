use runmat_meshing_core::MeshingCancellationSignal;

use super::{
    error, resource, DelaunayFacetRecoveryError, DelaunayFacetRecoveryErrorKind,
    DelaunayFacetRecoveryOptions,
};

pub(super) struct FacetRecoveryWork<'a> {
    pub(super) options: DelaunayFacetRecoveryOptions,
    pub(super) cancellation: &'a dyn MeshingCancellationSignal,
    search_steps: u64,
    flip_attempts: u64,
}

impl<'a> FacetRecoveryWork<'a> {
    pub(super) fn new(
        options: DelaunayFacetRecoveryOptions,
        cancellation: &'a dyn MeshingCancellationSignal,
    ) -> Self {
        Self {
            options,
            cancellation,
            search_steps: 0,
            flip_attempts: 0,
        }
    }

    pub(super) fn search_step(
        &mut self,
        constraint_index: u32,
    ) -> Result<(), DelaunayFacetRecoveryError> {
        self.search_steps += 1;
        if self.search_steps > self.options.maximum_search_steps {
            return Err(resource(
                constraint_index,
                "facet topology-search limit exceeded",
            ));
        }
        self.check_cancelled(constraint_index, self.search_steps)
    }

    pub(super) fn flip_attempt(
        &mut self,
        constraint_index: u32,
    ) -> Result<(), DelaunayFacetRecoveryError> {
        self.flip_attempts += 1;
        if self.flip_attempts > self.options.maximum_flip_attempts {
            return Err(resource(
                constraint_index,
                "facet edge-flip attempt limit exceeded",
            ));
        }
        self.check_cancelled(constraint_index, self.flip_attempts)
    }

    fn check_cancelled(
        &self,
        constraint_index: u32,
        step: u64,
    ) -> Result<(), DelaunayFacetRecoveryError> {
        if step.is_multiple_of(
            self.options
                .segment_recovery
                .constraints
                .cancellation_check_interval,
        ) && self.cancellation.is_cancelled()
        {
            return Err(error(
                DelaunayFacetRecoveryErrorKind::Cancelled,
                Some(constraint_index),
                "cancelled",
            ));
        }
        Ok(())
    }
}
