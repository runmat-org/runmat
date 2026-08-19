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
    support_steps: u64,
    cavity_steps: u64,
    cavity_apex_attempts: u64,
    cavity_steiner_nodes: u64,
    cavity_steiner_insertion_attempts: u64,
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
            support_steps: 0,
            cavity_steps: 0,
            cavity_apex_attempts: 0,
            cavity_steiner_nodes: 0,
            cavity_steiner_insertion_attempts: 0,
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

    pub(super) fn support_step(
        &mut self,
        constraint_index: u32,
    ) -> Result<(), DelaunayFacetRecoveryError> {
        self.support_steps += 1;
        if self.support_steps > self.options.maximum_support_steps {
            return Err(resource(
                constraint_index,
                "facet support-construction limit exceeded",
            ));
        }
        self.check_cancelled(constraint_index, self.support_steps)
    }

    pub(super) fn cavity_step(
        &mut self,
        constraint_index: u32,
    ) -> Result<(), DelaunayFacetRecoveryError> {
        self.cavity_steps += 1;
        if self.cavity_steps > self.options.maximum_cavity_steps {
            return Err(resource(
                constraint_index,
                "facet edge-star cavity work limit exceeded",
            ));
        }
        self.check_cancelled(constraint_index, self.cavity_steps)
    }

    pub(super) fn cavity_apex_attempt(
        &mut self,
        constraint_index: u32,
    ) -> Result<(), DelaunayFacetRecoveryError> {
        self.cavity_apex_attempts += 1;
        if self.cavity_apex_attempts > self.options.maximum_cavity_apex_attempts {
            return Err(resource(
                constraint_index,
                "facet cavity apex-attempt limit exceeded",
            ));
        }
        self.check_cancelled(constraint_index, self.cavity_apex_attempts)
    }

    pub(super) fn cavity_steiner_node(
        &mut self,
        constraint_index: u32,
    ) -> Result<u64, DelaunayFacetRecoveryError> {
        self.cavity_steiner_nodes += 1;
        if self.cavity_steiner_nodes > self.options.maximum_cavity_steiner_nodes {
            return Err(resource(
                constraint_index,
                "facet cavity Steiner-node limit exceeded",
            ));
        }
        self.check_cancelled(constraint_index, self.cavity_steiner_nodes)?;
        Ok(self.cavity_steiner_nodes)
    }

    pub(super) fn cavity_steiner_insertion_attempt(
        &mut self,
        constraint_index: u32,
    ) -> Result<(), DelaunayFacetRecoveryError> {
        self.cavity_steiner_insertion_attempts += 1;
        if self.cavity_steiner_insertion_attempts > self.options.maximum_cavity_steiner_candidates {
            return Err(resource(
                constraint_index,
                "facet cavity Steiner insertion-attempt limit exceeded",
            ));
        }
        self.check_cancelled(constraint_index, self.cavity_steiner_insertion_attempts)
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
