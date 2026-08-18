use runmat_meshing_core::MeshingCancellationSignal;

use super::{
    error, resource, DelaunaySegmentRecoveryError, DelaunaySegmentRecoveryErrorKind,
    DelaunaySegmentRecoveryOptions,
};

pub(super) struct RecoveryWork<'a> {
    pub(super) options: DelaunaySegmentRecoveryOptions,
    pub(super) cancellation: &'a dyn MeshingCancellationSignal,
    recovery_steps: u64,
    search_steps: u64,
    inserted_nodes: u64,
}

impl<'a> RecoveryWork<'a> {
    pub(super) fn new(
        options: DelaunaySegmentRecoveryOptions,
        cancellation: &'a dyn MeshingCancellationSignal,
    ) -> Self {
        Self {
            options,
            cancellation,
            recovery_steps: 0,
            search_steps: 0,
            inserted_nodes: 0,
        }
    }

    pub(super) fn recovery_step(
        &mut self,
        constraint_index: u32,
    ) -> Result<(), DelaunaySegmentRecoveryError> {
        self.recovery_steps += 1;
        if self.recovery_steps > self.options.maximum_recovery_steps {
            return Err(resource(
                Some(constraint_index),
                "segment recovery-step limit exceeded",
            ));
        }
        if self
            .recovery_steps
            .is_multiple_of(self.options.constraints.cancellation_check_interval)
            && self.cancellation.is_cancelled()
        {
            return Err(error(
                DelaunaySegmentRecoveryErrorKind::Cancelled,
                Some(constraint_index),
                "cancelled",
            ));
        }
        Ok(())
    }

    pub(super) fn search_step(
        &mut self,
        constraint_index: u32,
    ) -> Result<(), DelaunaySegmentRecoveryError> {
        self.search_steps += 1;
        if self.search_steps > self.options.maximum_search_steps {
            return Err(resource(
                Some(constraint_index),
                "segment topology-search limit exceeded",
            ));
        }
        if self
            .search_steps
            .is_multiple_of(self.options.constraints.cancellation_check_interval)
            && self.cancellation.is_cancelled()
        {
            return Err(error(
                DelaunaySegmentRecoveryErrorKind::Cancelled,
                Some(constraint_index),
                "cancelled",
            ));
        }
        Ok(())
    }

    pub(super) fn inserted_node(
        &mut self,
        constraint_index: u32,
    ) -> Result<(), DelaunaySegmentRecoveryError> {
        self.inserted_nodes += 1;
        if self.inserted_nodes > self.options.maximum_steiner_nodes {
            return Err(resource(
                Some(constraint_index),
                "segment Steiner-node limit exceeded",
            ));
        }
        Ok(())
    }
}
