use runmat_meshing_core::MeshingCancellationSignal;

use super::{
    error, resource, DelaunaySegmentRecoveryError, DelaunaySegmentRecoveryErrorKind,
    DelaunaySegmentRecoveryOptions,
};

#[cfg(test)]
mod tests;

pub(super) struct RecoveryWork<'a> {
    pub(super) options: DelaunaySegmentRecoveryOptions,
    pub(super) cancellation: &'a dyn MeshingCancellationSignal,
    recovery_steps: u64,
    search_steps: u64,
    search_by_phase: [u64; 8],
    flip_attempts: u64,
    inserted_nodes: u64,
    pub(super) outcomes: RecoveryOutcomes,
}

/// Fixed-size cumulative observations; no retained topology or extra searches.
#[derive(Default)]
pub(super) struct RecoveryOutcomes {
    pub(super) present: u64,
    pub(super) local_reconnection: u64,
    pub(super) face_flip: u64,
    pub(super) interior_split: u64,
    pub(super) boundary_split: u64,
    pub(super) absent_witness_split: u64,
    pub(super) acute_split: u64,
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
            search_by_phase: [0; 8],
            flip_attempts: 0,
            inserted_nodes: 0,
            outcomes: RecoveryOutcomes::default(),
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
        self.search_in_phase(constraint_index, 0)
    }

    pub(super) fn search_location_step(
        &mut self,
        index: u32,
    ) -> Result<(), DelaunaySegmentRecoveryError> {
        self.search_in_phase(index, 1)
    }
    pub(super) fn search_encroachment_step(
        &mut self,
        index: u32,
    ) -> Result<(), DelaunaySegmentRecoveryError> {
        self.search_in_phase(index, 2)
    }
    pub(super) fn search_flip_step(
        &mut self,
        index: u32,
    ) -> Result<(), DelaunaySegmentRecoveryError> {
        self.search_in_phase(index, 3)
    }

    pub(super) fn search_point_index_step(
        &mut self,
        index: u32,
    ) -> Result<(), DelaunaySegmentRecoveryError> {
        self.search_in_phase(index, 4)
    }
    pub(super) fn search_point_handoff_step(
        &mut self,
        index: u32,
    ) -> Result<(), DelaunaySegmentRecoveryError> {
        self.search_in_phase(index, 5)
    }

    pub(super) fn search_cell_index_step(
        &mut self,
        index: u32,
    ) -> Result<(), DelaunaySegmentRecoveryError> {
        self.search_in_phase(index, 6)
    }
    pub(super) fn search_cell_query_step(
        &mut self,
        index: u32,
    ) -> Result<(), DelaunaySegmentRecoveryError> {
        self.search_in_phase(index, 7)
    }
    #[cfg(test)]
    pub(super) fn search_inventory(&self) -> [u64; 8] {
        self.search_by_phase
    }

    fn search_in_phase(
        &mut self,
        constraint_index: u32,
        phase: usize,
    ) -> Result<(), DelaunaySegmentRecoveryError> {
        self.search_steps = self.search_steps.checked_add(1).ok_or_else(|| {
            resource(
                Some(constraint_index),
                "segment search-work counter overflow",
            )
        })?;
        self.search_by_phase[phase] += 1;
        if self.search_steps > self.options.maximum_search_steps {
            let [topology, location, encroachment, flips, point_index, point_handoff, cell_index, cell_query] =
                self.search_by_phase;
            return Err(resource(
                Some(constraint_index),
                format!("segment topology-search limit exceeded: total={}, maximum={}, topology={topology}, location={location}, encroachment={encroachment}, flips={flips}, point_index={point_index}, point_handoff={point_handoff}, cell_index={cell_index}, cell_query={cell_query}", self.search_steps, self.options.maximum_search_steps),
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

    pub(super) fn flip_attempt(
        &mut self,
        constraint_index: u32,
    ) -> Result<(), DelaunaySegmentRecoveryError> {
        self.flip_attempts += 1;
        if self.flip_attempts > self.options.maximum_flip_attempts {
            return Err(resource(
                Some(constraint_index),
                "segment face-flip attempt limit exceeded",
            ));
        }
        if self
            .flip_attempts
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
}
