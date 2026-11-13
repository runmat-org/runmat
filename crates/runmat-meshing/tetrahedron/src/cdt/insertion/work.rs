use runmat_meshing_core::MeshingCancellationSignal;

use super::{
    error, resource, DelaunayInsertionError, DelaunayInsertionErrorKind, DelaunayInsertionOptions,
};

pub(super) struct InsertionWork<'a> {
    pub(super) options: DelaunayInsertionOptions,
    cancellation: &'a dyn MeshingCancellationSignal,
    predicate_evaluations: u64,
    checkpoints: u64,
    pub(super) legalization_flips: u64,
}

impl<'a> InsertionWork<'a> {
    pub(super) fn new(
        options: DelaunayInsertionOptions,
        cancellation: &'a dyn MeshingCancellationSignal,
    ) -> Self {
        Self {
            options,
            cancellation,
            predicate_evaluations: 0,
            checkpoints: 0,
            legalization_flips: 0,
        }
    }

    pub(super) fn checkpoint(&mut self) -> Result<(), DelaunayInsertionError> {
        self.checkpoints += 1;
        if self
            .checkpoints
            .is_multiple_of(self.options.topology.cancellation_check_interval)
            && self.cancellation.is_cancelled()
        {
            return Err(error(DelaunayInsertionErrorKind::Cancelled, "cancelled"));
        }
        Ok(())
    }

    pub(super) fn predicate(&mut self) -> Result<(), DelaunayInsertionError> {
        self.predicate_evaluations += 1;
        if self.predicate_evaluations > self.options.maximum_predicate_evaluations {
            return Err(resource("predicate-evaluation limit exceeded"));
        }
        if self
            .predicate_evaluations
            .is_multiple_of(self.options.topology.cancellation_check_interval)
            && self.cancellation.is_cancelled()
        {
            return Err(error(DelaunayInsertionErrorKind::Cancelled, "cancelled"));
        }
        Ok(())
    }

    pub(super) fn cancellation(&self) -> &'a dyn MeshingCancellationSignal {
        self.cancellation
    }

    pub(super) fn inventory(&self) -> (u64, u64, u64) {
        (
            self.predicate_evaluations,
            self.checkpoints,
            self.legalization_flips,
        )
    }
}
