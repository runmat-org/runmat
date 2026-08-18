use runmat_meshing_core::MeshingCancellationSignal;

use super::{
    error, resource, DelaunayCarvingError, DelaunayCarvingErrorKind, DelaunayCarvingOptions,
};

pub(super) struct CarvingWork<'a> {
    pub(super) options: DelaunayCarvingOptions,
    pub(super) cancellation: &'a dyn MeshingCancellationSignal,
    flood_steps: u64,
}

impl<'a> CarvingWork<'a> {
    pub(super) fn new(
        options: DelaunayCarvingOptions,
        cancellation: &'a dyn MeshingCancellationSignal,
    ) -> Self {
        Self {
            options,
            cancellation,
            flood_steps: 0,
        }
    }

    pub(super) fn flood(&mut self) -> Result<(), DelaunayCarvingError> {
        self.flood_steps += 1;
        if self.flood_steps > self.options.maximum_flood_steps {
            return Err(resource(None, "carving flood-step limit exceeded"));
        }
        self.cancelled(self.flood_steps)
    }

    fn cancelled(&self, step: u64) -> Result<(), DelaunayCarvingError> {
        if step.is_multiple_of(
            self.options
                .facet_recovery
                .segment_recovery
                .constraints
                .cancellation_check_interval,
        ) && self.cancellation.is_cancelled()
        {
            return Err(error(
                DelaunayCarvingErrorKind::Cancelled,
                None,
                "cancelled",
            ));
        }
        Ok(())
    }
}
