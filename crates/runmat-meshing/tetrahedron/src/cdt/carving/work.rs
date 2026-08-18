use runmat_meshing_core::MeshingCancellationSignal;

use super::{
    error, resource, DelaunayCarvingError, DelaunayCarvingErrorKind, DelaunayCarvingOptions,
};

pub(super) struct CarvingWork<'a> {
    pub(super) options: DelaunayCarvingOptions,
    pub(super) cancellation: &'a dyn MeshingCancellationSignal,
    location_steps: u64,
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
            location_steps: 0,
            flood_steps: 0,
        }
    }

    pub(super) fn location(&mut self, seed_index: u32) -> Result<(), DelaunayCarvingError> {
        self.location_steps += 1;
        if self.location_steps > self.options.maximum_location_steps {
            return Err(resource(
                Some(seed_index),
                "carving seed-location limit exceeded",
            ));
        }
        self.cancelled(Some(seed_index), self.location_steps)
    }

    pub(super) fn flood(&mut self) -> Result<(), DelaunayCarvingError> {
        self.flood_steps += 1;
        if self.flood_steps > self.options.maximum_flood_steps {
            return Err(resource(None, "carving flood-step limit exceeded"));
        }
        self.cancelled(None, self.flood_steps)
    }

    fn cancelled(&self, seed_index: Option<u32>, step: u64) -> Result<(), DelaunayCarvingError> {
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
                seed_index,
                "cancelled",
            ));
        }
        Ok(())
    }
}
