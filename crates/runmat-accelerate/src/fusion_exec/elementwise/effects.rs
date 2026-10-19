//! Successful provider operations are the replay boundary for a fused span.
//! A later failure retains its original source, but must never authorize the
//! VM to re-evaluate already-consumed inputs or prior side effects.

use runmat_accelerate_api::ProviderExecutionError;

#[derive(Default)]
pub(super) struct ElementwiseEffects {
    successful_provider_operation: bool,
}

impl ElementwiseEffects {
    pub(super) fn succeeded(&mut self) {
        self.successful_provider_operation = true;
    }

    pub(super) fn can_replay_unsupported(&self) -> bool {
        !self.successful_provider_operation
    }

    pub(super) fn classify(&self, error: ProviderExecutionError) -> ProviderExecutionError {
        if self.successful_provider_operation {
            error.after_effect()
        } else {
            error
        }
    }
}
