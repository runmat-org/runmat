//! Typed transfer authority for one selected stochastic-evolution source.
//! Uploads share the whole-call receipt but never certify arithmetic execution.

use super::compound::CompoundObservation;
use super::*;
use crate::stochastic_evolution::{
    ProviderStochasticEvolutionSelection, PROVIDER_STOCHASTIC_EVOLUTION_STEP_OPERATION,
};

mod retirement;
pub use retirement::{
    stochastic_evolution_source_transfer_retirement_host_upper_bytes,
    StochasticEvolutionSourceTransferRetirement,
    StochasticEvolutionSourceTransferRetirementNotifier,
};

pub(super) struct StochasticEvolutionTransferObservation {
    pub(super) selection: runmat_execution::Digest,
    pub(super) device_id: u32,
    retirement: Option<StochasticEvolutionSourceTransferRetirementNotifier>,
}

impl StochasticEvolutionTransferObservation {
    pub(super) fn allows_provider_execution_confirmation(&self, device_id: u32) -> bool {
        self.device_id == device_id
    }
}

impl ProviderOperationReceipt {
    /// Admit host-to-provider transfers owned by one exact selected stochastic
    /// source. The right covers the initial state and its ordered noise tiles;
    /// it is bound to the selected device and immutable plan/proposal pair.
    pub fn require_stochastic_evolution_source_transfers(
        &mut self,
        device_id: u32,
        selection: &ProviderStochasticEvolutionSelection,
    ) {
        let operation =
            crate::ProviderOperationIdentity::new(PROVIDER_STOCHASTIC_EVOLUTION_STEP_OPERATION);
        self.require_provider_execution(device_id, operation);
        let valid = selection.proposal.validate_for(&selection.plan);
        let identity = runmat_execution::Digest::sha256_serialized(selection).ok();
        self.0.with(|state| {
            let Some(identity) = identity.filter(|_| valid) else {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
                return;
            };
            if state.registered != 0
                || state.compound.is_some()
                || state.conditional_host_fallback
                || state.conditional_provider_upload
                || state.provider_host_mediated_allowed
                || state.failure.is_some()
                || state.publication.is_some()
                || state.taken
            {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else {
                state.compound = Some(CompoundObservation::StochasticEvolutionTransfer(
                    StochasticEvolutionTransferObservation {
                        selection: identity,
                        device_id,
                        retirement: None,
                    },
                ));
            }
        });
    }
}

impl ProviderOperationRecorder {
    /// Arm the next exact selected source upload. The returned proof resolves
    /// only after the provider's canonical submission-completion owner retires.
    pub fn arm_stochastic_evolution_source_transfer_retirement(
        &self,
        device_id: u32,
    ) -> Result<StochasticEvolutionSourceTransferRetirement, ProviderObservationFailure> {
        let (retirement, notifier) = retirement::pair();
        let armed = self.0.with(|state| {
            if state.failure.is_some() || state.publication.is_some() || state.taken {
                return false;
            }
            let Some(CompoundObservation::StochasticEvolutionTransfer(expected)) =
                state.compound.as_mut()
            else {
                return false;
            };
            if expected.device_id != device_id || expected.retirement.is_some() {
                return false;
            }
            expected.retirement = Some(notifier);
            true
        });
        if armed {
            Ok(retirement)
        } else {
            self.fail(ProviderObservationFailure::ExecutionUnproven);
            Err(ProviderObservationFailure::ExecutionUnproven)
        }
    }

    /// Provider-only transfer of the armed notifier into the next canonical
    /// exact-submission completion owner.
    #[doc(hidden)]
    pub fn take_stochastic_evolution_source_transfer_retirement(
        &self,
        device_id: u32,
    ) -> Option<StochasticEvolutionSourceTransferRetirementNotifier> {
        self.0.with(|state| {
            if state.failure.is_some() || state.publication.is_some() || state.taken {
                return None;
            }
            match state.compound.as_mut() {
                Some(CompoundObservation::StochasticEvolutionTransfer(expected))
                    if expected.device_id == device_id =>
                {
                    expected.retirement.take()
                }
                _ => None,
            }
        })
    }

    /// Revalidate the immutable selected plan/proposal against the transfer
    /// right retained by this receipt before entering the upload sequence.
    pub fn expects_stochastic_evolution_selection(
        &self,
        selection: &ProviderStochasticEvolutionSelection,
    ) -> bool {
        let identity = runmat_execution::Digest::sha256_serialized(selection).ok();
        self.0.with(|state| {
            state.failure.is_none()
                && state.publication.is_none()
                && !state.taken
                && selection.proposal.validate_for(&selection.plan)
                && matches!(
                    &state.compound,
                    Some(CompoundObservation::StochasticEvolutionTransfer(expected))
                        if identity == Some(expected.selection)
                )
        })
    }

    /// True only for transfers belonging to the exact selected stochastic
    /// source on this provider. This merely requests physical upload fencing;
    /// selected arithmetic still needs its independent execution witness.
    pub fn observes_stochastic_evolution_source_transfer(&self, device_id: u32) -> bool {
        self.0.with(|state| {
            state.failure.is_none()
                && state.publication.is_none()
                && !state.taken
                && state
                    .expected_execution
                    .as_ref()
                    .is_some_and(|(expected_device, _)| *expected_device == device_id)
                && matches!(
                    &state.compound,
                    Some(CompoundObservation::StochasticEvolutionTransfer(expected))
                        if expected.device_id == device_id
                )
        })
    }
}
