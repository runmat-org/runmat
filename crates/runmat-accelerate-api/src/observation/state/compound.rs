//! One exact compound authority per receipt; variants cannot borrow or replace
//! one another's consumed-recipe proof. No source leases/runtime live here.
use super::real_unary::RealUnaryObservation;
use super::stochastic_transfer::StochasticEvolutionTransferObservation;
use super::*;
use crate::power_service::{
    PowerPairCompound, PowerPairExecutionEvidence, PowerScalarExecutionEvidence,
    PowerTypedScalarProjection,
};

pub(super) enum CompoundObservation {
    RealUnary(RealUnaryObservation),
    PowerPair(PowerPairObservation),
    PowerScalar(PowerScalarObservation),
    StochasticEvolutionTransfer(StochasticEvolutionTransferObservation),
}

impl CompoundObservation {
    pub(super) fn unary(&self) -> Option<&RealUnaryObservation> {
        match self {
            Self::RealUnary(value) => Some(value),
            Self::PowerPair(_) | Self::PowerScalar(_) | Self::StochasticEvolutionTransfer(_) => {
                None
            }
        }
    }
    pub(super) fn unary_mut(&mut self) -> Option<&mut RealUnaryObservation> {
        match self {
            Self::RealUnary(value) => Some(value),
            Self::PowerPair(_) | Self::PowerScalar(_) | Self::StochasticEvolutionTransfer(_) => {
                None
            }
        }
    }

    /// Most compound recipes own their own typed completion protocol and must
    /// reject the generic provider confirmer. A selected stochastic source is
    /// the narrow exception: its compound authority governs preparation
    /// transfers while its arithmetic tiles retain the canonical provider
    /// execution witness.
    pub(super) fn allows_provider_execution_confirmation(&self, device_id: u32) -> bool {
        match self {
            Self::StochasticEvolutionTransfer(value) => {
                value.allows_provider_execution_confirmation(device_id)
            }
            Self::RealUnary(_) | Self::PowerPair(_) | Self::PowerScalar(_) => false,
        }
    }
}

pub(super) struct PowerPairObservation {
    recipe: runmat_execution::Digest,
    sources: [crate::allocation::TensorViewIdentity; 2],
    completed: bool,
}

pub(super) struct PowerScalarObservation {
    projection: runmat_execution::Digest,
    source: crate::allocation::TensorViewIdentity,
    completed: bool,
}

impl ProviderOperationReceipt {
    pub fn require_power_scalar_execution(
        &mut self,
        operation: crate::ProviderOperationIdentity,
        projection: &PowerTypedScalarProjection,
        source: &crate::GpuTensorHandle,
    ) {
        self.require_provider_execution(source.device_id(), operation);
        let identity = projection.identity().ok();
        let source_valid = projection.source_matches(source);
        let view = source.view_identity();
        self.0.with(|state| {
            let Some(identity) = identity else {
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
                || !source_valid
            {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else {
                state.compound = Some(CompoundObservation::PowerScalar(PowerScalarObservation {
                    projection: identity,
                    source: view,
                    completed: false,
                }));
            }
        });
    }
    pub fn require_power_pair_execution(
        &mut self,
        operation: crate::ProviderOperationIdentity,
        recipe: &PowerPairCompound,
    ) {
        self.require_provider_execution(recipe.probe().sources()[0].device_id(), operation);
        let identity = recipe.identity().ok();
        let sources = recipe
            .probe()
            .sources()
            .each_ref()
            .map(crate::GpuTensorHandle::view_identity);
        self.0.with(|state| {
            let Some(identity) = identity else {
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
                state.compound = Some(CompoundObservation::PowerPair(PowerPairObservation {
                    recipe: identity,
                    sources,
                    completed: false,
                }));
            }
        });
    }
}

impl ProviderOperationRecorder {
    /// The selected scalar route has exactly one native host-control upload.
    /// Its mapped initialization is followed by the upload owner's queue
    /// completion fence while this pending compound receipt is active.
    pub fn observes_power_scalar_upload(&self, device_id: u32) -> bool {
        self.0.with(|state| {
            state.failure.is_none()
                && state.publication.is_none()
                && !state.taken
                && state
                    .expected_execution
                    .as_ref()
                    .is_some_and(|(expected, _)| *expected == device_id)
                && matches!(&state.compound, Some(CompoundObservation::PowerScalar(expected))
                    if !expected.completed)
        })
    }

    pub fn expects_power_scalar(
        &self,
        projection: &PowerTypedScalarProjection,
        source: &crate::GpuTensorHandle,
    ) -> bool {
        let identity = projection.identity().ok();
        self.0.with(|state| {
            state.failure.is_none()
                && state.publication.is_none()
                && !state.taken
                && projection.source_matches(source)
                && matches!(&state.compound, Some(CompoundObservation::PowerScalar(expected))
                    if identity == Some(expected.projection)
                        && source.view_identity() == expected.source
                        && !expected.completed)
        })
    }

    pub fn confirm_power_scalar_execution(&self, evidence: &PowerScalarExecutionEvidence) {
        self.0.with(|state| {
            let valid = matches!(&state.compound, Some(CompoundObservation::PowerScalar(expected))
                if expected.projection == evidence.projection()
                    && expected.source == evidence.source()
                    && !expected.completed);
            if !valid || state.registered == 0 || state.publication.is_some() || state.taken {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else if state.failure.is_none() {
                if let Some(CompoundObservation::PowerScalar(expected)) = &mut state.compound {
                    expected.completed = true;
                }
                state.selected_execution = Some(true);
                state.route_outcome = Some(ProviderRouteOutcome::SelectedProvider);
            }
        });
    }
    pub fn expects_power_pair(&self, recipe: &PowerPairCompound) -> bool {
        let identity = recipe.identity().ok();
        let sources = recipe
            .probe()
            .sources()
            .each_ref()
            .map(crate::GpuTensorHandle::view_identity);
        self.0.with(|state| {
            state.failure.is_none() && state.publication.is_none() && !state.taken
                && matches!(&state.compound, Some(CompoundObservation::PowerPair(expected))
                    if identity == Some(expected.recipe) && sources == expected.sources && !expected.completed)
        })
    }

    pub fn confirm_power_pair_execution(&self, evidence: &PowerPairExecutionEvidence) {
        self.0.with(|state| {
            let valid = matches!(&state.compound, Some(CompoundObservation::PowerPair(expected))
                if expected.recipe == evidence.recipe() && expected.sources == evidence.sources()
                    && !expected.completed);
            if !valid || state.registered == 0 || state.publication.is_some() || state.taken {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else if state.failure.is_none() {
                if let Some(CompoundObservation::PowerPair(expected)) = &mut state.compound {
                    expected.completed = true;
                }
                state.selected_execution = Some(true);
                state.route_outcome = Some(ProviderRouteOutcome::SelectedProvider);
            }
        });
    }
}
