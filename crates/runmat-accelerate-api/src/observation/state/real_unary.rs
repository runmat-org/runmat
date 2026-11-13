//! Exact consumed unary compound, distinct from Unsupported-only fallback.
use super::compound::CompoundObservation;
use super::*;
use crate::{
    ProviderRealUnaryCompound, ProviderRealUnaryContinuation, ProviderRealUnaryContinuations,
    ProviderRealUnaryExecutionEvidence, ProviderRealUnaryPublication,
};

#[derive(Clone, Copy)]
enum Phase {
    Pending,
    Continuation,
    Publishing(ProviderRealUnaryPublication, u64),
    Completed,
}

pub(super) struct RealUnaryObservation {
    recipe: runmat_execution::Digest,
    source: crate::allocation::TensorViewIdentity,
    continuations: ProviderRealUnaryContinuations,
    phase: Phase,
}
impl RealUnaryObservation {
    fn matches(
        &self,
        recipe: runmat_execution::Digest,
        source: crate::allocation::TensorViewIdentity,
    ) -> bool {
        self.recipe == recipe && self.source == source
    }
}

impl ProviderOperationReceipt {
    pub fn require_real_unary_execution(
        &mut self,
        operation: crate::ProviderOperationIdentity,
        recipe: &ProviderRealUnaryCompound,
        continuations: ProviderRealUnaryContinuations,
    ) {
        self.require_provider_execution(recipe.source().device_id(), operation);
        let identity = recipe.identity().ok();
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
                state.compound = Some(CompoundObservation::RealUnary(RealUnaryObservation {
                    recipe: identity,
                    source: recipe.source().view_identity(),
                    continuations,
                    phase: Phase::Pending,
                }));
            }
        });
    }
}

impl ProviderOperationRecorder {
    pub fn expects_real_unary(&self, recipe: &ProviderRealUnaryCompound) -> bool {
        let identity = recipe.identity().ok();
        self.0.with(|state| {
            state.failure.is_none()
                && state.publication.is_none()
                && !state.taken
                && state
                    .compound
                    .as_ref()
                    .and_then(CompoundObservation::unary)
                    .is_some_and(|expected| {
                        identity == Some(expected.recipe)
                            && expected.source == recipe.source().view_identity()
                            && matches!(expected.phase, Phase::Pending)
                    })
        })
    }
    pub fn confirm_real_unary_execution(&self, evidence: &ProviderRealUnaryExecutionEvidence) {
        self.0.with(|state| {
            let valid = state
                .compound
                .as_ref()
                .and_then(CompoundObservation::unary)
                .is_some_and(|expected| {
                    expected.matches(evidence.recipe(), evidence.source())
                        && matches!(expected.phase, Phase::Pending)
                });
            if !valid || state.registered == 0 || state.publication.is_some() || state.taken {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else if state.failure.is_none() {
                state
                    .compound
                    .as_mut()
                    .and_then(CompoundObservation::unary_mut)
                    .unwrap()
                    .phase = Phase::Completed;
                state.selected_execution = Some(true);
                state.route_outcome = Some(ProviderRouteOutcome::SelectedProvider);
            }
        });
    }
    pub fn begin_real_unary_continuation(&self, evidence: &ProviderRealUnaryContinuation) {
        self.0.with(|state| {
            let valid = state
                .compound
                .as_ref()
                .and_then(CompoundObservation::unary)
                .is_some_and(|expected| {
                    expected.matches(evidence.recipe(), evidence.source())
                        && matches!(expected.phase, Phase::Pending)
                });
            if !valid || state.registered == 0 || state.publication.is_some() || state.taken {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else if state.failure.is_none() {
                state
                    .compound
                    .as_mut()
                    .and_then(CompoundObservation::unary_mut)
                    .unwrap()
                    .phase = Phase::Continuation;
            }
        });
    }
    pub fn begin_real_unary_publication(
        &self,
        evidence: &ProviderRealUnaryContinuation,
        domain: runmat_types::NumericDomain,
    ) {
        self.0.with(|state| {
            let valid = state
                .compound
                .as_ref()
                .and_then(CompoundObservation::unary)
                .is_some_and(|expected| {
                    expected.matches(evidence.recipe(), evidence.source())
                        && matches!(expected.phase, Phase::Continuation)
                });
            if !valid || state.publication.is_some() || state.taken {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else if state.failure.is_none() {
                let checkpoint = state.registered;
                let expected = state
                    .compound
                    .as_mut()
                    .and_then(CompoundObservation::unary_mut)
                    .unwrap();
                let publication = match domain {
                    runmat_types::NumericDomain::Real => expected.continuations.real,
                    runmat_types::NumericDomain::Complex => expected.continuations.complex,
                };
                expected.phase = Phase::Publishing(publication, checkpoint);
            }
        });
    }
    pub fn confirm_real_unary_continuation(
        &self,
        evidence: &ProviderRealUnaryContinuation,
        publication: ProviderRealUnaryPublication,
    ) {
        self.0.with(|state| {
            let valid = state
                .compound
                .as_ref()
                .and_then(CompoundObservation::unary)
                .is_some_and(|expected| {
                    expected.matches(evidence.recipe(), evidence.source())
                        && match expected.phase {
                            Phase::Publishing(actual, checkpoint) => {
                                actual == publication
                                    && (actual == ProviderRealUnaryPublication::Host
                                        || state.registered > checkpoint)
                            }
                            _ => false,
                        }
                });
            if !valid || state.publication.is_some() || state.taken {
                state.fail(ProviderObservationFailure::ExecutionUnproven);
            } else if state.failure.is_none() {
                state
                    .compound
                    .as_mut()
                    .and_then(CompoundObservation::unary_mut)
                    .unwrap()
                    .phase = Phase::Completed;
                state.selected_execution = Some(true);
                state.route_outcome = Some(match publication {
                    ProviderRealUnaryPublication::Host => {
                        ProviderRouteOutcome::DomainHostContinuation
                    }
                    ProviderRealUnaryPublication::ProviderUpload => {
                        ProviderRouteOutcome::DomainProviderUpload
                    }
                });
            }
        });
    }
    pub fn observes_real_unary_upload(&self, device_id: u32) -> bool {
        self.0.with(|state| {
            state.failure.is_none()
                && state.publication.is_none()
                && !state.taken
                && state
                    .expected_execution
                    .as_ref()
                    .is_some_and(|(actual, _)| *actual == device_id)
                && state
                    .compound
                    .as_ref()
                    .and_then(CompoundObservation::unary)
                    .is_some_and(|expected| {
                        matches!(
                            expected.phase,
                            Phase::Publishing(ProviderRealUnaryPublication::ProviderUpload, _)
                        )
                    })
        })
    }
}
