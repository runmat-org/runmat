use runmat_analysis_core::{
    AnalysisModel, ProcedureStepId, ReferenceStatePolicy, StateCompleteness, StateId, StateKind,
    StateOutput, StatePersistence, StateSource, StateTransition, StateTransitionMethod, StateUse,
    TopologyTransitionPolicy,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};

use crate::contracts::canonical_digest::{canonical_contract_digest, wire_bounded};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, SequenceIdentity, SolverResultIdentity,
    StateIdentity, StateTransitionIdentity, StepIdentity,
};

const STATE_PREFIX: &[u8] = b"runmat-analysis-semantic-state/v1\0";
/// Semantic states carry complete fields of any admitted mesh. Their size is admitted by the
/// producing resource plan and checked against it by the object inventory, so the codec bounds
/// them only by the wire range; identities are digested from the streamed encoding.
const STATE_LIMITS: CanonicalLimits = wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

pub(super) mod field_validation;
mod resource_profile;
mod retention;
pub use resource_profile::SemanticStatePublicationProfileSource;
pub use retention::semantic_state_retained_storage_bytes;
mod acoustic;
mod electrical;
mod fluid;
mod line_parameters;
mod magnetic;
mod maxwell;
mod nonlinear;
pub use electrical::CanonicalElectricalStaticStatePayload;
pub use fluid::{
    fluid_state_publication_profile, fluid_state_publication_requests,
    fluid_state_publication_workspace_bytes, CanonicalFluidStatePayload,
    CanonicalFluidStatePayloadContents,
};
pub use line_parameters::CanonicalLineParametersStatePayload;
pub use magnetic::CanonicalMagnetostaticStatePayload;
pub use maxwell::CanonicalMaxwellStatePayload;
pub use nonlinear::{
    nonlinear_solid_state_publication_requests, CanonicalNonlinearSolidStatePayload,
};
mod payload;
mod thermal;
mod thermal_transfer;
pub use acoustic::{AcousticPhasorConvention, CanonicalAcousticStatePayload};
use field_validation::{select_structural_state_fields, validate_portable_fields};
pub use payload::CanonicalSemanticStatePayload;
pub use thermal::CanonicalThermalStatePayload;
pub use thermal_transfer::{BoundThermalStateTransfer, SolvedTemperatureField};

mod publication_request;
pub use publication_request::{SemanticStatePublicationRequest, SemanticStatePublicationSource};
mod publication_profiles;
pub use publication_profiles::{
    StatePublicationProfileEntry, StatePublicationProfiles, STATE_PUBLICATION_PROFILES_MAX_ENTRIES,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalSemanticState {
    pub schema_version: u16,
    pub identity: StateIdentity,
    pub content_digest: Digest,
    pub producer: StepIdentity,
    pub state_id: StateId,
    pub kind: StateKind,
    pub completeness: StateCompleteness,
    pub persistence: StatePersistence,
    pub topology_digest: Digest,
    pub source_result: SolverResultIdentity,
    pub payload: CanonicalSemanticStatePayload,
}

pub struct CanonicalSemanticStateInput {
    pub producer: StepIdentity,
    pub declaration: StateOutput,
    pub topology_digest: Digest,
    pub source_result: SolverResultIdentity,
    pub payload: CanonicalSemanticStatePayload,
}

mod publication;

impl CanonicalSemanticState {
    pub fn new(input: CanonicalSemanticStateInput) -> AnalysisExecutionResult<Self> {
        let mut state = Self {
            schema_version: 1,
            identity: StateIdentity::new(
                input.producer,
                Digest::sha256([]),
                input.topology_digest,
                input.source_result,
                Digest::sha256([]),
            )?,
            content_digest: Digest::sha256([]),
            producer: input.producer,
            state_id: input.declaration.state_id,
            kind: input.declaration.kind,
            completeness: input.declaration.completeness,
            persistence: input.declaration.persistence,
            topology_digest: input.topology_digest,
            source_result: input.source_result,
            payload: input.payload,
        };
        state.content_digest = state.derive_content_digest()?;
        state.identity = state.derive_identity()?;
        state.validate()?;
        Ok(state)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1 || self.state_id.0.trim().is_empty() {
            return Err(AnalysisExecutionError::Invalid(
                "semantic state metadata or field inventory is incomplete".into(),
            ));
        }
        if self.completeness != StateCompleteness::Complete
            && self.persistence == StatePersistence::Canonical
        {
            return Err(AnalysisExecutionError::Invalid(
                "only complete semantic state may have canonical persistence".into(),
            ));
        }
        match &self.payload {
            CanonicalSemanticStatePayload::NonlinearSolid { state } => state.validate_for(self)?,
            CanonicalSemanticStatePayload::Fields { fields } => {
                if matches!(
                    self.kind,
                    StateKind::Thermal
                        | StateKind::Fluid
                        | StateKind::ElectromagneticStatic
                        | StateKind::ElectromagneticComplex
                        | StateKind::AcousticComplex
                        | StateKind::ModalBasis
                        | StateKind::StructuralComplex
                        | StateKind::StructuralResponseEnvelope
                        | StateKind::StructuralRandomStatistics
                        | StateKind::MaterialHistory
                ) || fields.is_empty()
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "field semantic-state payload does not match its declared state kind"
                            .into(),
                    ));
                }
                validate_portable_fields(fields)?;
            }
            CanonicalSemanticStatePayload::Thermal { state } => {
                if self.kind != StateKind::Thermal {
                    return Err(AnalysisExecutionError::Invalid(
                        "thermal semantic-state payload requires thermal state kind".into(),
                    ));
                }
                state.validate()?;
            }
            CanonicalSemanticStatePayload::Fluid { state } => {
                // A coupled flow study (conjugate heat transfer) declares its complete state as
                // the coupled interface state; its payload is the same retained flow result.
                if !matches!(self.kind, StateKind::Fluid | StateKind::CoupledInterface)
                    || self.completeness != StateCompleteness::Complete
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "fluid semantic payload requires complete fluid state".into(),
                    ));
                }
                state.validate_for(self.topology_digest, self.source_result, self.producer)?;
                if !state.workload.model.sequence.steps[0]
                    .state_outputs
                    .iter()
                    .any(|d| {
                        d.state_id == self.state_id
                            && d.kind == self.kind
                            && d.completeness == self.completeness
                            && d.persistence == self.persistence
                    })
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "fluid state declaration differs from original authored authority".into(),
                    ));
                }
            }
            CanonicalSemanticStatePayload::Acoustic { state } => {
                if self.kind != StateKind::AcousticComplex
                    || self.completeness != StateCompleteness::Complete
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "acoustic semantic-state payload requires acoustic-complex state kind"
                            .into(),
                    ));
                }
                state.validate_for(self.topology_digest, self.source_result, self.producer)?;
            }
            CanonicalSemanticStatePayload::ElectricalStatic { state } => {
                if self.kind != StateKind::ElectromagneticStatic
                    || self.completeness != StateCompleteness::Complete
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "electrical static payload requires complete electromagnetic-static state"
                            .into(),
                    ));
                }
                state.validate_for(self.topology_digest, self.source_result, self.producer)?;
            }
            CanonicalSemanticStatePayload::LineParameters { state } => {
                if self.kind != StateKind::ElectromagneticStatic
                    || self.completeness != StateCompleteness::Complete
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "line-parameter payload requires complete electromagnetic-static state"
                            .into(),
                    ));
                }
                state.validate_for(self.source_result, self.producer)?;
            }
            CanonicalSemanticStatePayload::Magnetostatic { state } => {
                if self.kind != StateKind::ElectromagneticStatic
                    || self.completeness != StateCompleteness::Complete
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "magnetostatic payload requires complete electromagnetic-static state"
                            .into(),
                    ));
                }
                state.validate_for(self.topology_digest, self.source_result, self.producer)?;
                if !state
                    .result
                    .publication_declarations()
                    .iter()
                    .any(|declaration| {
                        declaration.state_id == self.state_id
                            && declaration.kind == self.kind
                            && declaration.completeness == self.completeness
                            && declaration.persistence == self.persistence
                    })
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "magnetostatic state declaration differs from retained authored authority"
                            .into(),
                    ));
                }
            }
            CanonicalSemanticStatePayload::MaxwellHarmonic { state } => {
                if self.kind != StateKind::ElectromagneticComplex
                    || self.completeness != StateCompleteness::Complete
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "Maxwell payload requires complete electromagnetic-complex state".into(),
                    ));
                }
                state.validate_for(self.topology_digest, self.source_result, self.producer)?;
                if !state
                    .result
                    .publication_declarations()
                    .iter()
                    .any(|declaration| {
                        declaration.state_id == self.state_id
                            && declaration.kind == self.kind
                            && declaration.completeness == self.completeness
                            && declaration.persistence == self.persistence
                    })
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "Maxwell state declaration differs from retained authored authority".into(),
                    ));
                }
            }
            CanonicalSemanticStatePayload::StructuralFrequency { publication } => {
                if self.kind != StateKind::StructuralComplex {
                    return Err(AnalysisExecutionError::Invalid(
                        "structural-frequency semantic-state payload does not match its declared state kind"
                            .into(),
                    ));
                }
                publication.validate()?;
                if publication.result().identity() != self.source_result {
                    return Err(AnalysisExecutionError::Identity(
                        "structural-frequency semantic state names another solver result",
                    ));
                }
            }
            CanonicalSemanticStatePayload::ModalBasis { basis } => {
                if self.kind != StateKind::ModalBasis || basis.modes().is_empty() {
                    return Err(AnalysisExecutionError::Invalid(
                        "modal-basis semantic-state payload does not match its declared state kind"
                            .into(),
                    ));
                }
            }
            CanonicalSemanticStatePayload::StructuralLinearization { linearization } => {
                if self.kind != StateKind::GeometricStiffness
                    || linearization.topology_digest() != self.topology_digest
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "structural-linearization payload does not match its declared state kind or topology"
                            .into(),
                    ));
                }
                linearization
                    .validate()
                    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
                if let Some(source) = linearization.thermal_source() {
                    if source.transition().consumer != self.producer
                        || source.structural_result() != self.source_result
                    {
                        return Err(AnalysisExecutionError::Identity(
                            "thermal preload publication names another structural producer or result",
                        ));
                    }
                }
            }
            CanonicalSemanticStatePayload::StructuralResponseEnvelope { envelope } => {
                if self.kind != StateKind::StructuralResponseEnvelope
                    || envelope.scope().topology_digest() != self.topology_digest
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "structural response-envelope payload does not match its declared state kind or topology"
                            .into(),
                    ));
                }
                envelope.validate()?;
            }
            CanonicalSemanticStatePayload::StructuralRandomStatistics { statistics } => {
                if self.kind != StateKind::StructuralRandomStatistics
                    || statistics.scope().topology_digest() != self.topology_digest
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "structural random-statistics payload does not match its declared state kind or topology"
                            .into(),
                    ));
                }
                statistics.validate()?;
            }
        }
        let content_digest = self.derive_content_digest()?;
        if self.content_digest != content_digest || self.identity != self.derive_identity()? {
            return Err(AnalysisExecutionError::Identity(
                "semantic state identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            STATE_PREFIX,
            "semantic-state",
            self,
            STATE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let state: Self = decode_contract(STATE_PREFIX, "semantic-state", bytes, STATE_LIMITS)?;
        state.validate()?;
        Ok(state)
    }

    fn derive_content_digest(&self) -> AnalysisExecutionResult<Digest> {
        Ok(canonical_contract_digest(
            STATE_PREFIX,
            "semantic-state-content",
            &(self.kind, self.topology_digest, &self.payload),
            STATE_LIMITS,
        )?)
    }

    fn derive_identity(&self) -> AnalysisExecutionResult<StateIdentity> {
        Self::identity_for(
            self.producer,
            &self.state_id,
            self.kind,
            self.completeness,
            self.persistence,
            self.topology_digest,
            self.source_result,
            self.content_digest,
        )
    }

    /// The identity a state with this header and content digest has. Lets a persisted
    /// reference prove its header without loading the state payload.
    #[allow(clippy::too_many_arguments)]
    pub fn identity_for(
        producer: StepIdentity,
        state_id: &StateId,
        kind: StateKind,
        completeness: StateCompleteness,
        persistence: StatePersistence,
        topology_digest: Digest,
        source_result: SolverResultIdentity,
        content_digest: Digest,
    ) -> AnalysisExecutionResult<StateIdentity> {
        let declaration_digest = canonical_contract_digest(
            STATE_PREFIX,
            "semantic-state-declaration",
            &(state_id, kind, completeness, persistence),
            STATE_LIMITS,
        )?;
        StateIdentity::new(
            producer,
            declaration_digest,
            topology_digest,
            source_result,
            content_digest,
        )
    }
}

mod transition;
pub use transition::{CanonicalStateTransition, CanonicalStateTransitionInput};
