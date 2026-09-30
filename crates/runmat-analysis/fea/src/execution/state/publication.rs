use super::*;
use crate::execution::{
    StructuralModalResponseSpectrumSolverResult, StructuralRandomResponseSolverResult,
    StructuralRandomStatistics, StructuralResponseEnvelope,
};
use runmat_analysis_core::AnalysisField;

impl CanonicalSemanticState {
    pub fn from_publication(
        request: &SemanticStatePublicationRequest,
        fields: &[AnalysisField],
    ) -> AnalysisExecutionResult<Self> {
        request.validate()?;
        if request.source != SemanticStatePublicationSource::StructuralPrimaryFields {
            return Err(AnalysisExecutionError::Invalid(
                "field state publication requires a structural-primary source".into(),
            ));
        }
        let selected = select_structural_state_fields(request.declaration.kind, fields)?
            .into_iter()
            .cloned()
            .collect();
        Self::new(CanonicalSemanticStateInput {
            producer: request.producer,
            declaration: request.declaration.clone(),
            topology_digest: request.topology_digest,
            source_result: request.source_result,
            payload: CanonicalSemanticStatePayload::Fields { fields: selected },
        })
    }

    pub fn from_modal_publication(
        request: &SemanticStatePublicationRequest,
        basis: crate::pipeline::modal_basis::ModalBasisArtifact,
    ) -> AnalysisExecutionResult<Self> {
        request.validate()?;
        if request.source != SemanticStatePublicationSource::ModalBasis {
            return Err(AnalysisExecutionError::Invalid(
                "modal-basis state publication requires a modal result source".into(),
            ));
        }
        basis
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        Self::new(CanonicalSemanticStateInput {
            producer: request.producer,
            declaration: request.declaration.clone(),
            topology_digest: request.topology_digest,
            source_result: request.source_result,
            payload: CanonicalSemanticStatePayload::ModalBasis {
                basis: Box::new(basis),
            },
        })
    }

    pub fn from_linearization_publication(
        request: &SemanticStatePublicationRequest,
        linearization: crate::pipeline::structural_linearization::StructuralLinearizationArtifact,
    ) -> AnalysisExecutionResult<Self> {
        request.validate()?;
        if request.source != SemanticStatePublicationSource::StructuralLinearization
            || linearization.topology_digest() != request.topology_digest
        {
            return Err(AnalysisExecutionError::Invalid(
                "geometric-stiffness state publication requires a topology-matched structural linearization"
                    .into(),
            ));
        }
        linearization
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        Self::new(CanonicalSemanticStateInput {
            producer: request.producer,
            declaration: request.declaration.clone(),
            topology_digest: request.topology_digest,
            source_result: request.source_result,
            payload: CanonicalSemanticStatePayload::StructuralLinearization {
                linearization: Box::new(linearization),
            },
        })
    }

    pub fn from_structural_response_envelope_publication(
        request: &SemanticStatePublicationRequest,
        result: &StructuralModalResponseSpectrumSolverResult,
    ) -> AnalysisExecutionResult<Self> {
        request.validate()?;
        if request.source != SemanticStatePublicationSource::StructuralResponseEnvelope
            || result.assembly().scope().topology_digest() != request.topology_digest
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural response-envelope state publication requires a topology-matched response-spectrum result"
                    .into(),
            ));
        }
        Self::new(CanonicalSemanticStateInput {
            producer: request.producer,
            declaration: request.declaration.clone(),
            topology_digest: request.topology_digest,
            source_result: request.source_result,
            payload: CanonicalSemanticStatePayload::StructuralResponseEnvelope {
                envelope: Box::new(StructuralResponseEnvelope::from_result(result)?),
            },
        })
    }

    pub fn from_structural_random_statistics_publication(
        request: &SemanticStatePublicationRequest,
        result: &StructuralRandomResponseSolverResult,
    ) -> AnalysisExecutionResult<Self> {
        request.validate()?;
        if request.source != SemanticStatePublicationSource::StructuralRandomStatistics
            || result.assembly().scope().topology_digest() != request.topology_digest
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural random-statistics state publication requires a topology-matched random-response result"
                    .into(),
            ));
        }
        Self::new(CanonicalSemanticStateInput {
            producer: request.producer,
            declaration: request.declaration.clone(),
            topology_digest: request.topology_digest,
            source_result: request.source_result,
            payload: CanonicalSemanticStatePayload::StructuralRandomStatistics {
                statistics: Box::new(StructuralRandomStatistics::from_result(result)?),
            },
        })
    }
}
