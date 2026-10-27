//! Original displacement, stress/strain and point history; cold integrity is not equilibrium proof.
use super::*;
use crate::execution::{
    NonlinearSolidBlockArtifact, NonlinearSolidNumericalReceiptArtifact, NonlinearSolidWorkload,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalNonlinearSolidStatePayload {
    pub workload: Box<NonlinearSolidWorkload>,
    pub block: Box<NonlinearSolidBlockArtifact>,
}

impl CanonicalNonlinearSolidStatePayload {
    pub fn validate_for(&self, state: &CanonicalSemanticState) -> AnalysisExecutionResult<()> {
        validate_source(&self.workload, &self.block)?;
        if !matches!(
            state.kind,
            StateKind::MaterialHistory
                | StateKind::StructuralPrimary
                | StateKind::StructuralStressStrain
        ) || state.completeness != StateCompleteness::Complete
            || state.producer != self.workload.producer_identity()?
            || state.topology_digest != self.block.request.topology
            || state.source_result != self.block.source_result_identity()?
            || !self.workload.model.sequence.steps[0]
                .state_outputs
                .iter()
                .any(|d| {
                    d.state_id == state.state_id
                        && d.kind == state.kind
                        && d.completeness == state.completeness
                        && d.persistence == state.persistence
                })
        {
            return Err(invalid(
                "nonlinear state differs from original complete declaration and block",
            ));
        }
        Ok(())
    }
}

fn invalid(e: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(e.to_string())
}

fn validate_source(
    workload: &NonlinearSolidWorkload,
    block: &NonlinearSolidBlockArtifact,
) -> AnalysisExecutionResult<()> {
    NonlinearSolidNumericalReceiptArtifact::new(workload, block)?;
    if !block.complete {
        return Err(invalid(
            "partial nonlinear load block is not completed material-history publication",
        ));
    }
    Ok(())
}

pub fn nonlinear_solid_state_publication_requests(
    workload: &NonlinearSolidWorkload,
    block: &NonlinearSolidBlockArtifact,
) -> AnalysisExecutionResult<Vec<SemanticStatePublicationRequest>> {
    validate_source(workload, block)?;
    workload.model.sequence.steps[0]
        .state_outputs
        .iter()
        .filter(|declaration| {
            matches!(
                declaration.kind,
                StateKind::MaterialHistory
                    | StateKind::StructuralPrimary
                    | StateKind::StructuralStressStrain
            )
        })
        .cloned()
        .map(|declaration| {
            SemanticStatePublicationRequest::new(
                SemanticStatePublicationSource::NonlinearSolidOriginalState,
                workload.producer_identity()?,
                declaration,
                block.request.topology,
                block.source_result_identity()?,
            )
        })
        .collect()
}

pub(super) fn validate_request(
    request: &SemanticStatePublicationRequest,
    workload: &NonlinearSolidWorkload,
    block: &NonlinearSolidBlockArtifact,
) -> AnalysisExecutionResult<()> {
    if !nonlinear_solid_state_publication_requests(workload, block)?.contains(request) {
        return Err(invalid(
            "nonlinear state request differs from original authored output",
        ));
    }
    Ok(())
}

impl CanonicalSemanticState {
    pub fn from_nonlinear_solid_publication(
        request: &SemanticStatePublicationRequest,
        workload: &NonlinearSolidWorkload,
        block: &NonlinearSolidBlockArtifact,
    ) -> AnalysisExecutionResult<Self> {
        validate_request(request, workload, block)?;
        // Borrowed shape reservation precedes cloning; whole-task/RSS admission is separate.
        let profile = request.encoding_profile(
            SemanticStatePublicationProfileSource::NonlinearSolidOriginalState { workload, block },
        )?;
        let required = profile
            .materialization_resident_bytes_estimate
            .checked_add(profile.encoded_bytes_upper_bound)
            .ok_or_else(|| invalid("material history construction overflow"))?;
        if required
            > workload.model.sequence.steps[0]
                .accuracy
                .resources
                .resident_byte_budget()
        {
            return Err(invalid(
                "material history construction exceeds original authored ceiling",
            ));
        }
        Self::new(CanonicalSemanticStateInput {
            producer: request.producer,
            declaration: request.declaration.clone(),
            topology_digest: request.topology_digest,
            source_result: request.source_result,
            payload: CanonicalSemanticStatePayload::NonlinearSolid {
                state: std::sync::Arc::new(CanonicalNonlinearSolidStatePayload {
                    workload: Box::new(workload.clone()),
                    block: Box::new(block.clone()),
                }),
            },
        })
    }
}
