use super::*;
use runmat_analysis_core::{AnalysisModel, ProcedureStep};

pub fn export_initial_field(
    run_id: &str,
    field_id: &str,
    maximum: u64,
) -> Result<Option<InitialFieldBinding>, AnalysisPublicationError> {
    capture_artifact_store()
        .map_err(AnalysisPublicationError::from)?
        .export_initial_field(run_id, field_id, maximum)
}

pub fn resolve_initial_fields(
    model: &AnalysisModel,
    step: &ProcedureStep,
    maximum: u64,
) -> Result<Vec<InitialFieldBinding>, AnalysisPublicationError> {
    let references = model
        .initial_conditions
        .iter()
        .filter(|condition| condition.applies_to_step(step))
        .filter_map(|condition| condition.field_artifact_ref());
    resolve_initial_field_references(references, maximum)
}

/// Resolve a closed source union with one cumulative read/residency allowance.
pub fn resolve_initial_field_references<'a>(
    references: impl Iterator<Item = &'a runmat_analysis_core::InitialFieldArtifactRef>,
    maximum: u64,
) -> Result<Vec<InitialFieldBinding>, AnalysisPublicationError> {
    let mut references = references.peekable();
    if references.peek().is_none() {
        return Ok(Vec::new());
    }
    let store = capture_artifact_store().map_err(AnalysisPublicationError::from)?;
    let mut result: Vec<InitialFieldBinding> = Vec::new();
    let mut remaining = maximum;
    for reference in references {
        if result.iter().any(|field| field.reference() == reference) {
            continue;
        }
        let binding = store
            .export_initial_field(&reference.artifact_id, &reference.field_id, remaining)?
            .ok_or("selected initial field has no authenticated original source catalog")?;
        if binding.reference() != reference {
            return Err(
                "selected initial field reference differs from its original publication".into(),
            );
        }
        let charge = publication::resident(&binding, remaining)?
            .checked_mul(3)
            .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                message: "initial source residency overflow".into(),
            })?;
        remaining = remaining.checked_sub(charge).ok_or_else(|| {
            AnalysisPublicationError::ResourceLimit {
                message: "initial sources exceed cumulative read allowance".into(),
            }
        })?;
        result.push(binding);
    }
    Ok(result)
}
