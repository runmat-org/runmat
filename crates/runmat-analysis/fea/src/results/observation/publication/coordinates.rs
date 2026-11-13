//! Compare original source coordinates one at a time; no copied field or coordinate inventory.
use super::*;

pub(super) fn validate_history_coordinates(
    source: &ObservationFieldCatalog,
    producer: StepIdentity,
    plan: &ResolvedObservationPlan,
    observation: &ResolvedObservation,
    history: &ObservationHistoryManifest,
) -> AnalysisExecutionResult<()> {
    let mut expected = history.samples.iter();
    let mut visit = |field: &crate::results::field::RecoveredFieldArtifactEntry| -> AnalysisExecutionResult<()> {
        if field.definition.identity() == observation.source_field_definition_identity
            && resolved_observation_accepts_sample(plan, observation, field.sample, None).map_err(invalid)?
            && expected.next().map(|sample| sample.coordinate) != Some(field.sample)
        {
            return Err(invalid("history omits or substitutes an accepted source coordinate"));
        }
        Ok(())
    };
    match source.source() {
        ObservationFieldCatalogSource::SampledRecovery { .. } => {
            return Err(invalid("sampled native histories are not admitted"))
        }
        ObservationFieldCatalogSource::Structural { manifest, .. }
        | ObservationFieldCatalogSource::VolumeMeshRecovery { manifest, .. } => {
            for field in &manifest.fields {
                visit(field)?;
            }
        }
        ObservationFieldCatalogSource::EntryHistory { samples } => {
            for sample in samples {
                let (recovery_request, manifest) = sample.entry_recovery()?;
                if recovery_request.producer != producer {
                    return Err(invalid("entry history producer differs"));
                }
                for field in &manifest.fields {
                    visit(field)?;
                }
            }
        }
        ObservationFieldCatalogSource::Thermal { manifest } => {
            if manifest.producer != producer {
                return Err(invalid("thermal history producer differs"));
            }
            for field in manifest.fields.iter().flatten() {
                visit(field)?;
            }
        }
    }
    if expected.next().is_some() {
        return Err(invalid(
            "history omits or substitutes an accepted source coordinate",
        ));
    }
    Ok(())
}
