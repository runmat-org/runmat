//! Complete frequency source authority for terminal history finalization.
use super::*;
use crate::{
    execution::{FieldRecoverySource, StructuralFrequencyWorkload},
    results::field::RecoveredFieldSample,
};

impl ObservationFieldCatalog {
    pub fn structural_frequency_history(samples: Vec<Self>) -> AnalysisExecutionResult<Self> {
        Self::new(ObservationFieldCatalogSource::StructuralFrequencyHistory { samples })
    }

    pub fn validate_frequency_history_plan(
        &self,
        plan: &ResolvedObservationPlan,
        workload: &StructuralFrequencyWorkload,
        mesh: Option<&SolverMeshArtifact>,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        workload.validate()?;
        let ObservationFieldCatalogSource::StructuralFrequencyHistory { samples } = self.source()
        else {
            return Err(invalid(
                "frequency history requires the complete source catalog",
            ));
        };
        if samples.len() != workload.plan().frequencies_hz().len() {
            return Err(invalid("frequency history omits original workload samples"));
        }
        for (sample, frequency) in samples.iter().zip(workload.plan().frequencies_hz()) {
            sample.validate_structural_plan(plan, workload.model(), mesh)?;
            let ObservationFieldCatalogSource::Structural { manifest, .. } = sample.source() else {
                return Err(invalid("frequency history contains a nonstructural sample"));
            };
            if manifest.fields.first().map(|field| field.sample)
                != Some(RecoveredFieldSample::FrequencyHertz(*frequency))
            {
                return Err(invalid(
                    "frequency history differs from the original workload schedule",
                ));
            }
        }
        Ok(())
    }
}

pub(super) fn validate_samples(samples: &[ObservationFieldCatalog]) -> AnalysisExecutionResult<()> {
    let mut first = None;
    let mut previous = None;
    let mut sources = std::collections::BTreeSet::new();
    if samples.is_empty() {
        return Err(invalid("frequency history source catalog is empty"));
    }
    for sample in samples {
        let ObservationFieldCatalogSource::Structural {
            recovery_request,
            manifest,
        } = sample.source()
        else {
            return Err(invalid(
                "frequency history requires individual structural recoveries",
            ));
        };
        sample.validate()?;
        if !matches!(
            recovery_request.source,
            FieldRecoverySource::StructuralFrequencyEntry(_)
        ) || !sources.insert(recovery_request.source)
        {
            return Err(invalid(
                "frequency history source is not a unique frequency entry",
            ));
        }
        let Some(RecoveredFieldSample::FrequencyHertz(frequency)) =
            manifest.fields.first().map(|field| field.sample)
        else {
            return Err(invalid("frequency history source has no frequency sample"));
        };
        if !frequency.is_finite()
            || previous.is_some_and(|value| frequency <= value)
            || manifest
                .fields
                .iter()
                .any(|field| field.sample != RecoveredFieldSample::FrequencyHertz(frequency))
        {
            return Err(invalid(
                "frequency history samples are inconsistent or unordered",
            ));
        }
        if let Some((request, fields)) = first {
            let request: &FieldRecoveryRequest = request;
            let fields: &RecoveredFieldArtifactManifest = fields;
            if recovery_request.producer != request.producer
                || recovery_request.topology != request.topology
                || manifest
                    .fields
                    .iter()
                    .map(|field| &field.definition)
                    .ne(fields.fields.iter().map(|field| &field.definition))
            {
                return Err(invalid(
                    "frequency history mixes producers, topology or field definitions",
                ));
            }
        } else {
            first = Some((recovery_request.as_ref(), manifest.as_ref()));
        }
        previous = Some(frequency);
    }
    Ok(())
}
