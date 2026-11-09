//! Complete entry source authority for terminal finalization of a history that spans the
//! entries of a sampled study (frequencies of a sweep, for example).
use super::*;
use crate::{
    execution::{FieldRecoverySource, StructuralFrequencyWorkload},
    results::field::RecoveredFieldSample,
};

impl ObservationFieldCatalog {
    /// The complete source authority of a history over a sampled study's entries.
    pub fn entry_history(samples: Vec<Self>) -> AnalysisExecutionResult<Self> {
        Self::new(ObservationFieldCatalogSource::EntryHistory { samples })
    }

    pub fn validate_frequency_history_plan(
        &self,
        plan: &ResolvedObservationPlan,
        workload: &StructuralFrequencyWorkload,
        mesh: Option<&SolverMeshArtifact>,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        workload.validate()?;
        let ObservationFieldCatalogSource::EntryHistory { samples } = self.source() else {
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

impl ObservationFieldCatalog {
    /// The recovery request and field manifest of one entry child of an entry history.
    pub(crate) fn entry_recovery(
        &self,
    ) -> AnalysisExecutionResult<(&FieldRecoveryRequest, &RecoveredFieldArtifactManifest)> {
        match self.source() {
            ObservationFieldCatalogSource::Structural {
                recovery_request,
                manifest,
            }
            | ObservationFieldCatalogSource::VolumeMeshRecovery {
                recovery_request,
                manifest,
                ..
            } => Ok((recovery_request, manifest)),
            _ => Err(invalid("entry history child is not an individual recovery")),
        }
    }

    /// Resolve the plan against every volume-mesh entry of an entry history on the mesh the
    /// entries were solved on. Every entry carries the producing authored step.
    pub fn validate_volume_mesh_history_plan(
        &self,
        plan: &ResolvedObservationPlan,
        producer: crate::execution::StepIdentity,
        step: &runmat_analysis_core::ProcedureStep,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        let ObservationFieldCatalogSource::EntryHistory { samples } = self.source() else {
            return Err(invalid(
                "entry history requires the complete source catalog",
            ));
        };
        for sample in samples {
            let ObservationFieldCatalogSource::VolumeMeshRecovery {
                recovery_request,
                step: producing_step,
                ..
            } = sample.source()
            else {
                return Err(invalid(
                    "volume-mesh entry history contains another source kind",
                ));
            };
            if recovery_request.producer != producer || producing_step.as_ref() != step {
                return Err(invalid(
                    "volume-mesh entry history producer or authored step differs",
                ));
            }
            sample.validate_volume_mesh_plan(plan, mesh)?;
        }
        Ok(())
    }
}

/// The single sample coordinate of one entry catalog and its increasing-order key.
fn entry_sample(
    sample: &ObservationFieldCatalog,
) -> AnalysisExecutionResult<(RecoveredFieldSample, f64)> {
    let manifest = match sample.source() {
        ObservationFieldCatalogSource::Structural { manifest, .. }
        | ObservationFieldCatalogSource::VolumeMeshRecovery { manifest, .. } => manifest,
        _ => {
            return Err(invalid(
                "entry history requires individual structural or volume-mesh recoveries",
            ))
        }
    };
    let Some(coordinate) = manifest.fields.first().map(|field| field.sample) else {
        return Err(invalid("entry history source has no sample"));
    };
    let key = match coordinate {
        RecoveredFieldSample::TimeSeconds(value)
        | RecoveredFieldSample::FrequencyHertz(value)
        | RecoveredFieldSample::LoadFactor(value)
        | RecoveredFieldSample::StudyParameterValue(value) => value,
        RecoveredFieldSample::ModeIndex(index) => index as f64,
        RecoveredFieldSample::Final => {
            return Err(invalid("entry history source has no ordered sample"))
        }
    };
    if !key.is_finite()
        || manifest
            .fields
            .iter()
            .any(|field| field.sample != coordinate)
    {
        return Err(invalid("entry history source mixes sample coordinates"));
    }
    Ok((coordinate, key))
}

/// Children share one kind, producer, topology and field-definition inventory; each is one
/// distinct entry at one sample, in strictly increasing sample order of a single sample kind.
pub(super) fn validate_samples(samples: &[ObservationFieldCatalog]) -> AnalysisExecutionResult<()> {
    let Some(first) = samples.first() else {
        return Err(invalid("entry history source catalog is empty"));
    };
    type EntryIdentity<'a> = (
        u8,
        &'a FieldRecoveryRequest,
        &'a RecoveredFieldArtifactManifest,
        Option<(
            &'a crate::execution::SequenceIdentity,
            &'a runmat_analysis_core::ProcedureStep,
        )>,
    );
    fn identity(catalog: &ObservationFieldCatalog) -> AnalysisExecutionResult<EntryIdentity<'_>> {
        match catalog.source() {
            ObservationFieldCatalogSource::Structural {
                recovery_request,
                manifest,
            } => {
                if !matches!(
                    recovery_request.source,
                    FieldRecoverySource::StructuralFrequencyEntry(_)
                ) {
                    return Err(invalid(
                        "structural entry history source is not a frequency entry",
                    ));
                }
                Ok((0_u8, recovery_request.as_ref(), manifest.as_ref(), None))
            }
            ObservationFieldCatalogSource::VolumeMeshRecovery {
                recovery_request,
                manifest,
                sequence,
                step,
            } => Ok((
                1_u8,
                recovery_request.as_ref(),
                manifest.as_ref(),
                Some((sequence, step.as_ref())),
            )),
            _ => Err(invalid(
                "entry history requires individual structural or volume-mesh recoveries",
            )),
        }
    }
    let (kind, request, fields, authored) = identity(first)?;
    let mut sources = std::collections::BTreeSet::new();
    let mut previous: Option<(RecoveredFieldSample, f64)> = None;
    for sample in samples {
        sample.validate()?;
        let (sample_kind, recovery_request, manifest, sample_authored) = identity(sample)?;
        if !sources.insert(recovery_request.source) {
            return Err(invalid("entry history source is not a unique entry"));
        }
        let (coordinate, key) = entry_sample(sample)?;
        if previous.is_some_and(|(prior, value)| {
            std::mem::discriminant(&prior) != std::mem::discriminant(&coordinate) || key <= value
        }) {
            return Err(invalid(
                "entry history samples are inconsistent or unordered",
            ));
        }
        if sample_kind != kind
            || recovery_request.producer != request.producer
            || recovery_request.topology != request.topology
            || sample_authored != authored
            || manifest
                .fields
                .iter()
                .map(|field| &field.definition)
                .ne(fields.fields.iter().map(|field| &field.definition))
        {
            return Err(invalid(
                "entry history mixes kinds, producers, topology, steps or field definitions",
            ));
        }
        previous = Some((coordinate, key));
    }
    Ok(())
}
