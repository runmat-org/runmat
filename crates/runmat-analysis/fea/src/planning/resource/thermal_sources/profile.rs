//! Borrowed traversal of the retained source history, using the evidence owner's wire schema.
use super::*;
use crate::excitation::{ProjectionSample, SpatialIntegrationRule};
use runmat_analysis_core::{IndependentDomain, PhysicalQuantityKind, SignalSampleApproach};
use runmat_canonical_codec::{profile_serialized, profile_serialized_shape, CanonicalLimits};
use serde::{ser::SerializeSeq, Serialize, Serializer};

const LIMITS: CanonicalLimits = CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64);

struct History<'a> {
    model: &'a AnalysisModel,
    step: &'a ProcedureStep,
    samples: usize,
    records: usize,
    sample: ProjectionSample,
}

impl Serialize for History<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let selected =
            active_actions(self.model, Some(self.step)).map_err(serde::ser::Error::custom)?;
        let mut sequence = serializer.serialize_seq(Some(self.records))?;
        for _ in 0..self.samples {
            for selected in selected.clone() {
                let (quantity, rule) = match selected.action {
                    ExcitationAction::HeatGeneration(_) => (
                        PhysicalQuantityKind::HeatGenerationDensity,
                        SpatialIntegrationRule::VolumeKeastFifteen,
                    ),
                    ExcitationAction::HeatFlux(_) => (
                        PhysicalQuantityKind::HeatFlux,
                        SpatialIntegrationRule::BoundaryDunavantSeven,
                    ),
                    _ => continue,
                };
                let excitation = selected.action.excitation();
                sequence.serialize_element(
                    &SpatialProjectionEvidence::scalar_projection_serialization(
                        &excitation.excitation_id.0,
                        &excitation.signal_id.0,
                        quantity,
                        &self.sample,
                        rule,
                        &excitation.projection,
                    ),
                )?;
            }
        }
        sequence.end()
    }
}

pub(super) fn derive(
    model: &AnalysisModel,
    step: &ProcedureStep,
    schedule: &ThermalSolveSchedule,
    records: Count,
) -> Result<SerializedCanonicalProfile, ResourcePlanningError> {
    let mut bound = SerializedCanonicalProfile {
        encoded_bytes_upper_bound: 0,
        materialization_resident_bytes_estimate: 0,
    };
    // Profile every closed approach, rather than relying on a string-length assumption.
    for approach in [
        SignalSampleApproach::At,
        SignalSampleApproach::LeftLimit,
        SignalSampleApproach::RightLimit,
    ] {
        let sample = match step.independent_domain {
            IndependentDomain::Static => ProjectionSample::Static,
            IndependentDomain::TimeSeconds { .. } => ProjectionSample::TimeSeconds {
                value: 0.0,
                approach,
            },
            _ => {
                return Err(ResourcePlanningError::Invalid(
                    "thermal evidence sample domain".into(),
                ))
            }
        };
        // Stand-in records bound every future value, so integers reserve their full range.
        let actual = profile_serialized_shape(
            &History {
                model,
                step,
                samples: schedule.sample_count(),
                records: usize::try_from(records.0).map_err(|_| {
                    ResourcePlanningError::Overflow("thermal source history length")
                })?,
                sample,
            },
            LIMITS,
        )
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        bound.encoded_bytes_upper_bound = bound
            .encoded_bytes_upper_bound
            .max(actual.encoded_bytes_upper_bound);
        bound.materialization_resident_bytes_estimate = bound
            .materialization_resident_bytes_estimate
            .max(actual.materialization_resident_bytes_estimate);
    }
    Ok(bound)
}

pub(super) fn verify(
    facts: &ThermalSourceEvidenceResourceFacts,
    evidence: &[SpatialProjectionEvidence],
) -> Result<(), ResourcePlanningError> {
    if evidence.len() as u64 != facts.retained_records.0 {
        return Err(ResourcePlanningError::Invalid(
            "thermal source inventory differs from admission".into(),
        ));
    }
    let Some(expected) = facts.serialization else {
        return Ok(());
    };
    let actual = profile_serialized(evidence, LIMITS)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    if actual.encoded_bytes_upper_bound > expected.encoded_bytes_upper_bound
        || actual.materialization_resident_bytes_estimate
            > expected.materialization_resident_bytes_estimate
    {
        return Err(ResourcePlanningError::Invalid(
            "thermal source serialization exceeds admission".into(),
        ));
    }
    Ok(())
}
