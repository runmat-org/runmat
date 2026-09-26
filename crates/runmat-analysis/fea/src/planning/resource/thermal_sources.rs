//! Retained thermal projection evidence plus simultaneous assembly/copy records.
use runmat_analysis_core::{AnalysisModel, ExcitationAction, ProcedureStep};
use runmat_canonical_codec::SerializedCanonicalProfile;

mod profile;

use super::{ByteCount, Count, ResourcePlanningError};
use crate::{
    excitation::{active_actions, SpatialProjectionEvidence},
    execution::ThermalSolveSchedule,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThermalSourceEvidenceResourceFacts {
    pub records_per_sample: Count,
    pub retained_records: Count,
    pub record_bytes_per_sample: ByteCount,
    pub simultaneous_record_bytes: ByteCount,
    pub serialization: Option<SerializedCanonicalProfile>,
}

impl ThermalSourceEvidenceResourceFacts {
    pub fn derive(
        model: &AnalysisModel,
        step: &ProcedureStep,
        schedule: &ThermalSolveSchedule,
    ) -> Result<Self, ResourcePlanningError> {
        if !matches!(
            step.independent_domain,
            runmat_analysis_core::IndependentDomain::Static
                | runmat_analysis_core::IndependentDomain::TimeSeconds { .. }
        ) {
            return Err(ResourcePlanningError::Invalid(
                "thermal source evidence requires static or time samples".into(),
            ));
        }
        let mut records = 0_u64;
        let mut record_bytes = 0_u64;
        for selected in active_actions(model, Some(step))
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?
        {
            let action = selected.action;
            if !matches!(
                action,
                ExcitationAction::HeatGeneration(_) | ExcitationAction::HeatFlux(_)
            ) {
                continue;
            }
            let excitation = action.excitation();
            // Static/time samples contain no heap-owned parameter ID. Thermal admission rejects
            // other independent domains before reaching this preflight.
            let bytes = (std::mem::size_of::<SpatialProjectionEvidence>() as u64)
                .checked_add(excitation.excitation_id.0.len() as u64)
                .and_then(|bytes| bytes.checked_add(excitation.signal_id.0.len() as u64))
                .ok_or(ResourcePlanningError::Overflow(
                    "thermal source record bytes",
                ))?;
            records = records
                .checked_add(1)
                .ok_or(ResourcePlanningError::Overflow(
                    "thermal source record count",
                ))?;
            record_bytes =
                record_bytes
                    .checked_add(bytes)
                    .ok_or(ResourcePlanningError::Overflow(
                        "thermal source records per sample",
                    ))?;
        }
        let mut facts = Self::from_counts(records, record_bytes, schedule.sample_count() as u64)?;
        if records != 0 {
            facts.serialization = Some(profile::derive(
                model,
                step,
                schedule,
                facts.retained_records,
            )?);
        }
        Ok(facts)
    }

    fn from_counts(records: u64, bytes: u64, samples: u64) -> Result<Self, ResourcePlanningError> {
        let retained = records
            .checked_mul(samples)
            .ok_or(ResourcePlanningError::Overflow(
                "thermal retained source records",
            ))?;
        // History plus initial assembly, current assembly and cloned extension scratch.
        // Factor two reserves geometric Vec growth (also conservatively applied to strings).
        // Zero-source cases allocate no evidence. This is requested heap payload, not RSS or
        // allocator overhead; realized actions, projection scratch and model storage are separate.
        let simultaneous = samples
            .checked_add(3)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_mul(bytes))
            // Small Vec allocations may reserve four slots even for one record. Four live
            // vectors (history + the three copies above) each receive that extra allowance.
            .and_then(|n| {
                n.checked_add(if records == 0 {
                    0
                } else {
                    16 * std::mem::size_of::<SpatialProjectionEvidence>() as u64
                })
            })
            .ok_or(ResourcePlanningError::Overflow(
                "thermal simultaneous source evidence",
            ))?;
        Ok(Self {
            records_per_sample: Count(records),
            retained_records: Count(retained),
            record_bytes_per_sample: ByteCount(bytes),
            simultaneous_record_bytes: ByteCount(simultaneous),
            serialization: None,
        })
    }

    pub fn reservation_bytes(&self) -> Result<ByteCount, ResourcePlanningError> {
        self.simultaneous_record_bytes
            .0
            .checked_add(
                self.serialization
                    .map_or(0, |profile| profile.materialization_resident_bytes_estimate),
            )
            .map(ByteCount)
            .ok_or(ResourcePlanningError::Overflow(
                "thermal source serialization",
            ))
    }

    pub(crate) fn verify(
        &self,
        evidence: &[SpatialProjectionEvidence],
    ) -> Result<(), ResourcePlanningError> {
        profile::verify(self, evidence)
    }
}

#[cfg(test)]
mod tests;
