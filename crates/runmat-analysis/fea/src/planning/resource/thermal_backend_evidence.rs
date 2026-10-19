//! Successful thermal backend evidence, including retained slots and one in-flight record.
use runmat_canonical_codec::{profile_serialized, CanonicalLimits, SerializedCanonicalProfile};

use super::{ByteCount, ResourcePlanningError};
use crate::{execution::ThermalSolveSchedule, solve::accelerator::AcceleratorSolveEvidence};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThermalBackendEvidenceResourceFacts {
    pub maximum_records: usize,
    pub record: SerializedCanonicalProfile,
}

impl ThermalBackendEvidenceResourceFacts {
    pub fn derive(schedule: &ThermalSolveSchedule) -> Result<Self, ResourcePlanningError> {
        Ok(Self {
            maximum_records: schedule.maximum_linear_solve_count()?,
            record: AcceleratorSolveEvidence::completed_profile()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
        })
    }

    pub fn reservation_bytes(&self) -> Result<ByteCount, ResourcePlanningError> {
        let count = u64::try_from(self.maximum_records)
            .map_err(|_| ResourcePlanningError::Overflow("thermal backend evidence count"))?;
        // Slots are reserved lazily on the first actual accelerator record. Their inline
        // storage and one current solve record coexist with the materialization estimate.
        count
            .checked_add(1)
            .and_then(|slots| {
                self.record
                    .materialization_resident_bytes_estimate
                    .checked_add(std::mem::size_of::<AcceleratorSolveEvidence>() as u64)
                    .and_then(|record| slots.checked_mul(record))
            })
            .and_then(|bytes| {
                bytes.checked_add(std::mem::size_of::<Vec<AcceleratorSolveEvidence>>() as u64)
            })
            .map(ByteCount)
            .ok_or(ResourcePlanningError::Overflow(
                "thermal backend evidence residency",
            ))
    }

    pub(crate) fn verify(
        &self,
        evidence: &AcceleratorSolveEvidence,
    ) -> Result<(), ResourcePlanningError> {
        evidence
            .validate()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        if !evidence.satisfies_tolerance() {
            return Err(ResourcePlanningError::Invalid(
                "thermal backend evidence is not an accepted completed solve".into(),
            ));
        }
        let actual = profile_serialized(
            evidence,
            CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 32),
        )
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        if actual.encoded_bytes_upper_bound > self.record.encoded_bytes_upper_bound
            || actual.materialization_resident_bytes_estimate
                > self.record.materialization_resident_bytes_estimate
        {
            return Err(ResourcePlanningError::Invalid(
                "thermal backend evidence exceeds projected metadata".into(),
            ));
        }
        Ok(())
    }
}
