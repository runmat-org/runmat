//! Exact retained numerical history payload, a necessary floor rather than total solve memory.
use super::{ByteCount, Count, ResourcePlanningError};
use crate::execution::ThermalSolveSchedule;
use runmat_meshing_core::SolverMeshArtifact;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThermalHistoryResourceFacts {
    pub samples: Count,
    pub temperature_values: Count,
    pub recovered_element_values: Count,
    pub retained_field_bytes: ByteCount,
}

impl ThermalHistoryResourceFacts {
    pub fn derive(
        mesh: &SolverMeshArtifact,
        schedule: &ThermalSolveSchedule,
    ) -> Result<Self, ResourcePlanningError> {
        Self::from_counts(
            mesh.topology.nodes.len(),
            mesh.topology.volume_elements.len(),
            schedule.sample_count(),
        )
    }

    fn from_counts(
        nodes: usize,
        elements: usize,
        samples: usize,
    ) -> Result<Self, ResourcePlanningError> {
        let count = |value| {
            u64::try_from(value)
                .map_err(|_| ResourcePlanningError::Overflow("thermal history cardinality"))
        };
        let samples = count(samples)?;
        let temperature_values =
            count(nodes)?
                .checked_mul(samples)
                .ok_or(ResourcePlanningError::Overflow(
                    "thermal temperature history",
                ))?;
        let recovered_element_values = count(elements)?
            .checked_mul(7)
            .and_then(|value| value.checked_mul(samples))
            .ok_or(ResourcePlanningError::Overflow("thermal element history"))?;
        let retained_field_bytes = temperature_values
            .checked_add(recovered_element_values)
            .and_then(|value| value.checked_mul(8))
            .ok_or(ResourcePlanningError::Overflow(
                "thermal retained field bytes",
            ))?;
        Ok(Self {
            samples: Count(samples),
            temperature_values: Count(temperature_values),
            recovered_element_values: Count(recovered_element_values),
            retained_field_bytes: ByteCount(retained_field_bytes),
        })
    }

    /// Reject a ceiling already smaller than mandatory retained field values. Passing this
    /// necessary check does not admit operators, solver proof/workspace, metadata or transport.
    pub(crate) fn require_fits(&self, maximum: u64) -> Result<(), ResourcePlanningError> {
        if self.retained_field_bytes.0 > maximum {
            return Err(ResourcePlanningError::AccuracyCeiling {
                resource: "thermal retained field bytes (excluding workspace)",
                required: self.retained_field_bytes.0,
                maximum,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn thermal_history_counts_all_four_fields_and_checks_overflow() {
        let facts = ThermalHistoryResourceFacts::from_counts(10, 1, 5).unwrap();
        assert_eq!(facts.temperature_values, Count(50));
        assert_eq!(facts.recovered_element_values, Count(35));
        assert_eq!(facts.retained_field_bytes, ByteCount(680));
        assert!(facts.require_fits(679).is_err());
        assert!(facts.require_fits(680).is_ok());
        assert!(
            ThermalHistoryResourceFacts::from_counts(usize::MAX, usize::MAX, usize::MAX).is_err()
        );
    }
}
