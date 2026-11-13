//! Retained numerical history payload and shared metadata estimates; not total solve memory.
use super::{field_metadata::FIELD_METADATA_BYTES, ByteCount, Count, ResourcePlanningError};
use crate::execution::ThermalSolveSchedule;
use runmat_meshing_core::SolverMeshArtifact;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThermalHistoryResourceFacts {
    pub samples: Count,
    /// Upper bound on the nodes of finite-conductance interfaces whose second side's
    /// temperature is retained (zero without such an interface).
    pub interface_nodes: usize,
    pub temperature_values: Count,
    pub recovered_element_values: Count,
    pub retained_field_bytes: ByteCount,
    pub retained_metadata_bytes: ByteCount,
    /// One fallibly reserved entity-ID copy; node/element checks are sequential.
    pub validation_scratch_bytes: ByteCount,
}

impl ThermalHistoryResourceFacts {
    /// A finite-conductance thermal interface also retains its second side's temperature at
    /// each interface node (bounded here by every node of a conformal interface face).
    pub fn derive(
        mesh: &SolverMeshArtifact,
        schedule: &ThermalSolveSchedule,
        interfaces: &[runmat_analysis_core::AnalysisInterface],
    ) -> Result<Self, ResourcePlanningError> {
        let conductance = interfaces.iter().any(|interface| {
            matches!(
                &interface.kind,
                runmat_analysis_core::AnalysisInterfaceKind::ConjugateHeatTransfer(model)
                    if !matches!(model.transfer, runmat_analysis_core::ThermalInterfaceTransfer::Perfect)
            )
        });
        let interface_nodes = if conductance {
            mesh.topology
                .boundary_faces
                .iter()
                .filter(|face| {
                    face.role == runmat_meshing_core::BoundaryFaceRole::ConformalInterface
                })
                .flat_map(|face| face.node_ids.iter().copied())
                .collect::<std::collections::BTreeSet<_>>()
                .len()
        } else {
            0
        };
        let mut facts = Self::from_counts(
            mesh.topology.nodes.len() + interface_nodes,
            mesh.topology.volume_elements.len(),
            schedule.sample_count(),
        )?;
        facts.interface_nodes = interface_nodes;
        Ok(facts)
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
        let validation_scratch_bytes = if nodes.max(elements) == 0 {
            0
        } else {
            count(nodes.max(elements))?
                .checked_mul(std::mem::size_of::<u64>() as u64)
                .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Vec<u64>>() as u64))
                .ok_or(ResourcePlanningError::Overflow(
                    "thermal validation scratch",
                ))?
        };
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
        // Four typed fields per sample (five with a conductance interface), including conversion-time Vec headers and field IDs.
        // Additional arrays coexist: schedule points/events, result time points/residuals,
        // six-scalar energy balance, and optional observation coordinates. Reserve S slots
        // even when a particular array contains only S-1 or zero entries.
        let sample_arrays = (4 * std::mem::size_of::<f64>()
            + std::mem::size_of::<crate::contracts::ThermalEnergyBalance>()
            + std::mem::size_of::<crate::contracts::ThermalNonlinearSampleEvidence>()
            + std::mem::size_of::<crate::results::field::RecoveredFieldSample>())
            as u64;
        let retained_metadata_bytes = samples
            .checked_mul(5 * FIELD_METADATA_BYTES + sample_arrays)
            .ok_or(ResourcePlanningError::Overflow("thermal history metadata"))?;
        Ok(Self {
            samples: Count(samples),
            interface_nodes: 0,
            temperature_values: Count(temperature_values),
            recovered_element_values: Count(recovered_element_values),
            retained_field_bytes: ByteCount(retained_field_bytes),
            retained_metadata_bytes: ByteCount(retained_metadata_bytes),
            validation_scratch_bytes: ByteCount(validation_scratch_bytes),
        })
    }

    pub fn reservation_bytes(&self) -> Result<ByteCount, ResourcePlanningError> {
        self.retained_field_bytes
            .0
            .checked_add(self.retained_metadata_bytes.0)
            .and_then(|bytes| bytes.checked_add(self.validation_scratch_bytes.0))
            .map(ByteCount)
            .ok_or(ResourcePlanningError::Overflow(
                "thermal history reservation",
            ))
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
        assert!(facts.retained_metadata_bytes.0 >= 5 * 4 * FIELD_METADATA_BYTES);
        assert_eq!(
            facts.reservation_bytes().unwrap().0,
            facts.retained_field_bytes.0
                + facts.retained_metadata_bytes.0
                + facts.validation_scratch_bytes.0
        );
        let mut overflowing = facts;
        overflowing.retained_metadata_bytes = ByteCount(u64::MAX);
        assert!(overflowing.reservation_bytes().is_err());
        assert!(facts.require_fits(679).is_err());
        assert!(facts.require_fits(680).is_ok());
        assert_eq!(
            facts.validation_scratch_bytes.0,
            (10 * std::mem::size_of::<u64>() + std::mem::size_of::<Vec<u64>>()) as u64
        );
        let element_dominant = ThermalHistoryResourceFacts::from_counts(2, 100, 1).unwrap();
        assert_eq!(
            element_dominant.validation_scratch_bytes.0,
            (100 * std::mem::size_of::<u64>() + std::mem::size_of::<Vec<u64>>()) as u64
        );
        assert!(ThermalHistoryResourceFacts::from_counts(usize::MAX, 1, 1).is_err());
        assert!(
            ThermalHistoryResourceFacts::from_counts(usize::MAX, usize::MAX, usize::MAX).is_err()
        );
    }

    #[test]
    fn metadata_scales_with_samples_even_without_field_values() {
        let one = ThermalHistoryResourceFacts::from_counts(0, 0, 1).unwrap();
        let many = ThermalHistoryResourceFacts::from_counts(0, 0, 257).unwrap();
        assert_eq!(many.retained_field_bytes.0, 0);
        assert_eq!(
            many.retained_metadata_bytes.0,
            257 * one.retained_metadata_bytes.0
        );
        assert_eq!(
            ThermalHistoryResourceFacts::from_counts(0, 0, 0)
                .unwrap()
                .reservation_bytes()
                .unwrap()
                .0,
            0
        );
        if usize::BITS == 64 {
            assert!(ThermalHistoryResourceFacts::from_counts(0, 0, usize::MAX).is_err());
        }
    }
}
