//! Immutable preassembly preparation shared by numerical and transport-aware entrypoints.
use super::*;
use crate::execution::{
    AnalysisNumericalObjectKind, AnalysisTransportProfile, AnalysisTransportProfiler,
    ThermalSolveSchedule,
};
use crate::planning::resource::{
    ByteCount, ResourcePlanningError, ThermalPreassemblyResourceFacts,
};
use crate::results::field::{
    RecoveredFieldPartition, RecoveredFieldSample, RECOVERED_FIELD_PARTITION_ENTITY_COUNT,
};

pub struct PreparedThermalSolve<'a> {
    pub(super) model: &'a AnalysisModel,
    pub(super) backend: ComputeBackend,
    pub(super) options: ThermalSolveOptions,
    pub(super) mesh: &'a SolverMeshArtifact,
    pub(super) step: &'a ProcedureStep,
    pub(super) schedule: ThermalSolveSchedule,
    pub(super) facts: ThermalPreassemblyResourceFacts,
}

impl PreparedThermalSolve<'_> {
    pub fn numerical_reservation_bytes(&self) -> Result<ByteCount, ResourcePlanningError> {
        self.facts.reservation_bytes()
    }

    /// Transport is additional to conservatively retained numerical/producer/encoding storage.
    pub fn require_output_reservation(
        &self,
        output: ByteCount,
    ) -> Result<ByteCount, ResourcePlanningError> {
        let required = self
            .numerical_reservation_bytes()?
            .0
            .checked_add(output.0)
            .ok_or(ResourcePlanningError::Overflow(
                "thermal numerical and output residency",
            ))?;
        let maximum = self.step.accuracy.resources.maximum_resident_bytes;
        if required > maximum {
            return Err(ResourcePlanningError::AccuracyCeiling {
                resource: "thermal numerical and output residency",
                required,
                maximum,
            });
        }
        Ok(ByteCount(required))
    }

    /// Bounds the header and every recovered-field block without numerical values or invented IDs.
    /// The last sample's decimal name suffix bounds every earlier name; float widths are fixed.
    pub fn transport_profile(
        &self,
        chunk_bytes: usize,
    ) -> Result<AnalysisTransportProfile, FeaRunError> {
        let invalid = |error: crate::execution::AnalysisExecutionError| {
            FeaRunError::InvalidModel(error.to_string())
        };
        let mut profile = AnalysisTransportProfiler::new(chunk_bytes).map_err(invalid)?;
        crate::execution::ThermalSolverResultManifest::require_header_byte_bound(
            self.facts
                .manifest_metadata
                .complete_header
                .encoded_bytes_upper_bound,
        )
        .map_err(invalid)?;
        profile
            .push(
                self.facts
                    .manifest_metadata
                    .complete_header
                    .encoded_bytes_upper_bound,
            )
            .map_err(invalid)?;
        let samples = self.schedule.sample_count();
        let ordinal = samples
            .checked_sub(1)
            .ok_or_else(|| FeaRunError::InvalidModel("thermal output requires samples".into()))?;
        for (index, entities) in [
            self.mesh.topology.nodes.len(),
            self.mesh.topology.volume_elements.len(),
            self.mesh.topology.volume_elements.len(),
            self.mesh.topology.volume_elements.len(),
        ]
        .into_iter()
        .enumerate()
        {
            let definition =
                crate::physics::thermal::history_fields::snapshot_definition(index, ordinal)
                    .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
            let full = entities / RECOVERED_FIELD_PARTITION_ENTITY_COUNT;
            let remainder = entities % RECOVERED_FIELD_PARTITION_ENTITY_COUNT;
            for (count, blocks) in [
                (RECOVERED_FIELD_PARTITION_ENTITY_COUNT, full),
                (remainder, usize::from(remainder != 0)),
            ] {
                if blocks == 0 {
                    continue;
                }
                let shape = RecoveredFieldPartition::profile_shape(
                    &definition,
                    RecoveredFieldSample::TimeSeconds(0.0),
                    count,
                )
                .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
                let repetitions = blocks.checked_mul(samples).ok_or_else(|| {
                    FeaRunError::InvalidModel("thermal output block count overflows".into())
                })?;
                for _ in 0..repetitions {
                    profile
                        .push(shape.encoded_bytes_upper_bound)
                        .map_err(invalid)?;
                }
            }
        }
        profile
            .finish(AnalysisNumericalObjectKind::SolverResult)
            .map_err(invalid)
    }

    pub fn execute(self) -> Result<FeaThermalRunResult, FeaRunError> {
        physical::run(
            self.model,
            self.backend,
            self.options,
            self.mesh,
            self.step,
            &self.schedule,
            &self.facts,
        )
    }
}
