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
    /// The backend the solve executes with (the admitted placement).
    pub(super) backend: ComputeBackend,
    pub(super) accelerator: (
        crate::contracts::accelerator_admission::AcceleratorFamily,
        crate::contracts::accelerator_admission::AcceleratorAdmission,
    ),
    pub(super) options: ThermalSolveOptions,
    pub(super) mesh: &'a SolverMeshArtifact,
    pub(super) step: &'a ProcedureStep,
    pub(super) schedule: ThermalSolveSchedule,
    pub(super) facts: ThermalPreassemblyResourceFacts,
    pub(super) maximum_resident_bytes: u64,
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
        let maximum = self.maximum_resident_bytes;
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
        for (index, entities) in crate::physics::thermal::history_fields::snapshot_entities(
            self.mesh.topology.nodes.len(),
            self.mesh.topology.volume_elements.len(),
            self.facts.history.interface_nodes,
        ) {
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
        let step = self.step;
        let mesh = self.mesh;
        let mut result = self.execute_primary_fields()?;
        super::inline_history::evaluate(&mut result, step, mesh)?;
        Ok(result)
    }

    /// Execute numerical fields only. Authored observations still govern sample scheduling and
    /// admission; their evaluation belongs to the shared published-field task chain.
    /// The authored model and producer identity are never rewritten to defer observations.
    pub fn execute_primary_fields(self) -> Result<FeaThermalRunResult, FeaRunError> {
        let (family, admission) = self.accelerator;
        let mut result = physical::run(
            self.model,
            self.backend,
            self.options,
            self.mesh,
            self.step,
            &self.schedule,
            &self.facts,
        )?;
        crate::contracts::accelerator_admission::complete_accelerator_admission(
            family,
            admission,
            &mut result.run,
        )?;
        Ok(result)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ThermalObservationExecution {
    Inline,
    Deferred,
}

#[derive(Clone, Copy)]
pub(crate) struct ThermalPreparationAdmission {
    pub observations: ThermalObservationExecution,
    pub maximum_resident_bytes: u64,
}
/// Numerical fields with observations deferred to their separately admitted published-field tasks.
/// This opaque carrier cannot execute an inline history under a fields-only reservation.
pub struct PreparedThermalPrimarySolve<'a>(PreparedThermalSolve<'a>);
impl<'a> PreparedThermalPrimarySolve<'a> {
    pub(super) fn new(prepared: PreparedThermalSolve<'a>) -> Self {
        Self(prepared)
    }
    /// Effective placement ceiling, bounded by the original authored resources.
    pub fn maximum_resident_bytes(&self) -> u64 {
        self.0.maximum_resident_bytes
    }
    pub fn numerical_reservation_bytes(&self) -> Result<ByteCount, ResourcePlanningError> {
        self.0.numerical_reservation_bytes()
    }
    pub fn require_output_reservation(
        &self,
        output: ByteCount,
    ) -> Result<ByteCount, ResourcePlanningError> {
        self.0.require_output_reservation(output)
    }
    pub fn transport_profile(
        &self,
        chunk_bytes: usize,
    ) -> Result<AnalysisTransportProfile, FeaRunError> {
        self.0.transport_profile(chunk_bytes)
    }
    pub fn execute_primary_fields(self) -> Result<FeaThermalRunResult, FeaRunError> {
        self.0.execute_primary_fields()
    }
}
