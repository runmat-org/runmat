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
        self.transport_profile_of_samples(chunk_bytes, self.schedule.sample_count())
    }

    /// The same bound for the header and `retained` samples of every recovered field, each
    /// charged as the last (widest-named) sample. With one sample it bounds every product that
    /// copies one sample's fields plus header inventories (node and element IDs, times).
    pub fn transport_profile_of_samples(
        &self,
        chunk_bytes: usize,
        retained: usize,
    ) -> Result<AnalysisTransportProfile, FeaRunError> {
        self.result_profile(chunk_bytes, retained)
            .map(|(profile, _)| profile)
    }

    /// Typed storage of the complete decoded result: the header at the codec's per-byte decode
    /// bound, each field partition priced from its shape.
    pub fn result_typed_resident_bytes(&self, chunk_bytes: usize) -> Result<u64, FeaRunError> {
        self.result_profile(chunk_bytes, self.schedule.sample_count())
            .map(|(_, typed)| typed)
    }

    /// The workspace an original-object import charges for the complete result, as the shared
    /// import profiler prices it: the transport held three times (original, assembled payload,
    /// verification copy), the fixed per-reference codec allowance for every chunk and the
    /// manifest, the manifest's typed decode, and every partition's typed decode (its typed
    /// storage, `result_typed_resident_bytes`, with its encoded bytes alongside).
    pub fn result_import_workspace_bytes(&self, chunk_bytes: usize) -> Result<u64, FeaRunError> {
        let (profile, partitions) =
            self.result_profile(chunk_bytes, self.schedule.sample_count())?;
        let overflow = || FeaRunError::InvalidModel("thermal import workspace overflows".into());
        let references = profile
            .chunk_count_upper_bound
            .checked_add(1)
            .ok_or_else(overflow)?;
        let fixed =
            runmat_canonical_codec::projected_canonical_materialization_resident_bytes_estimate(
                0, references,
            )
            .ok_or_else(overflow)?;
        let manifest =
            runmat_canonical_codec::projected_canonical_typed_decode_resident_bytes_estimate(
                profile.root.encoded_bytes_upper_bound,
                1,
            )
            .ok_or_else(overflow)?;
        profile
            .total_bytes_upper_bound
            .checked_mul(3)
            .and_then(|bytes| bytes.checked_add(fixed))
            .and_then(|bytes| bytes.checked_add(manifest))
            .and_then(|bytes| bytes.checked_add(partitions))
            .and_then(|bytes| bytes.checked_add(profile.payload_bytes_upper_bound))
            .ok_or_else(overflow)
    }

    /// The transport profile of the header and `retained` samples, with the summed typed storage
    /// of their decoded partitions.
    fn result_profile(
        &self,
        chunk_bytes: usize,
        retained: usize,
    ) -> Result<(AnalysisTransportProfile, u64), FeaRunError> {
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
        let header = self
            .facts
            .manifest_metadata
            .complete_header
            .encoded_bytes_upper_bound;
        profile.push(header).map_err(invalid)?;
        let overflow =
            || FeaRunError::InvalidModel("thermal typed decode estimate overflows".into());
        // The header's integer inventories may encode in one byte per value: charge it at the
        // codec's per-byte typed decode bound, without the encoded bytes counted separately.
        let mut typed =
            runmat_canonical_codec::projected_canonical_typed_decode_resident_bytes_estimate(
                header, 1,
            )
            .ok_or_else(overflow)?;
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
                let decoded = RecoveredFieldPartition::profile_shape_typed_resident_bytes(
                    &definition,
                    RecoveredFieldSample::TimeSeconds(0.0),
                    count,
                )
                .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
                let repetitions = blocks.checked_mul(retained.min(samples)).ok_or_else(|| {
                    FeaRunError::InvalidModel("thermal output block count overflows".into())
                })?;
                typed = u64::try_from(repetitions)
                    .ok()
                    .and_then(|n| decoded.checked_mul(n))
                    .and_then(|bytes| typed.checked_add(bytes))
                    .ok_or_else(overflow)?;
                for _ in 0..repetitions {
                    profile
                        .push(shape.encoded_bytes_upper_bound)
                        .map_err(invalid)?;
                }
            }
        }
        let profile = profile
            .finish(AnalysisNumericalObjectKind::SolverResult)
            .map_err(invalid)?;
        Ok((profile, typed))
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
    /// Transport bound of the result header and one sample of every recovered field: the most
    /// a single-sample product of this solve (a state, a field observation) copies from it.
    pub fn sample_transport_profile(
        &self,
        chunk_bytes: usize,
    ) -> Result<AnalysisTransportProfile, FeaRunError> {
        self.0.transport_profile_of_samples(chunk_bytes, 1)
    }
    /// Typed storage of the complete decoded result.
    pub fn result_typed_resident_bytes(&self, chunk_bytes: usize) -> Result<u64, FeaRunError> {
        self.0.result_typed_resident_bytes(chunk_bytes)
    }
    /// The workspace an original-object import charges for the complete result.
    pub fn result_import_workspace_bytes(&self, chunk_bytes: usize) -> Result<u64, FeaRunError> {
        self.0.result_import_workspace_bytes(chunk_bytes)
    }
    pub fn execute_primary_fields(self) -> Result<FeaThermalRunResult, FeaRunError> {
        self.0.execute_primary_fields()
    }
}
