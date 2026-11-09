//! Authored observations of a statistical step (random-response RMS, response-spectrum combined
//! peak) over its recovered fields, through the shared sampled observation engine and the
//! family-neutral volume-mesh authority. Both statistical coordinators own one of these after
//! their assembly completes and before accuracy.
use crate::sampled_study_observations::SampledStudyObservations;
use crate::{
    import_statistical_recovered_fields, prepare_statistical_field_recovery, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisHostWorkload, AnalysisOutputPolicy, ImportedRecoveredFields,
    ObservationPreparation, ObservationPublishedSource, ObservationWorkloadSource,
    PreparedAnalysisObjects, PreparedObservationAssembly, PreparedObservationFanout,
    PreparedObservationPartition, PreparedObservationReduction, PreparedStatisticalFieldRecovery,
    StatisticalFieldRecoveryPreparation,
};
use runmat_analysis_core::ProcedureStep;
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, ObservationHistorySourceContext, SequenceIdentity,
};
use runmat_analysis_fea::results::observation::{
    ObservationFieldCatalog, ObservationLimits, ObservationPublicationCatalog,
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ProgramExecutionResponse,
};
use runmat_meshing_core::SolverMeshArtifact;

/// The completed statistical assembly whose fields authored outputs observe.
pub struct StatisticalObservationSource {
    pub assembly_host: AnalysisHostWorkload,
    /// The assembly's workload and canonical mesh inputs.
    pub assembly_inputs: Vec<PreparedAnalysisObjects>,
    pub source_result: PreparedAnalysisObjects,
    /// The response-spectrum assembly's field recovery; absent for random response.
    pub source_fields: Option<PreparedAnalysisObjects>,
    pub sequence: SequenceIdentity,
    pub step: ProcedureStep,
    pub mesh: SolverMeshArtifact,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Phase {
    ReadyForFieldRecovery,
    AwaitingFieldRecovery(Digest),
    ReadyForFanout,
    AwaitingPartitions,
    ReadyForReduction { ordinal: usize },
    AwaitingReduction { ordinal: usize },
    ReadyForAssembly,
    AwaitingAssembly,
    Complete,
}

pub struct StatisticalStudyObservations {
    source: StatisticalObservationSource,
    recovery: Option<PreparedStatisticalFieldRecovery>,
    fields: Option<ImportedRecoveredFields>,
    state: SampledStudyObservations,
    phase: Phase,
}

impl StatisticalStudyObservations {
    pub fn new(source: StatisticalObservationSource) -> AnalysisBridgeResult<Self> {
        if source.step.observations.is_empty() {
            return Err(invalid("statistical step has no authored observations"));
        }
        Ok(Self {
            source,
            recovery: None,
            fields: None,
            state: SampledStudyObservations::default(),
            phase: Phase::ReadyForFieldRecovery,
        })
    }

    fn output_policy(&self) -> AnalysisBridgeResult<AnalysisOutputPolicy> {
        AnalysisOutputPolicy::new(self.source.chunk_bytes, self.source.inventory_limits)
    }

    fn assembly_input(
        &self,
        kind: AnalysisNumericalObjectKind,
    ) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
        let mut matching = self
            .source
            .assembly_inputs
            .iter()
            .filter(|input| input.manifest.kind == kind);
        match (matching.next(), matching.next()) {
            (Some(input), None) => Ok(input.clone()),
            _ => Err(invalid("statistical assembly source input is not unique")),
        }
    }

    fn workload_input(&self) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
        let kind = if self.source.source_fields.is_some() {
            AnalysisNumericalObjectKind::StructuralModalResponseSpectrumWorkload
        } else {
            AnalysisNumericalObjectKind::StructuralRandomResponseWorkload
        };
        self.assembly_input(kind)
    }

    pub fn is_complete(&self) -> bool {
        self.phase == Phase::Complete
    }

    /// The publication catalog accuracy and result publication bind; present once complete.
    pub fn catalog(&self) -> Option<ObservationPublicationCatalog> {
        self.state.catalog.clone()
    }

    pub fn prepare_field_recovery(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedStatisticalFieldRecovery> {
        if self.phase != Phase::ReadyForFieldRecovery {
            return Err(invalid("statistical study is not ready for field recovery"));
        }
        let task = prepare_statistical_field_recovery(StatisticalFieldRecoveryPreparation {
            source_host: self.source.assembly_host.clone(),
            workload: self.workload_input()?,
            mesh: self.assembly_input(AnalysisNumericalObjectKind::Mesh)?,
            source_result: self.source.source_result.clone(),
            source_fields: self.source.source_fields.clone(),
            program_revision: self.source.program_revision.clone(),
            output_policy: self.output_policy()?,
            local_execution: self.source.local_execution,
        })?;
        self.phase = Phase::AwaitingFieldRecovery(task.host.workload.identity.digest());
        Ok(self.recovery.insert(task))
    }

    pub fn complete_field_recovery(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
    ) -> AnalysisBridgeResult<()> {
        let task = self
            .recovery
            .as_ref()
            .ok_or_else(|| invalid("statistical field recovery is absent"))?;
        if self.phase != Phase::AwaitingFieldRecovery(task.host.workload.identity.digest()) {
            return Err(AnalysisBridgeError::Identity(
                "statistical field recovery differs from the awaited task",
            ));
        }
        self.fields = Some(import_statistical_recovered_fields(
            source,
            response,
            task,
            self.source.inventory_limits,
        )?);
        self.phase = Phase::ReadyForFanout;
        Ok(())
    }

    pub fn prepare_observation_fanout(
        &mut self,
        source: &impl CacheImport,
    ) -> AnalysisBridgeResult<&PreparedObservationFanout> {
        if self.phase != Phase::ReadyForFanout {
            return Err(invalid("statistical study is not ready for observations"));
        }
        let recovery = self
            .recovery
            .as_ref()
            .ok_or_else(|| invalid("statistical field recovery is absent"))?;
        let fields = self
            .fields
            .as_ref()
            .ok_or_else(|| invalid("statistical recovered fields are absent"))?;
        let task = SampledStudyObservations::prepare_fanout(
            source,
            ObservationPreparation {
                source_host: recovery.host.clone(),
                source: ObservationPublishedSource::VolumeMeshRecovery {
                    recovery_request: recovery.request.clone(),
                    fields: fields.clone(),
                },
                source_workload: ObservationWorkloadSource::VolumeMesh(self.workload_input()?),
                mesh: Some(self.assembly_input(AnalysisNumericalObjectKind::Mesh)?),
                accelerator_requests: Vec::new(),
                sequence: self.source.sequence,
                step: self.source.step.clone(),
                limits: ObservationLimits::default(),
                program_revision: self.source.program_revision.clone(),
                chunk_bytes: self.source.chunk_bytes,
                inventory_limits: self.source.inventory_limits,
                local_execution: self.source.local_execution,
            },
        )?;
        self.phase = Phase::AwaitingPartitions;
        Ok(self.state.fanout.insert(task))
    }

    pub fn observation_partition(
        &self,
        ordinal: usize,
    ) -> AnalysisBridgeResult<&PreparedObservationPartition> {
        if self.phase != Phase::AwaitingPartitions {
            return Err(invalid(
                "statistical observation partitions are not awaited",
            ));
        }
        self.state.partition(ordinal)
    }

    /// Completion order is arbitrary; each immutable partition ordinal completes exactly once.
    pub fn complete_observation_partition(
        &mut self,
        source: &impl CacheImport,
        ordinal: usize,
        response: ProgramExecutionResponse,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != Phase::AwaitingPartitions {
            return Err(invalid(
                "statistical observation partitions are not awaited",
            ));
        }
        let contribution =
            self.state
                .import_partition(source, ordinal, response, self.output_policy()?)?;
        self.state.contributions.insert(ordinal, contribution);
        let count = self
            .state
            .fanout
            .as_ref()
            .expect("prepared fanout")
            .partitions
            .len();
        if self.state.contributions.len() == count {
            self.phase = Phase::ReadyForReduction { ordinal: 0 };
        }
        Ok(())
    }

    pub fn prepare_observation_reduction(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedObservationReduction> {
        let ordinal = self.state.reduced.len();
        if self.phase != (Phase::ReadyForReduction { ordinal }) {
            return Err(invalid("statistical observation reduction is not ready"));
        }
        let task = self
            .state
            .prepare_reduction(&self.source.program_revision, self.output_policy()?)?;
        self.phase = Phase::AwaitingReduction { ordinal };
        Ok(self.state.reduction.insert(task))
    }

    pub fn complete_observation_reduction(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
    ) -> AnalysisBridgeResult<()> {
        let ordinal = self.state.reduced.len();
        if self.phase != (Phase::AwaitingReduction { ordinal }) {
            return Err(invalid("statistical observation reduction is not awaited"));
        }
        let reduced = self
            .state
            .import_reduction(source, response, self.output_policy()?)?;
        self.state.push_reduced(reduced)?;
        let count = self.state.reduction_count();
        self.phase = if self.state.reduced.len() == count {
            Phase::ReadyForAssembly
        } else {
            Phase::ReadyForReduction {
                ordinal: ordinal + 1,
            }
        };
        Ok(())
    }

    pub fn prepare_observation_assembly(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedObservationAssembly> {
        if self.phase != Phase::ReadyForAssembly {
            return Err(invalid("statistical observation assembly is not ready"));
        }
        let task = self
            .state
            .prepare_assembly(&self.source.program_revision, self.output_policy()?)?;
        self.phase = Phase::AwaitingAssembly;
        Ok(self.state.assembly.insert(task))
    }

    pub fn complete_observation_assembly(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != Phase::AwaitingAssembly {
            return Err(invalid("statistical observation assembly is not awaited"));
        }
        let artifact = self
            .state
            .import_assembly(source, response, self.output_policy()?)?;
        let recovery = self
            .recovery
            .as_ref()
            .ok_or_else(|| invalid("statistical field recovery is absent"))?;
        let fields = self
            .fields
            .as_ref()
            .ok_or_else(|| invalid("statistical recovered fields are absent"))?;
        let catalog = self.state.publication_catalog(
            &artifact,
            recovery.host.workload.step,
            ObservationFieldCatalog::volume_mesh_recovery(
                recovery.request.clone(),
                fields.manifest.clone(),
                self.source.sequence,
                self.source.step.clone(),
            )?,
            &self.source.step,
            ObservationHistorySourceContext::VolumeMesh {
                mesh: &self.source.mesh,
            },
            runmat_analysis_fea::results::observation::ObservationPublicationScope::Complete,
        )?;
        self.state.completed = Some(artifact);
        self.state.catalog = Some(catalog);
        self.phase = Phase::Complete;
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
