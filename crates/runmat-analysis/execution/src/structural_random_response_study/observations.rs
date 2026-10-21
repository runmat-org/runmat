//! Authored observations of a random-response step over its recovered RMS fields, through the
//! shared sampled observation engine and the family-neutral volume-mesh authority.
use super::*;
use crate::{
    import_random_response_recovered_fields, prepare_random_response_field_recovery,
    AnalysisOutputPolicy, ObservationPreparation, ObservationPublishedSource,
    ObservationWorkloadSource, PreparedObservationAssembly, PreparedObservationFanout,
    PreparedObservationPartition, PreparedObservationReduction,
    PreparedRandomResponseFieldRecovery, RandomResponseFieldRecoveryPreparation,
};
use runmat_analysis_fea::execution::ObservationHistorySourceContext;
use runmat_analysis_fea::results::observation::{ObservationFieldCatalog, ObservationLimits};

impl StructuralRandomResponseStudyCoordinator {
    fn output_policy(&self) -> AnalysisBridgeResult<AnalysisOutputPolicy> {
        AnalysisOutputPolicy::new(self.preparation.chunk_bytes, self.preparation.inventory_limits)
    }

    fn assembly_input(
        &self,
        kind: AnalysisNumericalObjectKind,
    ) -> AnalysisBridgeResult<crate::PreparedAnalysisObjects> {
        let mut matching = self
            .assembly_source_context_inputs
            .iter()
            .filter(|input| input.manifest.kind == kind);
        match (matching.next(), matching.next()) {
            (Some(input), None) => Ok(input.clone()),
            _ => Err(invalid("random-response assembly source input is not unique")),
        }
    }

    /// True once the assembled result must publish authored observations before accuracy.
    pub fn requires_observations(&self) -> bool {
        self.phase == StudyPhase::ReadyForFieldRecovery
    }

    pub fn prepare_field_recovery(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedRandomResponseFieldRecovery> {
        if self.phase != StudyPhase::ReadyForFieldRecovery {
            return Err(invalid("random-response study is not ready for field recovery"));
        }
        let task = prepare_random_response_field_recovery(RandomResponseFieldRecoveryPreparation {
            source_host: self
                .assembly_host
                .as_ref()
                .ok_or_else(|| invalid("random-response assembly host is absent"))?
                .clone(),
            workload: self
                .assembly_input(AnalysisNumericalObjectKind::StructuralRandomResponseWorkload)?,
            mesh: self.assembly_input(AnalysisNumericalObjectKind::Mesh)?,
            source_result: self
                .result
                .as_ref()
                .ok_or_else(|| invalid("completed random-response result is absent"))?
                .result_objects
                .clone(),
            program_revision: self.preparation.program_revision.clone(),
            output_policy: self.output_policy()?,
        })?;
        self.phase = StudyPhase::AwaitingFieldRecovery(task.host.workload.identity.digest());
        self.field_recovery = Some(task);
        Ok(self.field_recovery.as_ref().expect("just prepared"))
    }

    pub fn complete_field_recovery(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
    ) -> AnalysisBridgeResult<()> {
        let task = self
            .field_recovery
            .as_ref()
            .ok_or_else(|| invalid("random-response field recovery is absent"))?;
        if self.phase != StudyPhase::AwaitingFieldRecovery(task.host.workload.identity.digest()) {
            return Err(AnalysisBridgeError::Identity(
                "random-response field recovery differs from the awaited task",
            ));
        }
        self.fields = Some(import_random_response_recovered_fields(
            source,
            response,
            task,
            self.preparation.inventory_limits,
        )?);
        self.phase = StudyPhase::ReadyForObservationFanout;
        Ok(())
    }

    pub fn prepare_observation_fanout(
        &mut self,
        source: &impl CacheImport,
    ) -> AnalysisBridgeResult<&PreparedObservationFanout> {
        if self.phase != StudyPhase::ReadyForObservationFanout {
            return Err(invalid("random-response study is not ready for observations"));
        }
        let recovery = self
            .field_recovery
            .as_ref()
            .ok_or_else(|| invalid("random-response field recovery is absent"))?;
        let fields = self
            .fields
            .as_ref()
            .ok_or_else(|| invalid("random-response recovered fields are absent"))?;
        let workload = self.bound_workload()?;
        let task = crate::sampled_study_observations::SampledStudyObservations::prepare_fanout(
            source,
            ObservationPreparation {
                source_host: recovery.host.clone(),
                source: ObservationPublishedSource::VolumeMeshRecovery {
                    recovery_request: recovery.request.clone(),
                    fields: fields.clone(),
                },
                source_workload: ObservationWorkloadSource::VolumeMesh(
                    self.assembly_input(
                        AnalysisNumericalObjectKind::StructuralRandomResponseWorkload,
                    )?,
                ),
                mesh: Some(self.assembly_input(AnalysisNumericalObjectKind::Mesh)?),
                accelerator_requests: Vec::new(),
                sequence: SequenceIdentity::from_solver_contract(&workload.model().sequence)?,
                step: workload.procedure_step()?.clone(),
                limits: ObservationLimits::default(),
                program_revision: self.preparation.program_revision.clone(),
                chunk_bytes: self.preparation.chunk_bytes,
                inventory_limits: self.preparation.inventory_limits,
            },
        )?;
        self.observation_state.fanout = Some(task);
        self.phase = StudyPhase::AwaitingObservationPartitions;
        Ok(self
            .observation_state
            .fanout
            .as_ref()
            .expect("just prepared"))
    }

    pub fn observation_partition(
        &self,
        ordinal: usize,
    ) -> AnalysisBridgeResult<&PreparedObservationPartition> {
        if self.phase != StudyPhase::AwaitingObservationPartitions {
            return Err(invalid("random-response observation partitions are not awaited"));
        }
        self.observation_state.partition(ordinal)
    }

    /// Completion order is arbitrary; each immutable partition ordinal completes exactly once.
    pub fn complete_observation_partition(
        &mut self,
        source: &impl CacheImport,
        ordinal: usize,
        response: ProgramExecutionResponse,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != StudyPhase::AwaitingObservationPartitions {
            return Err(invalid("random-response observation partitions are not awaited"));
        }
        let contribution = self.observation_state.import_partition(
            source,
            ordinal,
            response,
            self.output_policy()?,
        )?;
        self.observation_state
            .contributions
            .insert(ordinal, contribution);
        let count = self
            .observation_state
            .fanout
            .as_ref()
            .expect("prepared fanout")
            .partitions
            .len();
        if self.observation_state.contributions.len() == count {
            self.phase = StudyPhase::ReadyForObservationReduction { ordinal: 0 };
        }
        Ok(())
    }

    pub fn prepare_observation_reduction(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedObservationReduction> {
        let ordinal = self.observation_state.reduced.len();
        if self.phase != (StudyPhase::ReadyForObservationReduction { ordinal }) {
            return Err(invalid("random-response observation reduction is not ready"));
        }
        let task = self
            .observation_state
            .prepare_reduction(&self.preparation.program_revision, self.output_policy()?)?;
        self.observation_state.reduction = Some(task);
        self.phase = StudyPhase::AwaitingObservationReduction { ordinal };
        Ok(self
            .observation_state
            .reduction
            .as_ref()
            .expect("just prepared"))
    }

    pub fn complete_observation_reduction(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
    ) -> AnalysisBridgeResult<()> {
        let ordinal = self.observation_state.reduced.len();
        if self.phase != (StudyPhase::AwaitingObservationReduction { ordinal }) {
            return Err(invalid("random-response observation reduction is not awaited"));
        }
        let reduced =
            self.observation_state
                .import_reduction(source, response, self.output_policy()?)?;
        self.observation_state.reduced.push(reduced);
        let count = self
            .observation_state
            .fanout
            .as_ref()
            .expect("fanout")
            .source_request
            .plan
            .observations
            .len();
        self.phase = if self.observation_state.reduced.len() == count {
            StudyPhase::ReadyForObservationAssembly
        } else {
            StudyPhase::ReadyForObservationReduction {
                ordinal: ordinal + 1,
            }
        };
        Ok(())
    }

    pub fn prepare_observation_assembly(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedObservationAssembly> {
        if self.phase != StudyPhase::ReadyForObservationAssembly {
            return Err(invalid("random-response observation assembly is not ready"));
        }
        let task = self
            .observation_state
            .prepare_assembly(&self.preparation.program_revision, self.output_policy()?)?;
        self.observation_state.assembly = Some(task);
        self.phase = StudyPhase::AwaitingObservationAssembly;
        Ok(self
            .observation_state
            .assembly
            .as_ref()
            .expect("just prepared"))
    }

    pub fn complete_observation_assembly(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != StudyPhase::AwaitingObservationAssembly {
            return Err(invalid("random-response observation assembly is not awaited"));
        }
        let artifact =
            self.observation_state
                .import_assembly(source, response, self.output_policy()?)?;
        let recovery = self
            .field_recovery
            .as_ref()
            .ok_or_else(|| invalid("random-response field recovery is absent"))?;
        let fields = self
            .fields
            .as_ref()
            .ok_or_else(|| invalid("random-response recovered fields are absent"))?;
        let workload = self.bound_workload()?;
        let sequence = SequenceIdentity::from_solver_contract(&workload.model().sequence)?;
        let step = workload.procedure_step()?.clone();
        let catalog = self.observation_state.publication_catalog(
            &artifact,
            recovery.host.workload.step,
            ObservationFieldCatalog::volume_mesh_recovery(
                recovery.request.clone(),
                fields.manifest.clone(),
                sequence,
                step.clone(),
            )?,
            &step,
            ObservationHistorySourceContext::VolumeMesh {
                mesh: &self.preparation.mesh,
            },
        )?;
        self.observation_state.completed = Some(artifact);
        self.observation_state.catalog = Some(catalog);
        self.phase = StudyPhase::ReadyForAccuracy;
        Ok(())
    }
}
