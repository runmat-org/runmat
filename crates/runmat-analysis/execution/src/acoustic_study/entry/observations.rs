//! Retained native fields and authored observations of one frequency entry.
//! Fields are copied from the accepted solve; observations evaluate them at this frequency.
use super::*;
use crate::{
    import_acoustic_recovered_fields, prepare_acoustic_field_recovery,
    AcousticFieldRecoveryPreparation, ObservationPreparation, ObservationPublishedSource,
    ObservationWorkloadSource, PreparedAnalysisObjects,
};
use runmat_analysis_fea::{
    execution::{AnalysisNumericalObjectKind as Kind, ObservationHistorySourceContext},
    results::{
        field::RecoveredFieldSample,
        observation::{ObservationFieldCatalog, ObservationLimits},
    },
};

impl AcousticStudyEntryCoordinator {
    pub(super) fn input(&self, kind: Kind) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
        self.solve_task
            .as_ref()
            .ok_or_else(|| invalid("acoustic solve task absent"))?
            .inputs
            .iter()
            .find(|input| input.manifest.kind == kind)
            .cloned()
            .ok_or_else(|| invalid("acoustic original solve input absent"))
    }

    pub fn prepare_field_recovery(
        &mut self,
    ) -> AnalysisBridgeResult<&crate::PreparedAcousticFieldRecovery> {
        self.require(AcousticStudyPhase::ReadyForFieldRecovery)?;
        let task = prepare_acoustic_field_recovery(AcousticFieldRecoveryPreparation {
            source_host: self
                .solve_task
                .as_ref()
                .expect("completed solve")
                .host
                .clone(),
            workload: self.input(Kind::AcousticHarmonicWorkload)?,
            mesh: self.input(Kind::Mesh)?,
            source_result: self
                .result
                .as_ref()
                .expect("completed solve")
                .objects
                .clone(),
            program_revision: self.preparation.program_revision.clone(),
            output_policy: self.preparation.output_policy,
            local_execution: self.preparation.local_execution,
        })?;
        self.recovery_task = Some(task);
        self.phase = AcousticStudyPhase::AwaitingFieldRecovery;
        Ok(self.recovery_task.as_ref().expect("just prepared"))
    }

    pub fn complete_field_recovery(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(AcousticStudyPhase::AwaitingFieldRecovery)?;
        let task = self.recovery_task.as_ref().expect("awaited recovery");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let task_attempts = accounting.resource_graph(&task.resource_plan)?.attempts;
        let closure = AcousticStudyResultClosure::new(
            AcousticStudyResultPhase::FieldRecovery,
            &task.host,
            &response,
            &accounting,
        );
        let fields = import_acoustic_recovered_fields(
            source,
            response,
            task,
            self.preparation.output_policy.inventory_limits(),
        )?;
        let field_objects = crate::import_analysis_objects(
            source,
            &fields.output.root,
            Kind::Field,
            fields.output.logical_identity,
            fields.output.payload_digest,
            &task.host.artifact_access,
            self.preparation.output_policy.inventory_limits(),
        )?;
        let usage = self.extend_usage(&task.resource_plan, accounting)?;
        self.fields = Some(fields);
        self.field_objects = Some(field_objects);
        self.usage = Some(usage);
        self.closures.push(closure);
        self.attempts.extend(task_attempts);
        self.phase = if self.preparation.selected_step().observations.is_empty() {
            AcousticStudyPhase::ReadyForAccuracy
        } else {
            AcousticStudyPhase::ReadyForObservationFanout
        };
        Ok(())
    }

    /// Admit the authored observation plan over this entry's retained fields.
    pub fn prepare_observation_fanout(
        &mut self,
        source: &impl CacheImport,
    ) -> AnalysisBridgeResult<()> {
        self.require(AcousticStudyPhase::ReadyForObservationFanout)?;
        let recovery = self.recovery_task.as_ref().expect("completed recovery");
        let fields = self.fields.as_ref().expect("completed fields");
        let policy = self.preparation.output_policy;
        let workload = self.workload();
        let fanout = AcousticObservations::prepare_fanout_if_sampled(
            source,
            ObservationPreparation {
                source_host: recovery.host.clone(),
                source: ObservationPublishedSource::VolumeMeshRecovery {
                    recovery_request: recovery.request.clone(),
                    fields: fields.clone(),
                },
                source_workload: ObservationWorkloadSource::VolumeMesh(
                    self.input(Kind::AcousticHarmonicWorkload)?,
                ),
                mesh: Some(self.input(Kind::Mesh)?),
                accelerator_requests: Vec::new(),
                sequence: SequenceIdentity::from_solver_contract(&workload.model.sequence)?,
                step: self.preparation.selected_step().clone(),
                limits: ObservationLimits::default(),
                program_revision: self.preparation.program_revision.clone(),
                chunk_bytes: policy.chunk_bytes(),
                inventory_limits: policy.inventory_limits(),
                local_execution: self.preparation.local_execution,
            },
            RecoveredFieldSample::FrequencyHertz(workload.frequency.canonical_value()),
        )?;
        match fanout {
            Some(fanout) => {
                self.observation_state.fanout = Some(fanout);
                self.phase = AcousticStudyPhase::ReadyForObservationPartition { ordinal: 0 };
            }
            // No authored observation samples this frequency: the entry publishes its
            // fields and accuracy without an observation manifest.
            None => self.phase = AcousticStudyPhase::ReadyForAccuracy,
        }
        Ok(())
    }

    /// The solve task's producing step, shared by every entry of the study.
    pub(in crate::acoustic_study) fn producer(
        &self,
    ) -> AnalysisBridgeResult<runmat_analysis_fea::execution::StepIdentity> {
        Ok(self
            .solve_task
            .as_ref()
            .ok_or_else(|| invalid("acoustic solve task absent"))?
            .host
            .workload
            .step)
    }

    /// The canonical mesh objects the entry was solved on.
    pub(in crate::acoustic_study) fn mesh_objects(
        &self,
    ) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
        self.input(Kind::Mesh)
    }

    /// This complete entry's field catalog, plan and reduced observations for the study's
    /// entry-history phase.
    pub(in crate::acoustic_study) fn history_sample(
        &self,
    ) -> AnalysisBridgeResult<crate::sampled_study_observations::history::EntryHistorySample> {
        if !self.is_complete() {
            return Err(invalid("acoustic entry is not complete"));
        }
        let recovery = self
            .recovery_task
            .as_ref()
            .ok_or_else(|| invalid("acoustic recovery absent"))?;
        let fields = self
            .fields
            .as_ref()
            .ok_or_else(|| invalid("acoustic fields absent"))?;
        let state = &self.observation_state;
        if state.reduced.len() != state.reduction_hosts.len() {
            return Err(invalid("acoustic reductions lost their task hosts"));
        }
        Ok(
            crate::sampled_study_observations::history::EntryHistorySample {
                catalog: ObservationFieldCatalog::volume_mesh_recovery(
                    recovery.request.clone(),
                    fields.manifest.clone(),
                    SequenceIdentity::from_solver_contract(&self.workload().model.sequence)?,
                    self.preparation.selected_step().clone(),
                )?,
                plan: state.plan().cloned(),
                reduced: state
                    .reduced
                    .iter()
                    .cloned()
                    .zip(state.reduction_hosts.iter().cloned())
                    .collect(),
            },
        )
    }

    pub(super) fn observation_partition_count(&self) -> usize {
        self.observation_state
            .fanout
            .as_ref()
            .map_or(0, |fanout| fanout.partitions.len())
    }

    pub fn prepare_observation_partition(&mut self) -> AnalysisBridgeResult<()> {
        let ordinal = self.observation_state.contributions.len();
        self.require(AcousticStudyPhase::ReadyForObservationPartition { ordinal })?;
        self.observation_state.partition(ordinal)?;
        self.phase = AcousticStudyPhase::AwaitingObservationPartition { ordinal };
        Ok(())
    }

    pub fn complete_observation_partition(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        let ordinal = self.observation_state.contributions.len();
        self.require(AcousticStudyPhase::AwaitingObservationPartition { ordinal })?;
        let task = self.observation_state.partition(ordinal)?;
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let task_attempts = accounting.resource_graph(&task.resource_plan)?.attempts;
        let closure = AcousticStudyResultClosure::new(
            AcousticStudyResultPhase::ObservationPartition { ordinal },
            &task.host,
            &response,
            &accounting,
        );
        let plan = task.resource_plan.clone();
        let contribution = self.observation_state.import_partition(
            source,
            ordinal,
            response,
            self.preparation.output_policy,
        )?;
        let usage = self.extend_usage(&plan, accounting)?;
        self.observation_state
            .contributions
            .insert(ordinal, contribution);
        self.usage = Some(usage);
        self.closures.push(closure);
        self.attempts.extend(task_attempts);
        self.phase = if ordinal + 1 < self.observation_partition_count() {
            AcousticStudyPhase::ReadyForObservationPartition {
                ordinal: ordinal + 1,
            }
        } else {
            AcousticStudyPhase::ReadyForObservationReduction { ordinal: 0 }
        };
        Ok(())
    }

    pub fn prepare_observation_reduction(&mut self) -> AnalysisBridgeResult<()> {
        let ordinal = self.observation_state.reduced.len();
        self.require(AcousticStudyPhase::ReadyForObservationReduction { ordinal })?;
        let task = self.observation_state.prepare_reduction(
            &self.preparation.program_revision,
            self.preparation.output_policy,
        )?;
        self.observation_state.reduction = Some(task);
        self.phase = AcousticStudyPhase::AwaitingObservationReduction { ordinal };
        Ok(())
    }

    pub fn complete_observation_reduction(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        let ordinal = self.observation_state.reduced.len();
        self.require(AcousticStudyPhase::AwaitingObservationReduction { ordinal })?;
        let task = self
            .observation_state
            .reduction
            .as_ref()
            .expect("awaited reduction");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let task_attempts = accounting.resource_graph(&task.resource_plan)?.attempts;
        let closure = AcousticStudyResultClosure::new(
            AcousticStudyResultPhase::ObservationReduction { ordinal },
            &task.host,
            &response,
            &accounting,
        );
        let plan = task.resource_plan.clone();
        let reduced = self.observation_state.import_reduction(
            source,
            response,
            self.preparation.output_policy,
        )?;
        let usage = self.extend_usage(&plan, accounting)?;
        self.observation_state.push_reduced(reduced)?;
        self.usage = Some(usage);
        self.closures.push(closure);
        self.attempts.extend(task_attempts);
        let count = self.observation_state.reduction_count();
        self.phase = if self.observation_state.reduced.len() == count {
            AcousticStudyPhase::ReadyForObservationAssembly
        } else {
            AcousticStudyPhase::ReadyForObservationReduction {
                ordinal: ordinal + 1,
            }
        };
        Ok(())
    }

    pub fn prepare_observation_assembly(&mut self) -> AnalysisBridgeResult<()> {
        self.require(AcousticStudyPhase::ReadyForObservationAssembly)?;
        let task = self.observation_state.prepare_assembly(
            &self.preparation.program_revision,
            self.preparation.output_policy,
        )?;
        self.observation_state.assembly = Some(task);
        self.phase = AcousticStudyPhase::AwaitingObservationAssembly;
        Ok(())
    }

    pub fn complete_observation_assembly(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(AcousticStudyPhase::AwaitingObservationAssembly)?;
        let task = self
            .observation_state
            .assembly
            .as_ref()
            .expect("awaited assembly");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let task_attempts = accounting.resource_graph(&task.resource_plan)?.attempts;
        let closure = AcousticStudyResultClosure::new(
            AcousticStudyResultPhase::ObservationAssembly,
            &task.host,
            &response,
            &accounting,
        );
        let plan = task.resource_plan.clone();
        let artifact = self.observation_state.import_assembly(
            source,
            response,
            self.preparation.output_policy,
        )?;
        let fields = self.fields.as_ref().expect("completed fields");
        let recovery = self.recovery_task.as_ref().expect("completed recovery");
        let catalog = self.observation_state.publication_catalog(
            &artifact,
            self.solve_task
                .as_ref()
                .expect("completed solve")
                .host
                .workload
                .step,
            ObservationFieldCatalog::volume_mesh_recovery(
                recovery.request.clone(),
                fields.manifest.clone(),
                SequenceIdentity::from_solver_contract(&self.workload().model.sequence)?,
                self.preparation.selected_step().clone(),
            )?,
            self.preparation.selected_step(),
            ObservationHistorySourceContext::VolumeMesh {
                mesh: &self.preparation.mesh,
            },
            runmat_analysis_fea::results::observation::ObservationPublicationScope::Entry,
        )?;
        let observation_objects = crate::import_analysis_objects(
            source,
            &artifact.output.root,
            artifact.output.kind,
            artifact.output.logical_identity,
            artifact.output.payload_digest,
            &artifact.access,
            self.preparation.output_policy.inventory_limits(),
        )?;
        let usage = self.extend_usage(&plan, accounting)?;
        self.observation_objects = Some(observation_objects);
        self.observation_state.completed = Some(artifact);
        self.observation_state.catalog = Some(catalog);
        self.usage = Some(usage);
        self.closures.push(closure);
        self.attempts.extend(task_attempts);
        self.phase = AcousticStudyPhase::ReadyForAccuracy;
        Ok(())
    }
}
