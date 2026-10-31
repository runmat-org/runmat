//! Acoustic transient retained samples through the shared volume-mesh history workflow.
use super::*;
use crate::{
    import_acoustic_response_recovered_fields, prepare_acoustic_response_field_recovery,
    sampled_history_observations::{SampledHistorySource, SampledObservationTask},
    AcousticResponseFieldRecoveryPreparation, ObservationHistoryFinalizationSource,
    ObservationPreparation, ObservationPublishedSource, ObservationWorkloadSource,
    PreparedAnalysisObjects,
};
use runmat_analysis_core::ProcedureStep;
use runmat_analysis_fea::{
    execution::{AnalysisNumericalObjectKind as Kind, ObservationHistorySourceContext},
    results::{
        field::RecoveredFieldSample,
        observation::{ObservationFieldCatalog, ObservationLimits},
    },
};

struct Context<'a> {
    study: &'a AcousticResponseStudyCoordinator,
    recovery: &'a crate::PreparedAcousticResponseFieldRecovery,
    fields: &'a crate::ImportedRecoveredFields,
}

impl SampledHistorySource for Context<'_> {
    fn step(&self) -> &ProcedureStep {
        self.study.selected_step()
    }
    fn solve_host(&self) -> &AnalysisHostWorkload {
        &self.recovery.host
    }
    fn program_revision(&self) -> &ProgramRevision {
        &self.study.preparation.program_revision
    }
    fn output_policy(&self) -> AnalysisOutputPolicy {
        self.study.preparation.output_policy
    }
    fn sample_count(&self) -> usize {
        self.study
            .result
            .as_ref()
            .expect("completed solve")
            .result
            .samples()
            .len()
    }
    fn observation_preparation(
        &self,
        _source: &impl CacheImport,
        ordinal: usize,
    ) -> AnalysisBridgeResult<(ObservationPreparation, RecoveredFieldSample)> {
        let sample = *self
            .study
            .result
            .as_ref()
            .expect("completed solve")
            .result
            .samples()
            .get(ordinal)
            .ok_or_else(|| {
                invalid("acoustic response observation sample outside retained history")
            })?;
        let policy = self.output_policy();
        Ok((
            ObservationPreparation {
                source_host: self.recovery.host.clone(),
                source: ObservationPublishedSource::VolumeMeshRecovery {
                    recovery_request: self.recovery.request.clone(),
                    fields: self.fields.clone(),
                },
                source_workload: ObservationWorkloadSource::VolumeMesh(
                    self.study.input(Kind::AcousticResponseWorkload)?,
                ),
                mesh: Some(self.study.input(Kind::Mesh)?),
                accelerator_requests: Vec::new(),
                sequence: SequenceIdentity::from_solver_contract(
                    &self.study.preparation.workload().model.sequence,
                )?,
                step: self.step().clone(),
                limits: ObservationLimits::default(),
                program_revision: self.program_revision().clone(),
                chunk_bytes: policy.chunk_bytes(),
                inventory_limits: policy.inventory_limits(),
                local_execution: self.study.preparation.local_execution,
            },
            sample,
        ))
    }
    fn field_catalog(&self) -> AnalysisBridgeResult<ObservationFieldCatalog> {
        Ok(ObservationFieldCatalog::volume_mesh_recovery(
            self.recovery.request.clone(),
            self.fields.manifest.clone(),
            SequenceIdentity::from_solver_contract(
                &self.study.preparation.workload().model.sequence,
            )?,
            self.step().clone(),
        )?)
    }
    fn finalization_source(&self) -> AnalysisBridgeResult<ObservationHistoryFinalizationSource> {
        Ok(ObservationHistoryFinalizationSource::VolumeMesh {
            mesh: self.study.input(Kind::Mesh)?,
        })
    }
    fn history_context(&self) -> ObservationHistorySourceContext<'_> {
        ObservationHistorySourceContext::VolumeMesh {
            mesh: &self.study.preparation.mesh,
        }
    }
}

impl AcousticResponseStudyCoordinator {
    fn input(&self, kind: Kind) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
        self.solve_task
            .as_ref()
            .expect("completed solve")
            .inputs
            .iter()
            .find(|input| input.manifest.kind == kind)
            .cloned()
            .ok_or_else(|| invalid("acoustic response original solve input absent"))
    }

    pub fn prepare_field_recovery(
        &mut self,
    ) -> AnalysisBridgeResult<&crate::PreparedAcousticResponseFieldRecovery> {
        self.require(AcousticResponseStudyPhase::ReadyForFieldRecovery)?;
        let task =
            prepare_acoustic_response_field_recovery(AcousticResponseFieldRecoveryPreparation {
                source_host: self
                    .solve_task
                    .as_ref()
                    .expect("completed solve")
                    .host
                    .clone(),
                workload: self.input(Kind::AcousticResponseWorkload)?,
                mesh: self.input(Kind::Mesh)?,
                source_result: self
                    .result
                    .as_ref()
                    .expect("completed result")
                    .objects
                    .clone(),
                program_revision: self.preparation.program_revision.clone(),
                output_policy: self.preparation.output_policy,
                local_execution: self.preparation.local_execution,
            })?;
        self.recovery_task = Some(task);
        self.phase = AcousticResponseStudyPhase::AwaitingFieldRecovery;
        Ok(self.recovery_task.as_ref().expect("just prepared"))
    }

    pub fn complete_field_recovery(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(AcousticResponseStudyPhase::AwaitingFieldRecovery)?;
        let task = self.recovery_task.as_ref().expect("awaited recovery");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = AcousticResponseStudyResultClosure::new(
            AcousticResponseStudyResultPhase::FieldRecovery,
            &task.host,
            &response,
            &accounting,
        );
        let fields = import_acoustic_response_recovered_fields(
            source,
            response,
            task,
            self.preparation.output_policy.inventory_limits(),
        )?;
        let graph = accounting.resource_graph(&task.resource_plan)?;
        let usage = crate::extend_resource_usage(
            self.usage.as_ref().expect("completed solve usage"),
            &graph,
            &self.preparation.artifact_access,
            self.preparation.output_policy,
        )?;
        self.fields = Some(fields);
        self.usage = Some(usage);
        self.closures.push(closure);
        self.phase = AcousticResponseStudyPhase::ReadyForObservations;
        Ok(())
    }

    pub fn prepare_next_observation(
        &mut self,
        source: &impl CacheImport,
    ) -> AnalysisBridgeResult<Option<SampledObservationTask<'_>>> {
        self.require(AcousticResponseStudyPhase::ReadyForObservations)?;
        let mut workflow = std::mem::take(&mut self.observations);
        let prepared = {
            let context = Context {
                study: self,
                recovery: self.recovery_task.as_ref().expect("completed recovery"),
                fields: self.fields.as_ref().expect("completed fields"),
            };
            workflow
                .prepare(source, &context)
                .map(|task| task.is_some())
        };
        self.observations = workflow;
        if prepared? {
            self.phase = AcousticResponseStudyPhase::AwaitingObservation;
            return Ok(Some(self.observations.pending()?));
        }
        let catalog = self.observations.completed().catalog.clone();
        self.prepare_tail(catalog)?;
        self.phase = AcousticResponseStudyPhase::ReadyForAccuracy;
        Ok(None)
    }

    pub fn complete_observation(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(AcousticResponseStudyPhase::AwaitingObservation)?;
        let (closure, usage) = {
            let task = self.observations.pending()?;
            validate_accounting(task.host, task.resource_plan, &response, &accounting)?;
            let closure = AcousticResponseStudyResultClosure::new(
                AcousticResponseStudyResultPhase::Observation,
                task.host,
                &response,
                &accounting,
            );
            let graph = accounting.resource_graph(task.resource_plan)?;
            let usage = crate::extend_resource_usage(
                self.usage.as_ref().expect("completed recovery usage"),
                &graph,
                &self.preparation.artifact_access,
                self.preparation.output_policy,
            )?;
            (closure, usage)
        };
        let mut workflow = std::mem::take(&mut self.observations);
        let completed = {
            let context = Context {
                study: self,
                recovery: self.recovery_task.as_ref().expect("completed recovery"),
                fields: self.fields.as_ref().expect("completed fields"),
            };
            workflow.complete(source, response, &context)
        };
        self.observations = workflow;
        completed?;
        self.closures.push(closure);
        self.usage = Some(usage);
        self.phase = AcousticResponseStudyPhase::ReadyForObservations;
        Ok(())
    }
}
