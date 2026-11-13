//! Typed source adapter; shared engine owns sampled task/import/catalog mechanics.
use super::*;
use crate::phase_accounting::validate_analysis_phase_accounting as validate_accounting;
pub(super) use crate::sampled_study_observations::SampledStudyObservations as NonlinearObservations;
use crate::{
    ObservationPreparation, ObservationPublishedSource, ObservationWorkloadSource,
    PreparedObservationAssembly, PreparedObservationFanout, PreparedObservationReduction,
};
use runmat_analysis_fea::{
    execution::{AnalysisNumericalObjectKind as Kind, ObservationHistorySourceContext},
    results::observation::{ObservationFieldCatalog, ObservationLimits},
};

impl NonlinearSolidStudyCoordinator {
    pub fn prepare_observation_fanout(
        &mut self,
        source: &impl CacheImport,
    ) -> AnalysisBridgeResult<&PreparedObservationFanout> {
        self.require(NonlinearSolidStudyPhase::ReadyForObservationFanout)?;
        self.admit_retention(None, None)?;
        let recovery = self.recovery_task.as_ref().expect("completed recovery");
        let fields = self.fields.as_ref().expect("completed fields");
        let policy = self.preparation.output_policy;
        let task = NonlinearObservations::prepare_fanout(
            source,
            ObservationPreparation {
                source_host: recovery.host.clone(),
                source: ObservationPublishedSource::Structural {
                    recovery_request: recovery.request.clone(),
                    fields: fields.clone(),
                },
                source_workload: ObservationWorkloadSource::NonlinearSolid(
                    self.original_input(Kind::NonlinearSolidWorkload)?,
                ),
                mesh: Some(self.original_input(Kind::Mesh)?),
                accelerator_requests: Vec::new(),
                sequence: runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
                    &self.preparation.product.workload.model.sequence,
                )?,
                step: self.preparation.product.workload.model.sequence.steps[0].clone(),
                limits: ObservationLimits::default(),
                program_revision: self.preparation.program_revision.clone(),
                chunk_bytes: policy.chunk_bytes(),
                inventory_limits: policy.inventory_limits(),
                local_execution: self.preparation.local_execution,
            },
        )?;
        if task.partitions.is_empty() {
            return Err(invalid(
                "nonlinear authored observations have no executable partitions",
            ));
        }
        self.observation_state.fanout = Some(task);
        if let Err(error) = self.admit_retention(None, None) {
            self.observation_state.fanout = None;
            return Err(error);
        }
        self.phase = NonlinearSolidStudyPhase::AwaitingObservationPartitions;
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
        self.require(NonlinearSolidStudyPhase::AwaitingObservationPartitions)?;
        self.observation_state.partition(ordinal)
    }

    /// Completion order is arbitrary; each immutable task ordinal may complete exactly once.
    pub fn complete_observation_partition(
        &mut self,
        source: &impl CacheImport,
        ordinal: usize,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(NonlinearSolidStudyPhase::AwaitingObservationPartitions)?;
        if self.observation_state.contributions.contains_key(&ordinal) {
            return Err(invalid("nonlinear observation partition already completed"));
        }
        let fanout = self
            .observation_state
            .fanout
            .as_ref()
            .expect("prepared fanout");
        let task = fanout
            .partitions
            .get(ordinal)
            .ok_or_else(|| invalid("nonlinear partition ordinal outside fanout"))?;
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = NonlinearSolidStudyResultClosure::new(
            NonlinearSolidStudyResultPhase::ObservationPartition { ordinal },
            &task.host,
            response.clone(),
            accounting.clone(),
        );
        self.admit_retention(
            Some((
                closure.phase,
                &closure.workload,
                &closure.response,
                &closure.execution,
            )),
            Some(&task.resource_plan),
        )?;
        let contribution = self.observation_state.import_partition(
            source,
            ordinal,
            response,
            self.preparation.output_policy,
        )?;
        let task_attempts = accounting.resource_graph(&task.resource_plan)?.attempts;
        let count = fanout.partitions.len();
        self.observation_state
            .contributions
            .insert(ordinal, contribution);
        self.attempts.extend(task_attempts);
        self.closures.push(closure);
        if self.observation_state.contributions.len() == count {
            self.phase = NonlinearSolidStudyPhase::ReadyForObservationReduction { ordinal: 0 };
        }
        Ok(())
    }

    pub fn prepare_observation_reduction(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedObservationReduction> {
        let ordinal = self.observation_state.reduced.len();
        self.require(NonlinearSolidStudyPhase::ReadyForObservationReduction { ordinal })?;
        let task = self.observation_state.prepare_reduction(
            &self.preparation.program_revision,
            self.preparation.output_policy,
        )?;
        self.admit_retention(None, Some(&task.resource_plan))?;
        self.observation_state.reduction = Some(task);
        self.phase = NonlinearSolidStudyPhase::AwaitingObservationReduction { ordinal };
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
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        let ordinal = self.observation_state.reduced.len();
        self.require(NonlinearSolidStudyPhase::AwaitingObservationReduction { ordinal })?;
        let task = self
            .observation_state
            .reduction
            .as_ref()
            .expect("awaited reduction");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = NonlinearSolidStudyResultClosure::new(
            NonlinearSolidStudyResultPhase::ObservationReduction { ordinal },
            &task.host,
            response.clone(),
            accounting.clone(),
        );
        self.admit_retention(
            Some((
                closure.phase,
                &closure.workload,
                &closure.response,
                &closure.execution,
            )),
            Some(&task.resource_plan),
        )?;
        let reduced = self.observation_state.import_reduction(
            source,
            response,
            self.preparation.output_policy,
        )?;
        let task_attempts = accounting.resource_graph(&task.resource_plan)?.attempts;
        self.observation_state.push_reduced(reduced)?;
        self.attempts.extend(task_attempts);
        self.closures.push(closure);
        let count = self.observation_state.reduction_count();
        self.phase = if self.observation_state.reduced.len() == count {
            NonlinearSolidStudyPhase::ReadyForObservationAssembly
        } else {
            NonlinearSolidStudyPhase::ReadyForObservationReduction {
                ordinal: ordinal + 1,
            }
        };
        Ok(())
    }

    pub fn prepare_observation_assembly(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedObservationAssembly> {
        self.require(NonlinearSolidStudyPhase::ReadyForObservationAssembly)?;
        let task = self.observation_state.prepare_assembly(
            &self.preparation.program_revision,
            self.preparation.output_policy,
        )?;
        self.admit_retention(None, Some(&task.resource_plan))?;
        self.observation_state.assembly = Some(task);
        self.phase = NonlinearSolidStudyPhase::AwaitingObservationAssembly;
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
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(NonlinearSolidStudyPhase::AwaitingObservationAssembly)?;
        let task = self
            .observation_state
            .assembly
            .as_ref()
            .expect("awaited assembly");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = NonlinearSolidStudyResultClosure::new(
            NonlinearSolidStudyResultPhase::ObservationAssembly,
            &task.host,
            response.clone(),
            accounting.clone(),
        );
        self.admit_retention(
            Some((
                closure.phase,
                &closure.workload,
                &closure.response,
                &closure.execution,
            )),
            Some(&task.resource_plan),
        )?;
        let artifact = self.observation_state.import_assembly(
            source,
            response,
            self.preparation.output_policy,
        )?;
        let fields = self.fields.as_ref().expect("completed fields");
        let recovery = self.recovery_task.as_ref().expect("completed recovery");
        let catalog = self.observation_state.publication_catalog(
            &artifact,
            self.source_host
                .as_ref()
                .expect("original terminal block")
                .workload
                .step,
            ObservationFieldCatalog::structural(recovery.request.clone(), fields.manifest.clone())?,
            &self.preparation.product.workload.model.sequence.steps[0],
            ObservationHistorySourceContext::Structural {
                source: runmat_analysis_fea::execution::StructuralObservationSource::NonlinearSolid(
                    &self.preparation.product.workload,
                ),
                mesh: Some(&self.preparation.mesh),
            },
            runmat_analysis_fea::results::observation::ObservationPublicationScope::Complete,
        )?;
        let task_attempts = accounting.resource_graph(&task.resource_plan)?.attempts;
        let mut attempts = self.attempts.clone();
        attempts.extend(task_attempts.iter().cloned());
        let tail = self.build_original_tail(
            self.source_host.as_ref().expect("terminal host"),
            self.source_plan.as_ref().expect("terminal plan"),
            self.block.as_ref().expect("terminal block"),
            &attempts,
            Some(catalog.clone()),
            accounting.graph_wall_millis(),
        )?;
        self.observation_state.completed = Some(artifact);
        self.observation_state.catalog = Some(catalog);
        self.observation_state.fanout = None;
        self.observation_state.contributions.clear();
        self.observation_state.reduction = None;
        self.observation_state.reduced.clear();
        self.observation_state.assembly = None;
        self.recovery_task = None;
        self.attempts.extend(task_attempts);
        self.closures.push(closure);
        self.tail = Some(tail);
        self.phase = NonlinearSolidStudyPhase::ReadyForAccuracy;
        Ok(())
    }
}
