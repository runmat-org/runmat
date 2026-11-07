//! Typed source adapter; shared engine owns sampled task/import/catalog mechanics.
use super::*;
pub(super) use crate::sampled_study_observations::SampledStudyObservations as MagneticObservations;
use crate::{
    ObservationPreparation, ObservationPublishedSource, ObservationWorkloadSource,
    PreparedObservationAssembly, PreparedObservationFanout, PreparedObservationReduction,
};
use runmat_analysis_fea::{
    execution::{AnalysisNumericalObjectKind as Kind, ObservationHistorySourceContext},
    results::observation::{ObservationFieldCatalog, ObservationLimits},
};

impl MagneticStudyCoordinator {
    pub fn prepare_observation_fanout(
        &mut self,
        source: &impl CacheImport,
    ) -> AnalysisBridgeResult<&PreparedObservationFanout> {
        self.require(MagneticStudyPhase::ReadyForObservationFanout)?;
        let recovery = self.recovery_task.as_ref().expect("completed recovery");
        let fields = self.fields.as_ref().expect("completed fields");
        let policy = self.preparation.output_policy;
        let task = MagneticObservations::prepare_fanout(
            source,
            ObservationPreparation {
                source_host: recovery.host.clone(),
                source: ObservationPublishedSource::SampledRecovery {
                    recovery_request: recovery.request.clone(),
                    fields: fields.clone(),
                    sampling: self
                        .result
                        .as_ref()
                        .expect("completed solve")
                        .result
                        .sampling_topology()
                        .clone(),
                },
                source_workload: ObservationWorkloadSource::Magnetostatic(
                    self.input(Kind::MagnetostaticWorkload)?,
                ),
                mesh: Some(self.input(Kind::Mesh)?),
                accelerator_requests: Vec::new(),
                sequence: SequenceIdentity::from_solver_contract(
                    &self.preparation.workload().model.sequence,
                )?,
                step: self.selected_step().clone(),
                limits: ObservationLimits::for_step(self.selected_step()),
                program_revision: self.preparation.program_revision.clone(),
                chunk_bytes: policy.chunk_bytes(),
                inventory_limits: policy.inventory_limits(),
                local_execution: self.preparation.local_execution,
            },
        )?;
        if task.partitions.is_empty() {
            return Err(invalid(
                "magnetic authored observations have no executable partitions",
            ));
        }
        self.observation_state.fanout = Some(task);
        self.phase = MagneticStudyPhase::AwaitingObservationPartitions;
        Ok(self
            .observation_state
            .fanout
            .as_ref()
            .expect("just prepared"))
    }

    /// Completion order is arbitrary; each immutable task ordinal may complete exactly once.
    pub fn complete_observation_partition(
        &mut self,
        source: &impl CacheImport,
        ordinal: usize,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(MagneticStudyPhase::AwaitingObservationPartitions)?;
        if self.observation_state.contributions.contains_key(&ordinal) {
            return Err(invalid("magnetic observation partition already completed"));
        }
        let fanout = self
            .observation_state
            .fanout
            .as_ref()
            .expect("prepared fanout");
        let task = fanout
            .partitions
            .get(ordinal)
            .ok_or_else(|| invalid("magnetic partition ordinal outside fanout"))?;
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = MagneticStudyResultClosure::new(
            MagneticStudyResultPhase::ObservationPartition { ordinal },
            &task.host,
            &response,
            &accounting,
        );
        let contribution = self.observation_state.import_partition(
            source,
            ordinal,
            response,
            self.preparation.output_policy,
        )?;
        let usage = self.extend_usage(&task.resource_plan, accounting)?;
        let count = fanout.partitions.len();
        self.observation_state
            .contributions
            .insert(ordinal, contribution);
        self.usage = Some(usage);
        self.closures.push(closure);
        if self.observation_state.contributions.len() == count {
            self.phase = MagneticStudyPhase::ReadyForObservationReduction { ordinal: 0 };
        }
        Ok(())
    }

    pub fn prepare_observation_reduction(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedObservationReduction> {
        let ordinal = self.observation_state.reduced.len();
        self.require(MagneticStudyPhase::ReadyForObservationReduction { ordinal })?;
        let task = self.observation_state.prepare_reduction(
            &self.preparation.program_revision,
            self.preparation.output_policy,
        )?;
        self.observation_state.reduction = Some(task);
        self.phase = MagneticStudyPhase::AwaitingObservationReduction { ordinal };
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
        self.require(MagneticStudyPhase::AwaitingObservationReduction { ordinal })?;
        let task = self
            .observation_state
            .reduction
            .as_ref()
            .expect("awaited reduction");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = MagneticStudyResultClosure::new(
            MagneticStudyResultPhase::ObservationReduction { ordinal },
            &task.host,
            &response,
            &accounting,
        );
        let reduced = self.observation_state.import_reduction(
            source,
            response,
            self.preparation.output_policy,
        )?;
        let usage = self.extend_usage(&task.resource_plan, accounting)?;
        self.observation_state.reduced.push(reduced);
        self.usage = Some(usage);
        self.closures.push(closure);
        let count = self.observation_state.reduction_count();
        self.phase = if self.observation_state.reduced.len() == count {
            MagneticStudyPhase::ReadyForObservationAssembly
        } else {
            MagneticStudyPhase::ReadyForObservationReduction {
                ordinal: ordinal + 1,
            }
        };
        Ok(())
    }

    pub fn prepare_observation_assembly(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedObservationAssembly> {
        self.require(MagneticStudyPhase::ReadyForObservationAssembly)?;
        let task = self.observation_state.prepare_assembly(
            &self.preparation.program_revision,
            self.preparation.output_policy,
        )?;
        self.observation_state.assembly = Some(task);
        self.phase = MagneticStudyPhase::AwaitingObservationAssembly;
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
        self.require(MagneticStudyPhase::AwaitingObservationAssembly)?;
        let task = self
            .observation_state
            .assembly
            .as_ref()
            .expect("awaited assembly");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = MagneticStudyResultClosure::new(
            MagneticStudyResultPhase::ObservationAssembly,
            &task.host,
            &response,
            &accounting,
        );
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
            ObservationFieldCatalog::sampled_recovery(
                recovery.request.clone(),
                fields.manifest.clone(),
                self.result
                    .as_ref()
                    .expect("completed solve")
                    .result
                    .sampling_topology()
                    .clone(),
            )?,
            self.selected_step(),
            ObservationHistorySourceContext::Magnetostatic {
                workload: self.preparation.workload(),
                mesh: &self.preparation.mesh,
            },
        )?;
        let usage = self.extend_usage(&task.resource_plan, accounting)?;
        let tail = self.prepare_tail(&usage, Some(catalog.clone()))?;
        self.observation_state.completed = Some(artifact);
        self.observation_state.catalog = Some(catalog);
        self.usage = Some(usage);
        self.closures.push(closure);
        self.tail = Some(tail);
        self.phase = MagneticStudyPhase::ReadyForAccuracy;
        Ok(())
    }
}
