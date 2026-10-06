//! Native finite-volume source adapter; shared engine owns task/import/catalog mechanics.
use super::*;
pub(super) use crate::sampled_study_observations::SampledStudyObservations as FluidObservations;
use crate::{
    ObservationPreparation, ObservationPublishedSource, ObservationWorkloadSource,
    PreparedObservationAssembly, PreparedObservationFanout, PreparedObservationReduction,
};
use runmat_analysis_fea::{
    execution::{AnalysisNumericalObjectKind as Kind, ObservationHistorySourceContext},
    results::observation::{ObservationFieldCatalog, ObservationLimits},
};

impl FluidStudyCoordinator {
    pub fn prepare_observation_fanout(
        &mut self,
        source: &impl CacheImport,
    ) -> AnalysisBridgeResult<&PreparedObservationFanout> {
        self.require(FluidStudyPhase::ReadyForObservationFanout)?;
        let recovery = self.recovery_task.as_ref().expect("completed recovery");
        let fields = self.fields.as_ref().expect("completed fields");
        let policy = self.preparation.output_policy;
        let task = FluidObservations::prepare_fanout(
            source,
            ObservationPreparation {
                source_host: recovery.host.clone(),
                source: ObservationPublishedSource::FiniteVolumeRecovery {
                    recovery_request: recovery.request.clone(),
                    fields: fields.clone(),
                    layout: self
                        .result
                        .as_ref()
                        .expect("completed solve")
                        .result
                        .native_field_topology(
                            self.preparation.workload(),
                            &self.preparation.mesh,
                        )?,
                    resource_plan: recovery
                        .inputs
                        .iter()
                        .find(|input| input.manifest.kind == Kind::ResourcePlan)
                        .cloned()
                        .ok_or_else(|| invalid("fluid recovery plan absent"))?,
                },
                source_workload: ObservationWorkloadSource::FluidSteady(
                    self.input(Kind::FluidSteadyWorkload)?,
                ),
                mesh: Some(self.input(Kind::Mesh)?),
                accelerator_requests: Vec::new(),
                sequence: SequenceIdentity::from_solver_contract(
                    &self.preparation.workload().model.sequence,
                )?,
                step: self.selected_step().clone(),
                limits: ObservationLimits::default(),
                program_revision: self.preparation.program_revision.clone(),
                chunk_bytes: policy.chunk_bytes(),
                inventory_limits: policy.inventory_limits(),
            },
        )?;
        if task.partitions.is_empty() {
            return Err(invalid(
                "fluid authored observations have no executable partitions",
            ));
        }
        self.observation_state.fanout = Some(task);
        self.phase = FluidStudyPhase::AwaitingObservationPartitions;
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
        self.require(FluidStudyPhase::AwaitingObservationPartitions)?;
        if self.observation_state.contributions.contains_key(&ordinal) {
            return Err(invalid("fluid observation partition already completed"));
        }
        let fanout = self
            .observation_state
            .fanout
            .as_ref()
            .expect("prepared fanout");
        let task = fanout
            .partitions
            .get(ordinal)
            .ok_or_else(|| invalid("fluid partition ordinal outside fanout"))?;
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = FluidStudyResultClosure::new(
            FluidStudyResultPhase::ObservationPartition { ordinal },
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
            self.phase = FluidStudyPhase::ReadyForObservationReduction { ordinal: 0 };
        }
        Ok(())
    }

    pub fn prepare_observation_reduction(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedObservationReduction> {
        let ordinal = self.observation_state.reduced.len();
        self.require(FluidStudyPhase::ReadyForObservationReduction { ordinal })?;
        let task = self.observation_state.prepare_reduction(
            &self.preparation.program_revision,
            self.preparation.output_policy,
        )?;
        self.observation_state.reduction = Some(task);
        self.phase = FluidStudyPhase::AwaitingObservationReduction { ordinal };
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
        self.require(FluidStudyPhase::AwaitingObservationReduction { ordinal })?;
        let task = self
            .observation_state
            .reduction
            .as_ref()
            .expect("awaited reduction");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = FluidStudyResultClosure::new(
            FluidStudyResultPhase::ObservationReduction { ordinal },
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
            FluidStudyPhase::ReadyForObservationAssembly
        } else {
            FluidStudyPhase::ReadyForObservationReduction {
                ordinal: ordinal + 1,
            }
        };
        Ok(())
    }

    pub fn prepare_observation_assembly(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedObservationAssembly> {
        self.require(FluidStudyPhase::ReadyForObservationAssembly)?;
        let task = self.observation_state.prepare_assembly(
            &self.preparation.program_revision,
            self.preparation.output_policy,
        )?;
        self.observation_state.assembly = Some(task);
        self.phase = FluidStudyPhase::AwaitingObservationAssembly;
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
        self.require(FluidStudyPhase::AwaitingObservationAssembly)?;
        let task = self
            .observation_state
            .assembly
            .as_ref()
            .expect("awaited assembly");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = FluidStudyResultClosure::new(
            FluidStudyResultPhase::ObservationAssembly,
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
            ObservationFieldCatalog::finite_volume_recovery(
                recovery.request.clone(),
                fields.manifest.clone(),
                self.result
                    .as_ref()
                    .expect("completed solve")
                    .result
                    .native_field_topology(self.preparation.workload(), &self.preparation.mesh)?,
            )?,
            self.selected_step(),
            ObservationHistorySourceContext::FluidSteady {
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
        self.phase = FluidStudyPhase::ReadyForAccuracy;
        Ok(())
    }
}
