//! Native fluid fields on the canonical solver mesh as a sampled-history observation source:
//! a steady flow's final state, or a transient's retained physical-time samples. The shared
//! workflow owns task, import, checkpoint and catalog mechanics.
use super::*;
use crate::{
    sampled_history_observations::{SampledHistorySource, SampledObservationTask},
    ObservationHistoryFinalizationSource, ObservationPreparation, ObservationPublishedSource,
    ObservationWorkloadSource,
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
    study: &'a FluidStudyCoordinator,
    recovery: &'a crate::PreparedFluidFieldRecovery,
    fields: &'a crate::ImportedRecoveredFields,
}

impl Context<'_> {
    /// Distinct retained samples in stored (increasing) order.
    fn samples(&self) -> Vec<RecoveredFieldSample> {
        let mut samples: Vec<RecoveredFieldSample> = Vec::new();
        for entry in &self.fields.manifest.fields {
            if samples.last() != Some(&entry.sample) {
                samples.push(entry.sample);
            }
        }
        samples
    }
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
        self.samples().len()
    }
    fn observation_preparation(
        &self,
        _source: &impl CacheImport,
        ordinal: usize,
    ) -> AnalysisBridgeResult<(ObservationPreparation, RecoveredFieldSample)> {
        let sample = *self
            .samples()
            .get(ordinal)
            .ok_or_else(|| invalid("fluid observation sample outside retained fields"))?;
        let policy = self.output_policy();
        Ok((
            ObservationPreparation {
                source_host: self.recovery.host.clone(),
                source: ObservationPublishedSource::VolumeMeshRecovery {
                    recovery_request: self.recovery.request.clone(),
                    fields: self.fields.clone(),
                },
                source_workload: ObservationWorkloadSource::VolumeMesh(
                    self.study.input(Kind::FluidSteadyWorkload)?,
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

impl FluidStudyCoordinator {
    /// Prepare exactly one owned observation phase. None means exact authored coverage is
    /// closed and the study moves on to accuracy.
    pub fn prepare_next_observation(
        &mut self,
        source: &impl CacheImport,
    ) -> AnalysisBridgeResult<Option<SampledObservationTask<'_>>> {
        self.require(FluidStudyPhase::ReadyForObservations)?;
        let mut workflow = std::mem::take(&mut self.observation_state);
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
        self.observation_state = workflow;
        if prepared? {
            self.phase = FluidStudyPhase::AwaitingObservation;
            return Ok(Some(self.observation_state.pending()?));
        }
        let usage = self.usage.clone().expect("completed recovery usage");
        let catalog = self.observation_state.completed().catalog.clone();
        self.tail = Some(self.prepare_tail(&usage, catalog)?);
        self.phase = FluidStudyPhase::ReadyForAccuracy;
        Ok(None)
    }

    pub fn complete_observation(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(FluidStudyPhase::AwaitingObservation)?;
        let (closure, usage) = {
            let task = self.observation_state.pending()?;
            validate_accounting(task.host, task.resource_plan, &response, &accounting)?;
            let closure = FluidStudyResultClosure::new(
                FluidStudyResultPhase::Observation,
                task.host,
                &response,
                &accounting,
            );
            (closure, self.extend_usage(task.resource_plan, accounting)?)
        };
        let mut workflow = std::mem::take(&mut self.observation_state);
        let completed = {
            let context = Context {
                study: self,
                recovery: self.recovery_task.as_ref().expect("completed recovery"),
                fields: self.fields.as_ref().expect("completed fields"),
            };
            workflow.complete(source, response, &context)
        };
        self.observation_state = workflow;
        completed?;
        self.closures.push(closure);
        self.usage = Some(usage);
        self.phase = FluidStudyPhase::ReadyForObservations;
        Ok(())
    }
}
