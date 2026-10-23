//! Exact original retained-field publication adapter; no numerical re-solve.
use super::*;
use crate::{
    extend_resource_usage, import_fluid_recovered_fields, prepare_fluid_field_recovery,
    FluidFieldRecoveryPreparation, PreparedAnalysisObjects, PreparedFluidFieldRecovery,
};
use runmat_analysis_fea::execution::AnalysisNumericalObjectKind as Kind;

impl FluidStudyCoordinator {
    pub fn prepare_field_recovery(&mut self) -> AnalysisBridgeResult<&PreparedFluidFieldRecovery> {
        self.require(FluidStudyPhase::ReadyForFieldRecovery)?;
        let task = prepare_fluid_field_recovery(FluidFieldRecoveryPreparation {
            source_host: self
                .solve_task
                .as_ref()
                .expect("completed solve")
                .host
                .clone(),
            workload: self.input(Kind::FluidSteadyWorkload)?,
            mesh: self.input(Kind::Mesh)?,
            source_resource_plan: self.input(Kind::ResourcePlan)?,
            source_result: self
                .result
                .as_ref()
                .expect("completed solve")
                .objects
                .clone(),
            program_revision: self.preparation.program_revision.clone(),
            output_policy: self.preparation.output_policy,
        })?;
        self.recovery_task = Some(task);
        self.phase = FluidStudyPhase::AwaitingFieldRecovery;
        Ok(self.recovery_task.as_ref().expect("just prepared"))
    }

    pub fn complete_field_recovery(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(FluidStudyPhase::AwaitingFieldRecovery)?;
        let task = self.recovery_task.as_ref().expect("awaited recovery");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = FluidStudyResultClosure::new(
            FluidStudyResultPhase::FieldRecovery,
            &task.host,
            &response,
            &accounting,
        );
        let fields = import_fluid_recovered_fields(
            source,
            response,
            task,
            self.preparation.output_policy.inventory_limits(),
        )?;
        let usage = self.extend_usage(&task.resource_plan, accounting)?;
        let tail = if self.selected_step().observations.is_empty() {
            Some(self.prepare_tail(&usage, None)?)
        } else {
            None
        };
        self.fields = Some(fields);
        self.usage = Some(usage);
        self.closures.push(closure);
        if tail.is_some() {
            self.tail = tail;
            self.phase = FluidStudyPhase::ReadyForAccuracy;
        } else {
            self.phase = FluidStudyPhase::ReadyForObservations;
        }
        Ok(())
    }

    pub(super) fn input(&self, kind: Kind) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
        self.solve_task
            .as_ref()
            .ok_or_else(|| invalid("fluid solve task absent"))?
            .inputs
            .iter()
            .find(|input| input.manifest.kind == kind)
            .cloned()
            .ok_or_else(|| invalid("fluid original solve input absent"))
    }

    pub(super) fn prepare_tail(
        &self,
        usage: &PreparedResourceUsage,
        observations: Option<
            runmat_analysis_fea::results::observation::ObservationPublicationCatalog,
        >,
    ) -> AnalysisBridgeResult<SteadyStudyTail> {
        let workload = self.preparation.workload();
        let task = self.solve_task.as_ref().expect("completed solve");
        let result = self.result.as_ref().expect("completed solve");
        Ok(SteadyStudyTail::new(
            SteadyStudySource {
                host: task.host.clone(),
                result: result.objects.clone(),
                state: crate::steady_study::TerminalStatePublicationSource::FluidOriginal,
                receipt: result.numerical_receipt_objects.clone(),
                context_inputs: task
                    .inputs
                    .iter()
                    .cloned()
                    .chain(std::iter::once(result.objects.clone()))
                    .collect(),
                accuracy_kind: AccuracySourceKind::FluidSteady,
                sequence: SequenceIdentity::from_solver_contract(&workload.model.sequence)?,
                step: self.preparation.selected_step().clone(),
                observations,
                program_revision: self.preparation.program_revision.clone(),
                output_policy: self.preparation.output_policy,
            },
            usage.clone(),
        ))
    }

    pub(super) fn extend_usage(
        &self,
        plan: &runmat_analysis_fea::planning::resource::FeaResourcePlan,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<PreparedResourceUsage> {
        let graph = accounting.resource_graph(plan)?;
        extend_resource_usage(
            self.usage
                .as_ref()
                .ok_or_else(|| invalid("fluid solve usage absent"))?,
            &graph,
            &self.preparation.artifact_access,
            self.preparation.output_policy,
        )
    }
}
