//! Exact original retained-field publication adapter; no numerical re-solve.
use super::*;
use crate::{
    extend_resource_usage, import_electrical_recovered_fields, prepare_electrical_field_recovery,
    ElectricalFieldRecoveryPreparation, PreparedAnalysisObjects, PreparedElectricalFieldRecovery,
};
use runmat_analysis_fea::execution::AnalysisNumericalObjectKind as Kind;

impl ElectricalStudyCoordinator {
    pub fn prepare_field_recovery(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedElectricalFieldRecovery> {
        self.require(ElectricalStudyPhase::ReadyForFieldRecovery)?;
        let task = prepare_electrical_field_recovery(ElectricalFieldRecoveryPreparation {
            source_host: self
                .solve_task
                .as_ref()
                .expect("completed solve")
                .host
                .clone(),
            workload: self.input(Kind::ElectricalStaticWorkload)?,
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
        self.phase = ElectricalStudyPhase::AwaitingFieldRecovery;
        Ok(self.recovery_task.as_ref().expect("just prepared"))
    }

    pub fn complete_field_recovery(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(ElectricalStudyPhase::AwaitingFieldRecovery)?;
        let task = self.recovery_task.as_ref().expect("awaited recovery");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = ElectricalStudyResultClosure::new(
            ElectricalStudyResultPhase::FieldRecovery,
            &task.host,
            &response,
            &accounting,
        );
        let fields = import_electrical_recovered_fields(
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
            self.phase = ElectricalStudyPhase::ReadyForAccuracy;
        } else {
            self.phase = ElectricalStudyPhase::ReadyForObservationFanout;
        }
        Ok(())
    }

    pub(super) fn input(&self, kind: Kind) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
        self.solve_task
            .as_ref()
            .ok_or_else(|| invalid("electrical solve task absent"))?
            .inputs
            .iter()
            .find(|input| input.manifest.kind == kind)
            .cloned()
            .ok_or_else(|| invalid("electrical original solve input absent"))
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
                receipt: result.numerical_receipt_objects.clone(),
                context_inputs: task
                    .inputs
                    .iter()
                    .filter(|input| {
                        matches!(input.manifest.kind,
                runmat_analysis_fea::execution::AnalysisNumericalObjectKind::ElectricalStaticWorkload
                | runmat_analysis_fea::execution::AnalysisNumericalObjectKind::Mesh)
                    })
                    .cloned()
                    .chain(std::iter::once(result.objects.clone()))
                    .collect(),
                accuracy_kind: AccuracySourceKind::ElectricalStatic,
                state: crate::steady_study::TerminalStatePublicationSource::Profiles {
                    profiles: Box::new(result.source_profiles.clone()),
                    kind: SemanticStatePublicationSource::StaticElectricalNativeFields,
                    topology: Digest::from_bytes(*self.preparation.mesh.canonical_digest.bytes()),
                },
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
                .ok_or_else(|| invalid("electrical solve usage absent"))?,
            &graph,
            &self.preparation.artifact_access,
            self.preparation.output_policy,
        )
    }
}
