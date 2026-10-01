//! Exact original retained-field publication adapter; no numerical re-solve.
use super::*;
use crate::{
    extend_resource_usage, import_magnetic_recovered_fields, prepare_magnetic_field_recovery,
    MagneticFieldRecoveryPreparation, PreparedAnalysisObjects, PreparedMagneticFieldRecovery,
};
use runmat_analysis_fea::execution::AnalysisNumericalObjectKind as Kind;

impl MagneticStudyCoordinator {
    pub fn prepare_field_recovery(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedMagneticFieldRecovery> {
        self.require(MagneticStudyPhase::ReadyForFieldRecovery)?;
        let task = prepare_magnetic_field_recovery(MagneticFieldRecoveryPreparation {
            source_host: self
                .solve_task
                .as_ref()
                .expect("completed solve")
                .host
                .clone(),
            workload: self.input(Kind::MagnetostaticWorkload)?,
            mesh: self.input(Kind::Mesh)?,
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
        self.phase = MagneticStudyPhase::AwaitingFieldRecovery;
        Ok(self.recovery_task.as_ref().expect("just prepared"))
    }

    pub fn complete_field_recovery(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AttemptResourceAccounting,
    ) -> AnalysisBridgeResult<()> {
        self.require(MagneticStudyPhase::AwaitingFieldRecovery)?;
        let task = self.recovery_task.as_ref().expect("awaited recovery");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = MagneticStudyResultClosure::new(
            MagneticStudyResultPhase::FieldRecovery,
            &task.host,
            &response,
            &accounting,
        );
        let fields = import_magnetic_recovered_fields(
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
            self.phase = MagneticStudyPhase::ReadyForAccuracy;
        } else {
            self.phase = MagneticStudyPhase::ReadyForObservationFanout;
        }
        Ok(())
    }

    pub(super) fn input(&self, kind: Kind) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
        self.solve_task
            .as_ref()
            .ok_or_else(|| invalid("magnetic solve task absent"))?
            .inputs
            .iter()
            .find(|input| input.manifest.kind == kind)
            .cloned()
            .ok_or_else(|| invalid("magnetic original solve input absent"))
    }

    pub(super) fn prepare_tail(
        &self,
        usage: &PreparedResourceUsage,
        observations: Option<
            runmat_analysis_fea::results::observation::ObservationPublicationCatalog,
        >,
    ) -> AnalysisBridgeResult<StaticEmStudyTail> {
        let workload = self.preparation.workload();
        let task = self.solve_task.as_ref().expect("completed solve");
        let result = self.result.as_ref().expect("completed solve");
        Ok(StaticEmStudyTail::new(
            StaticEmStudySource {
                host: task.host.clone(),
                result: result.objects.clone(),
                profiles: result.source_profiles.clone(),
                receipt: result.numerical_receipt_objects.clone(),
                context_inputs: task
                    .inputs
                    .iter()
                    .filter(|input| {
                        matches!(input.manifest.kind,
                runmat_analysis_fea::execution::AnalysisNumericalObjectKind::MagnetostaticWorkload
                | runmat_analysis_fea::execution::AnalysisNumericalObjectKind::Mesh)
                    })
                    .cloned()
                    .chain(std::iter::once(result.objects.clone()))
                    .collect(),
                accuracy_kind: AccuracySourceKind::Magnetostatic,
                state_kind: SemanticStatePublicationSource::MagnetostaticNativeFields,
                topology: Digest::from_bytes(
                    *result
                        .result
                        .sampling_topology()
                        .identity()
                        .map_err(|error| invalid(error.to_string()))?
                        .discretization_digest()
                        .bytes(),
                ),
                sequence: SequenceIdentity::from_solver_contract(&workload.model.sequence)?,
                step: self.preparation.selected_step().clone(),
                formulation_requirement: workload.formulation_requirement,
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
        accounting: AttemptResourceAccounting,
    ) -> AnalysisBridgeResult<PreparedResourceUsage> {
        let wall = accounting.resources.usage.wall_millis;
        let graph = PlannedResourceUsageGraph::new(
            vec![PlannedResourceUsageAttempt {
                plan: plan.clone(),
                accounting,
            }],
            wall,
        )?;
        extend_resource_usage(
            self.usage
                .as_ref()
                .ok_or_else(|| invalid("magnetic solve usage absent"))?,
            &graph,
            &self.preparation.artifact_access,
            self.preparation.output_policy,
        )
    }
}
