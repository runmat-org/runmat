//! Original field task and terminal-tail composition; shared engines own artifacts and accuracy.
use super::*;
use crate::phase_accounting::validate_analysis_phase_accounting as validate_accounting;
use crate::steady_study::{SteadyStudySource, TerminalStatePublicationSource};
use runmat_analysis_fea::{
    execution::{AccuracySourceKind, SequenceIdentity},
    planning::resource::FeaResourcePlan,
    results::observation::ObservationPublicationCatalog,
};
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::CanonicalMeshingContract;

impl NonlinearSolidStudyCoordinator {
    pub(super) fn require(&self, phase: NonlinearSolidStudyPhase) -> AnalysisBridgeResult<()> {
        if self.phase() != phase {
            return Err(invalid("nonlinear original lifecycle phase differs"));
        }
        Ok(())
    }
    pub(super) fn original_input(
        &self,
        kind: Kind,
    ) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
        let (identity, bytes) = match kind {
            Kind::NonlinearSolidWorkload => (
                self.preparation.product.workload.identity()?.digest(),
                self.preparation.product.workload.canonical_bytes()?,
            ),
            Kind::Mesh => (
                Digest::from_bytes(*self.preparation.mesh.canonical_digest.bytes()),
                self.preparation.mesh.canonical_encode().map_err(invalid)?,
            ),
            _ => return Err(invalid("nonlinear original source input kind differs")),
        };
        prepare_analysis_objects(
            kind,
            identity,
            &bytes,
            self.preparation.output_policy.chunk_bytes(),
            ObjectNamespace::InputValue,
            self.preparation.output_policy.inventory_limits(),
        )
    }
    pub fn prepare_field_recovery(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedNonlinearFieldRecovery> {
        self.require(NonlinearSolidStudyPhase::ReadyForFieldRecovery)?;
        self.admit_retention(None, None)?;
        let task = prepare_nonlinear_field_recovery(NonlinearFieldRecoveryPreparation {
            source_host: self.source_host.as_ref().expect("terminal source").clone(),
            workload: self.original_input(Kind::NonlinearSolidWorkload)?,
            mesh: self.original_input(Kind::Mesh)?,
            source_block: self.block.as_ref().expect("terminal block").objects.clone(),
            program_revision: self.preparation.program_revision.clone(),
            output_policy: self.preparation.output_policy,
            local_execution: self.preparation.local_execution,
        })?;
        self.admit_retention(None, Some(&task.resource_plan))?;
        self.recovery_task = Some(task);
        self.phase = NonlinearSolidStudyPhase::AwaitingFieldRecovery;
        Ok(self.recovery_task.as_ref().expect("prepared recovery"))
    }
    pub fn complete_field_recovery(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(NonlinearSolidStudyPhase::AwaitingFieldRecovery)?;
        let task = self.recovery_task.as_ref().expect("awaited recovery");
        let phase = NonlinearSolidStudyResultPhase::FieldRecovery;
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        self.admit_retention(
            Some((phase, &task.host.workload, &response, &accounting)),
            Some(&task.resource_plan),
        )?;
        let fields = import_nonlinear_recovered_fields(source, response.clone(), task)?;
        let closure =
            NonlinearSolidStudyResultClosure::new(phase, &task.host, response, accounting.clone());
        self.attempts
            .extend(accounting.resource_graph(&task.resource_plan)?.attempts);
        self.closures.push(closure);
        self.fields = Some(fields);
        self.phase = NonlinearSolidStudyPhase::ReadyForObservationFanout;
        Ok(())
    }
    pub(super) fn build_original_tail(
        &self,
        host: &AnalysisHostWorkload,
        plan: &FeaResourcePlan,
        block: &ImportedNonlinearSolidBlock,
        attempts: &[PlannedResourceUsageAttempt],
        observations: Option<ObservationPublicationCatalog>,
        incoming_wall_millis: u64,
    ) -> AnalysisBridgeResult<SteadyStudyTail> {
        // Native nonlinear phases execute sequentially; each closure retains the
        // enclosing host interval, independently of its workers' reserved time.
        let wall = self
            .closures
            .iter()
            .try_fold(incoming_wall_millis, |n, closure| {
                n.checked_add(closure.execution.graph_wall_millis())
            })
            .ok_or_else(|| invalid("nonlinear cumulative wall overflow"))?;
        let usage = prepare_planned_resource_graph_usage(
            plan,
            host.workload.identity.digest(),
            block.block.source_result_identity()?.digest(),
            &PlannedResourceUsageGraph::new(attempts.to_vec(), wall)?,
            &self.preparation.artifact_access,
            self.preparation.output_policy.chunk_bytes(),
            self.preparation.output_policy.inventory_limits(),
        )?;
        Ok(SteadyStudyTail::new(
            SteadyStudySource {
                host: host.clone(),
                result: block.objects.clone(),
                state: TerminalStatePublicationSource::NonlinearOriginal,
                receipt: block.receipt_objects.clone(),
                context_inputs: vec![
                    self.original_input(Kind::NonlinearSolidWorkload)?,
                    self.original_input(Kind::Mesh)?,
                    block.objects.clone(),
                ],
                accuracy_kind: AccuracySourceKind::NonlinearSolid,
                sequence: SequenceIdentity::from_solver_contract(
                    &self.preparation.product.workload.model.sequence,
                )?,
                step: self.preparation.product.workload.model.sequence.steps[0].clone(),
                observations,
                program_revision: self.preparation.program_revision.clone(),
                output_policy: self.preparation.output_policy,
            },
            usage,
        ))
    }
}
