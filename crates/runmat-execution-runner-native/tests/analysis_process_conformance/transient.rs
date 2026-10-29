use async_trait::async_trait;
use runmat_analysis_core::{
    AnalysisConstraint, ExcitationTarget, PrescribedStructuralDofs, RegionTarget,
    StructuralConstraint, StructuralConstraintKind,
};
use runmat_analysis_execution::{
    prepare_structural_transient_sequence, StructuralTransientRecoveryClosure,
    StructuralTransientSequencePlan, StructuralTransientSequencePreparationInput,
};
use runmat_analysis_fea::execution::{StructuralTransientBlockPlan, StructuralTransientWorkload};
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::solve::transient::StructuralTimeStepPolicy;
use runmat_analysis_fea::ComputeBackend;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ArtifactAccess;
use runmat_execution_runner_native::{NativeExecutionResult, StructuralTransientRecoveryPublisher};
use runmat_meshing_core::ElementOrder;

pub(super) struct RecordingPublisher {
    pub(super) closures: Vec<StructuralTransientRecoveryClosure>,
}

pub(super) struct FailAfterFirstPublisher {
    pub(super) closures: Vec<StructuralTransientRecoveryClosure>,
}

#[async_trait]
impl StructuralTransientRecoveryPublisher for RecordingPublisher {
    async fn publish(
        &mut self,
        closure: StructuralTransientRecoveryClosure,
    ) -> NativeExecutionResult<()> {
        self.closures.push(closure);
        Ok(())
    }
}

#[async_trait]
impl StructuralTransientRecoveryPublisher for FailAfterFirstPublisher {
    async fn publish(
        &mut self,
        closure: StructuralTransientRecoveryClosure,
    ) -> NativeExecutionResult<()> {
        self.closures.push(closure);
        Err(
            runmat_execution_runner_native::NativeExecutionError::Protocol(
                "injected driver loss after first accepted transient block".into(),
            ),
        )
    }
}

pub(super) fn preparation(
    revision: ProgramRevision,
    authorization_scope: &str,
) -> StructuralTransientSequencePlan {
    preparation_with_encryption(revision, authorization_scope, "remote-transient-encryption")
}

pub(super) fn preparation_with_encryption(
    revision: ProgramRevision,
    authorization_scope: &str,
    encryption_context: &str,
) -> StructuralTransientSequencePlan {
    let (model, mesh) = structural_transient_fixture();
    let workload = StructuralTransientWorkload {
        schema_version: 2,
        initial_fields: Vec::new(),
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        solver: runmat_analysis_fea::execution::StructuralTransientSolverPlan::Direct,
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 4 },
        residual_target: 1.0e-10,
    };
    let limits = ObjectInventoryLimits::default();
    prepare_structural_transient_sequence(StructuralTransientSequencePreparationInput {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        workload: &workload,
        mesh: &mesh,
        modal_state: None,
        block_plan: StructuralTransientBlockPlan::new(2).unwrap(),
        initial_dependencies: Vec::new(),
        artifact_access: ArtifactAccess {
            authorization_scope: authorization_scope.into(),
            encryption_context: Digest::sha256(encryption_context),
        },
        program_revision: revision,
        chunk_bytes: 256,
        inventory_limits: limits,
    })
    .unwrap()
}

pub(super) fn structural_transient_fixture() -> (
    runmat_analysis_core::AnalysisModel,
    runmat_meshing_core::SolverMeshArtifact,
) {
    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    *model.material_assignments[0]
        .target
        .selected_region_id_mut()
        .expect("fixture uses selected material region") =
        runmat_meshing_core::fixtures::fixture_region("region");
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "fixed-face".into(),
        frame: runmat_analysis_core::CoordinateFrame::Global,
        kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("face:0")),
            translations: [
                runmat_analysis_core::StructuralTranslationDof::Ux,
                runmat_analysis_core::StructuralTranslationDof::Uy,
                runmat_analysis_core::StructuralTranslationDof::Uz,
            ]
            .into_iter()
            .map(|dof| runmat_analysis_core::TranslationDofValue {
                dof,
                value: runmat_analysis_core::PrescribedValue::Constant(
                    runmat_analysis_core::Displacement::new(0.0),
                ),
            })
            .collect(),
            rotations: Vec::new(),
        }),
    })];
    model.excitations[0].excitation_mut().target = ExcitationTarget::Region(
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("face:3")),
    );
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    (model, mesh)
}
