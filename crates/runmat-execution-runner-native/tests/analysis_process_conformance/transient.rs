use async_trait::async_trait;
use runmat_analysis_core::{
    AnalysisConstraint, ExcitationTarget, PrescribedStructuralDofs, RegionTarget,
    StructuralConstraint, StructuralConstraintKind,
};
use runmat_analysis_execution::{
    prepare_analysis_objects, prepare_structural_transient_workload_objects,
    StructuralTransientRecoveryClosure, StructuralTransientSequencePreparation,
};
use runmat_analysis_fea::execution::{AnalysisNumericalObjectKind, StructuralTransientWorkload};
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::solve::transient::StructuralTimeStepPolicy;
use runmat_analysis_fea::ComputeBackend;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
use runmat_execution_runner_native::{NativeExecutionResult, StructuralTransientRecoveryPublisher};
use runmat_meshing_core::{CanonicalMeshingContract, ElementOrder};

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
) -> StructuralTransientSequencePreparation {
    preparation_with_encryption(revision, authorization_scope, "remote-transient-encryption")
}

pub(super) fn preparation_with_encryption(
    revision: ProgramRevision,
    authorization_scope: &str,
    encryption_context: &str,
) -> StructuralTransientSequencePreparation {
    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    model.material_assignments[0].region_id = "region".into();
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "fixed-face".into(),
        frame: runmat_analysis_core::CoordinateFrame::Global,
        kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
            target: RegionTarget::boundary("face:0"),
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
    model.excitations[0].excitation_mut().target =
        ExcitationTarget::Region(RegionTarget::boundary("face:3"));
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let workload = StructuralTransientWorkload {
        schema_version: 1,
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 4 },
        residual_target: 1.0e-10,
    };
    let scope = runmat_analysis_fea::pipeline::transient::prepare_transient_checkpoint_scope(
        &workload, &mesh,
    )
    .unwrap();
    let limits = ObjectInventoryLimits::default();
    let workload_objects = prepare_structural_transient_workload_objects(
        &workload,
        scope.source_workload,
        256,
        ObjectNamespace::InputValue,
        limits,
    )
    .unwrap();
    let mesh_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
        &mesh.canonical_encode().unwrap(),
        256,
        ObjectNamespace::InputValue,
        limits,
    )
    .unwrap();
    StructuralTransientSequencePreparation {
        workload: workload_objects,
        mesh: mesh_objects,
        scope,
        terminal_schedule_ordinals: vec![2, 4],
        artifact_access: ArtifactAccess {
            authorization_scope: authorization_scope.into(),
            encryption_context: Digest::sha256(encryption_context),
        },
        program_revision: revision,
        chunk_bytes: 256,
        inventory_limits: limits,
    }
}
