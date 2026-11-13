//! Original solved nodal histories become cold immutable initial inputs.
//! Equality verifies transport, not independent continuum qualification.
use super::*;
use crate::execution::{
    FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoverySource, FieldTopologyIdentity,
    InitialFieldBinding, SequenceIdentity, SolverResultIdentity, StepIdentity,
    StructuralTransientBlockRequest, StructuralTransientSolverPlan, StructuralTransientWorkload,
};
use crate::results::field::{
    RecoveredField, RecoveredFieldArtifact, RecoveredFieldDefinition, RecoveredFieldEntityCounts,
    RecoveredFieldEntityDomain, RecoveredFieldNumericRepresentation, RecoveredFieldSample,
};
use runmat_analysis_core::{
    AnalysisField, FieldLocation, InitialFieldArtifactRef, ObservationQuantity,
    ObservationValueShape, Velocity,
};
use runmat_execution::Digest;
mod planning;

fn source(
    mesh: &SolverMeshArtifact,
    result: &FeaTransientRunResult,
    field: AnalysisField,
    quantity: ObservationQuantity,
) -> InitialFieldBinding {
    let step = StepIdentity::new(
        SequenceIdentity::new(Digest::sha256("rod-producer")).unwrap(),
        Digest::sha256("rod-source-step"),
    )
    .unwrap();
    let source = SolverResultIdentity::from_step_result(
        step,
        Digest::sha256(serde_json::to_vec(result).unwrap()),
    )
    .unwrap();
    let request = FieldRecoveryRequest::new(
        step,
        FieldRecoverySource::SolverResult(source),
        FieldTopologyIdentity::VolumeMesh {
            mesh_digest: mesh.canonical_digest,
        },
        FieldRecoveryPlan::default(),
    )
    .unwrap();
    let id = field.field_id.clone();
    let field = RecoveredField::new(
        RecoveredFieldDefinition::new(
            &id,
            quantity,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::Nodes,
            ObservationValueShape::Vector3,
            RecoveredFieldNumericRepresentation::Real,
        )
        .unwrap(),
        RecoveredFieldSample::Final,
        field,
    )
    .unwrap();
    let artifact = RecoveredFieldArtifact::new(
        request.identity(),
        FieldRecoverySource::SolverResult(source),
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
        RecoveredFieldEntityCounts {
            nodes: mesh.topology.nodes.len(),
            volume_elements: mesh.topology.volume_elements.len(),
            oriented_faces: 0,
            volume_integration_points: 0,
            oriented_edges: 0,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: 0,
        },
        vec![field],
    )
    .unwrap();
    InitialFieldBinding::new(
        InitialFieldArtifactRef {
            artifact_id: "rod-terminal".into(),
            content_digest: artifact.identity().digest().to_string(),
            topology_digest: artifact.topology_digest.to_string(),
            field_id: id,
        },
        artifact,
    )
    .unwrap()
}

#[test]
fn solved_fields_survive_cold_workload_admission_and_first_dynamic_block() {
    for quadratic in [false, true] {
        let mesh = mesh(quadratic, 1);
        let producer = run(&mesh, false, 4);
        let u = producer.displacement_snapshots.last().unwrap().clone();
        let v = producer.velocity_snapshots.last().unwrap().clone();
        let fields = vec![
            source(
                &mesh,
                &producer,
                u.clone(),
                ObservationQuantity::Displacement,
            ),
            source(&mesh, &producer, v.clone(), ObservationQuantity::Velocity),
        ];
        let mut model = model(&mesh, false);
        model.initial_conditions = vec![
            InitialCondition::Displacement(InitialConditionValue::<Displacement> {
                initial_condition_id: "original-u".into(),
                target: InitialConditionTarget::WholeModel,
                frame: CoordinateFrame::Global,
                value: InitialValue::FieldArtifact(fields[0].reference().clone()),
                applies_before_step_id: "rod_transient".into(),
            }),
            InitialCondition::Velocity(InitialConditionValue::<Velocity> {
                initial_condition_id: "original-v".into(),
                target: InitialConditionTarget::WholeModel,
                frame: CoordinateFrame::Global,
                value: InitialValue::FieldArtifact(fields[1].reference().clone()),
                applies_before_step_id: "rod_transient".into(),
            }),
        ];
        drop(producer);
        if let IndependentDomain::TimeSeconds { interval_count, .. } =
            &mut model.sequence.steps[0].independent_domain
        {
            *interval_count = 4;
        }
        for criterion in &mut model.sequence.steps[0].accuracy.criteria {
            if criterion.contribution == ErrorContributionKind::Algebraic {
                criterion.tolerance = AccuracyTolerance::Relative(1e-10);
            }
        }
        let workload = StructuralTransientWorkload {
            schema_version: 2,
            initial_fields: fields,
            step_id: model.sequence.steps[0].step_id.clone(),
            model,
            backend: ComputeBackend::Cpu,
            solver: StructuralTransientSolverPlan::Direct,
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 4 },
            residual_target: 1e-10,
        };
        planning::verify_original_sources(&workload, &mesh);
        let encoded = workload.canonical_bytes().unwrap();
        let cold = StructuralTransientWorkload::from_canonical_bytes(&encoded).unwrap();
        let scope = super::super::prepare_transient_checkpoint_scope(&cold, &mesh, None).unwrap();
        let request = StructuralTransientBlockRequest::new(scope, None, 2).unwrap();
        let segment =
            super::super::run_transient_block(super::super::StructuralTransientRunBlockInput {
                workload: &cold,
                solver_mesh: &mesh,
                request: &request,
                predecessor: None,
                modal_state: None,
                observation_cursors: Vec::new(),
            })
            .unwrap();
        let direct = crate::run_transient_with_fields(
            &cold.model,
            ComputeBackend::Cpu,
            cold.solve_options(mesh.clone()),
            &cold.initial_fields,
        )
        .unwrap();
        assert_eq!(
            direct.displacement_snapshots[0].as_host_f64().unwrap(),
            u.as_host_f64().unwrap()
        );
        assert_eq!(
            direct.velocity_snapshots[0].as_host_f64().unwrap(),
            v.as_host_f64().unwrap()
        );
        let initial = segment.initial_sample().unwrap();
        assert_eq!(
            initial.state.displacement.as_slice(),
            u.as_host_f64().unwrap()
        );
        assert_eq!(initial.state.velocity.as_slice(), v.as_host_f64().unwrap());
        let mut missing = serde_json::to_value(&cold).unwrap();
        missing.as_object_mut().unwrap().remove("initial_fields");
        assert!(serde_json::from_value::<StructuralTransientWorkload>(missing).is_err());
        let mut insufficient = cold.clone();
        insufficient.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(1);
        assert!(insufficient.validate().is_err());
    }
}
