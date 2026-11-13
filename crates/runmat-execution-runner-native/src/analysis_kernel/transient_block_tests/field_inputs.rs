//! Native cold transport and continuation from original solved field inputs.
use super::*;
use runmat_analysis_core::{
    AnalysisField, Displacement, FieldLocation, InitialCondition, InitialConditionTarget,
    InitialConditionValue, InitialFieldArtifactRef, InitialValue, ObservationQuantity,
    ObservationValueShape, Velocity,
};
use runmat_analysis_fea::execution::{
    FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoverySource, FieldTopologyIdentity,
    InitialFieldBinding, SequenceIdentity, SolverResultIdentity, StepIdentity,
};
use runmat_analysis_fea::results::field::{
    RecoveredField, RecoveredFieldArtifact, RecoveredFieldDefinition, RecoveredFieldEntityCounts,
    RecoveredFieldEntityDomain, RecoveredFieldNumericRepresentation, RecoveredFieldSample,
};
use runmat_analysis_fea::{run_transient_with_options, FeaTransientRunResult};
use runmat_meshing_core::SolverMeshArtifact;

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
fn native_cold_field_inputs_preserve_original_state_and_partition_equivalence() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, _) = transient_fixture();
        let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
        let mut workload = StructuralTransientWorkload {
            schema_version: 2,
            initial_fields: Vec::new(),
            step_id: model.sequence.steps[0].step_id.clone(),
            model,
            backend: ComputeBackend::Cpu,
            solver: runmat_analysis_fea::execution::StructuralTransientSolverPlan::Direct,
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 4 },
            residual_target: 1e-10,
        };
        let producer = run_transient_with_options(
            &workload.model,
            ComputeBackend::Cpu,
            workload.solve_options(mesh.clone()),
        )
        .unwrap();
        let u = producer.displacement_snapshots.last().unwrap().clone();
        let v = producer.velocity_snapshots.last().unwrap().clone();
        workload.initial_fields = vec![
            source(
                &mesh,
                &producer,
                u.clone(),
                ObservationQuantity::Displacement,
            ),
            source(&mesh, &producer, v.clone(), ObservationQuantity::Velocity),
        ];
        workload.model.initial_conditions = vec![
            InitialCondition::Displacement(InitialConditionValue::<Displacement> {
                initial_condition_id: "original-u".into(),
                target: InitialConditionTarget::WholeModel,
                frame: runmat_analysis_core::CoordinateFrame::Global,
                value: InitialValue::FieldArtifact(workload.initial_fields[0].reference().clone()),
                applies_before_step_id: workload.step_id.0.clone(),
            }),
            InitialCondition::Velocity(InitialConditionValue::<Velocity> {
                initial_condition_id: "original-v".into(),
                target: InitialConditionTarget::WholeModel,
                frame: runmat_analysis_core::CoordinateFrame::Global,
                value: InitialValue::FieldArtifact(workload.initial_fields[1].reference().clone()),
                applies_before_step_id: workload.step_id.0.clone(),
            }),
        ];
        crate::structural_study_root::verify_original_field_transport(&workload.initial_fields);
        drop(producer);
        let workload =
            StructuralTransientWorkload::from_canonical_bytes(&workload.canonical_bytes().unwrap())
                .unwrap();
        chain::assert_two_block_chain(workload, mesh);
    }
}
