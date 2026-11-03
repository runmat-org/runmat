//! Independent nonuniform nodal sources exercise realization and closed evidence.
use super::*;
use crate::execution::{
    FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoverySource, FieldTopologyIdentity,
    InitialFieldBinding, SequenceIdentity, SolverResultIdentity, StepIdentity,
};
use crate::results::field::{
    RecoveredField, RecoveredFieldArtifact, RecoveredFieldDefinition, RecoveredFieldEntityCounts,
    RecoveredFieldEntityDomain, RecoveredFieldNumericRepresentation, RecoveredFieldSample,
};
use runmat_analysis_core::{
    AnalysisField, AnalysisFieldValues, FieldLocation, ObservationQuantity, ObservationValueShape,
};

fn binding(
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    quantity: ObservationQuantity,
    values: Vec<f64>,
) -> InitialFieldBinding {
    let step = StepIdentity::new(
        SequenceIdentity::new(Digest::sha256("independent-source")).unwrap(),
        Digest::sha256("source-step"),
    )
    .unwrap();
    let source =
        SolverResultIdentity::from_step_result(step, Digest::sha256("source-result")).unwrap();
    let request = FieldRecoveryRequest::new(
        step,
        FieldRecoverySource::SolverResult(source),
        FieldTopologyIdentity::VolumeMesh {
            mesh_digest: mesh.canonical_digest,
        },
        FieldRecoveryPlan::default(),
    )
    .unwrap();
    let field = RecoveredField::new(
        RecoveredFieldDefinition::new(
            "initial",
            quantity,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::Nodes,
            ObservationValueShape::Vector3,
            RecoveredFieldNumericRepresentation::Real,
        )
        .unwrap(),
        RecoveredFieldSample::Final,
        AnalysisField {
            field_id: "initial".into(),
            layout: runmat_analysis_core::AnalysisFieldLayout::vector3(quantity, runmat_analysis_core::FieldLocation::Node),
            shape: vec![mesh.topology.nodes.len(), 3],
            values: AnalysisFieldValues::HostF64(values),
        },
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
            artifact_id: "immutable-source".into(),
            content_digest: artifact.identity().digest().to_string(),
            topology_digest: artifact.topology_digest.to_string(),
            field_id: "initial".into(),
        },
        artifact,
    )
    .unwrap()
}

#[test]
fn cold_nonuniform_fields_preserve_every_original_component_and_source() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.sequence.steps = vec![ProcedureStep::structural_transient(STEP_ID)];
        model.structural = None;
        let u = (0..mesh.topology.nodes.len() * 3)
            .map(|i| (i as f64 - 5.) / 128.)
            .collect::<Vec<_>>();
        let v = u.iter().map(|x| -3. * x).collect::<Vec<_>>();
        let bindings = vec![
            binding(&mesh, ObservationQuantity::Displacement, u.clone()),
            binding(&mesh, ObservationQuantity::Velocity, v.clone()),
        ];
        model.initial_conditions = vec![
            InitialCondition::Displacement(InitialConditionValue::<Displacement> {
                initial_condition_id: "u".into(),
                target: InitialConditionTarget::WholeModel,
                frame: CoordinateFrame::Global,
                value: InitialValue::FieldArtifact(bindings[0].reference().clone()),
                applies_before_step_id: STEP_ID.into(),
            }),
            InitialCondition::Velocity(InitialConditionValue::<Velocity> {
                initial_condition_id: "v".into(),
                target: InitialConditionTarget::WholeModel,
                frame: CoordinateFrame::Global,
                value: InitialValue::FieldArtifact(bindings[1].reference().clone()),
                applies_before_step_id: STEP_ID.into(),
            }),
        ];
        let bindings: Vec<InitialFieldBinding> =
            serde_json::from_slice(&serde_json::to_vec(&bindings).unwrap()).unwrap();
        let layout = StructuralDofLayout::from_node_sets(vec![
            StructuralNodeDofSet::translational();
            mesh.topology.nodes.len()
        ]);
        let state = super::super::realize_with_fields(
            &model,
            Some(&mesh),
            &layout,
            StructuralInitialStateRealizationBasis::CanonicalTopology,
            StructuralInitialConstraintView::unconstrained(),
            STEP_ID,
            &bindings,
        )
        .unwrap();
        assert_eq!(state.displacement, u);
        assert_eq!(state.velocity, v);
        assert!(state
            .evidence
            .applied_conditions
            .iter()
            .all(|c| c.realized_component_bits.is_none()));
        let cold: crate::contracts::StructuralInitialStateEvidence =
            serde_json::from_slice(&serde_json::to_vec(&state.evidence).unwrap()).unwrap();
        assert_eq!(cold, state.evidence);
        let mut tampered = serde_json::to_value(&cold).unwrap();
        tampered["applied_conditions"][0]["realized_component_bits"] = serde_json::json!([0, 0, 0]);
        assert!(
            serde_json::from_value::<crate::contracts::StructuralInitialStateEvidence>(tampered)
                .is_err()
        );
        let step = &model.sequence.steps[0];
        assert!(crate::execution::validate_initial_field_selection(
            &model,
            step,
            Digest::sha256("foreign"),
            mesh.topology.nodes.len(),
            &bindings
        )
        .is_err());
        assert!(crate::execution::validate_initial_field_declarations(
            &model,
            step,
            &bindings[..1]
        )
        .is_err());
        let mut duplicates = bindings.clone();
        duplicates.push(bindings[0].clone());
        assert!(
            crate::execution::validate_initial_field_declarations(&model, step, &duplicates)
                .is_err()
        );
        let mut wrong = model.clone();
        wrong.initial_conditions.remove(1);
        assert!(
            crate::execution::validate_initial_field_declarations(&wrong, step, &bindings).is_err()
        );
    }
}
