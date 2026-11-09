use super::*;
use crate::execution::{
    FieldRecoveryRequestIdentity, SequenceIdentity, SolverResultIdentity, StepIdentity,
};
use crate::results::field::{
    RecoveredFieldDefinition, RecoveredFieldEntityCounts, RecoveredFieldSample,
};
use runmat_analysis_core::{
    AnalysisField, AnalysisFieldValues, Displacement, InitialConditionTarget,
    InitialConditionValue, Temperature,
};

fn artifact(
    quantity: ObservationQuantity,
    shape: ObservationValueShape,
    values: Vec<f64>,
) -> RecoveredFieldArtifact {
    let sequence = SequenceIdentity::new(Digest::sha256("source-sequence")).unwrap();
    let step = StepIdentity::new(sequence, Digest::sha256("source-step")).unwrap();
    let source =
        SolverResultIdentity::from_step_result(step, Digest::sha256("source-result")).unwrap();
    let request = crate::execution::FieldRecoveryRequest::new(
        step,
        crate::execution::FieldRecoverySource::SolverResult(source),
        crate::execution::FieldTopologyIdentity::Structural {
            topology: crate::constraints::CanonicalStructuralTopologyIdentity::Explicit {
                structural_topology_digest: Digest::sha256("topology"),
            },
        },
        crate::execution::FieldRecoveryPlan::default(),
    )
    .unwrap();
    let field = RecoveredField::new(
        RecoveredFieldDefinition::new(
            "initial",
            quantity,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::Nodes,
            shape.clone(),
            RecoveredFieldNumericRepresentation::Real,
        )
        .unwrap(),
        RecoveredFieldSample::Final,
        AnalysisField {
            field_id: "initial".into(),
            layout: runmat_analysis_core::AnalysisFieldLayout::new(
                quantity,
                runmat_analysis_core::FieldLocation::Node,
                shape.clone(),
            ),
            shape: if shape == ObservationValueShape::Scalar {
                vec![4]
            } else {
                vec![4, 3]
            },
            values: AnalysisFieldValues::HostF64(values),
        },
    )
    .unwrap();
    source_artifact(request.identity(), source, vec![field])
}

fn source_artifact(
    request: FieldRecoveryRequestIdentity,
    source: SolverResultIdentity,
    fields: Vec<RecoveredField>,
) -> RecoveredFieldArtifact {
    RecoveredFieldArtifact::new(
        request,
        crate::execution::FieldRecoverySource::SolverResult(source),
        Digest::sha256("topology"),
        RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes: 4,
            volume_elements: 1,
            volume_integration_points: 4,
            oriented_edges: 0,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: 0,
        },
        fields,
    )
    .unwrap()
}

fn binding(artifact: RecoveredFieldArtifact) -> InitialFieldBinding {
    InitialFieldBinding::new(reference(&artifact), artifact).unwrap()
}

fn reference(artifact: &RecoveredFieldArtifact) -> InitialFieldArtifactRef {
    InitialFieldArtifactRef {
        artifact_id: "published-source".into(),
        content_digest: artifact.identity().digest().to_string(),
        topology_digest: artifact.topology_digest.to_string(),
        field_id: "initial".into(),
    }
}

fn temperature(binding: &InitialFieldBinding) -> InitialCondition {
    InitialCondition::Temperature(InitialConditionValue::<Temperature> {
        initial_condition_id: "warm".into(),
        target: InitialConditionTarget::WholeModel,
        frame: CoordinateFrame::Global,
        value: InitialValue::FieldArtifact(binding.reference().clone()),
        applies_before_step_id: "consume".into(),
    })
}

#[test]
fn scalar_and_vector_admission_preserve_exact_source_content_without_cloning_values() {
    let thermal = binding(artifact(
        ObservationQuantity::Temperature,
        ObservationValueShape::Scalar,
        vec![300., 301., 302., 303.],
    ));
    let admitted = thermal
        .admit(&temperature(&thermal), Digest::sha256("topology"), 4)
        .unwrap();
    assert_eq!(admitted.values(), &[300., 301., 302., 303.]);
    assert_eq!(
        admitted.values().as_ptr(),
        thermal.artifact().fields()[0]
            .as_host_f64()
            .unwrap()
            .as_ptr()
    );
    assert_eq!(
        admitted.field().identity(),
        thermal.artifact().fields()[0].identity()
    );
    let mechanical = binding(artifact(
        ObservationQuantity::Displacement,
        ObservationValueShape::Vector3,
        (0..12).map(|i| i as f64 / 10.).collect(),
    ));
    let condition = InitialCondition::Displacement(InitialConditionValue::<Displacement> {
        initial_condition_id: "displaced".into(),
        target: InitialConditionTarget::WholeModel,
        frame: CoordinateFrame::Global,
        value: InitialValue::FieldArtifact(mechanical.reference().clone()),
        applies_before_step_id: "consume".into(),
    });
    assert_eq!(
        mechanical
            .admit(&condition, Digest::sha256("topology"), 4)
            .unwrap()
            .values()
            .len(),
        12
    );
    let wire = serde_json::to_value(&mechanical).unwrap();
    let cold: InitialFieldBinding = serde_json::from_value(wire).unwrap();
    assert_eq!(cold, mechanical);
    assert_eq!(
        cold.admit(&condition, Digest::sha256("topology"), 4)
            .unwrap()
            .values(),
        mechanical
            .admit(&condition, Digest::sha256("topology"), 4)
            .unwrap()
            .values()
    );
    let tensor = binding(artifact(
        ObservationQuantity::Displacement,
        ObservationValueShape::Tensor { extents: vec![3] },
        vec![0.1; 12],
    ));
    let mut condition = condition;
    let InitialCondition::Displacement(value) = &mut condition else {
        unreachable!()
    };
    value.value = InitialValue::FieldArtifact(tensor.reference().clone());
    assert_eq!(
        tensor
            .admit(&condition, Digest::sha256("topology"), 4)
            .unwrap()
            .values(),
        &[0.1; 12]
    );
    let InitialCondition::Displacement(value) = &mut condition else {
        unreachable!()
    };
    value.frame = CoordinateFrame::Element;
    assert!(tensor
        .admit(&condition, Digest::sha256("topology"), 4)
        .is_err());
}

#[test]
fn declaration_consumer_quantity_topology_and_completeness_are_not_interchangeable() {
    let thermal = binding(artifact(
        ObservationQuantity::Temperature,
        ObservationValueShape::Scalar,
        vec![300.; 4],
    ));
    assert!(thermal
        .admit(&temperature(&thermal), Digest::sha256("foreign"), 4)
        .is_err());
    assert!(thermal
        .admit(&temperature(&thermal), Digest::sha256("topology"), 3)
        .is_err());
    let pressure = binding(artifact(
        ObservationQuantity::AcousticPressure,
        ObservationValueShape::Scalar,
        vec![300.; 4],
    ));
    assert!(pressure
        .admit(&temperature(&pressure), Digest::sha256("topology"), 4)
        .is_err());
    let mut declared = thermal.reference().clone();
    declared.content_digest = Digest::sha256("different-content").to_string();
    assert!(InitialFieldBinding::new(declared, thermal.artifact().clone()).is_err());
    let mut wire = serde_json::to_value(&thermal).unwrap();
    wire["reference"]["field_id"] = serde_json::json!("absent");
    assert!(serde_json::from_value::<InitialFieldBinding>(wire).is_err());
    let mut condition = temperature(&thermal);
    let InitialCondition::Temperature(value) = &mut condition else {
        unreachable!()
    };
    value.value = InitialValue::Uniform(Temperature::new(300.));
    assert!(thermal
        .admit(&condition, Digest::sha256("topology"), 4)
        .is_err());
}

#[test]
fn initial_input_does_not_silently_select_a_history_sample_or_invalid_temperature() {
    let original = artifact(
        ObservationQuantity::Temperature,
        ObservationValueShape::Scalar,
        vec![300.; 4],
    );
    let first = original.fields()[0].clone();
    let second = RecoveredField::new(
        first.definition().clone(),
        RecoveredFieldSample::TimeSeconds(1.),
        first.storage().clone(),
    )
    .unwrap();
    let ambiguous = source_artifact(
        original.recovery_request,
        original.source.solver_result().unwrap(),
        vec![first, second],
    );
    assert!(InitialFieldBinding::new(reference(&ambiguous), ambiguous).is_err());
    let input = binding(artifact(
        ObservationQuantity::Temperature,
        ObservationValueShape::Scalar,
        vec![-1.; 4],
    ));
    assert!(input
        .admit(&temperature(&input), Digest::sha256("topology"), 4)
        .is_err());
    for value in [f64::NAN, f64::INFINITY] {
        assert!(RecoveredField::new(
            original.fields()[0].definition().clone(),
            RecoveredFieldSample::Final,
            AnalysisField {
                field_id: "initial".into(),
                layout: original.fields()[0].definition().layout(),
                shape: vec![4],
                values: AnalysisFieldValues::HostF64(vec![value; 4]),
            },
        )
        .is_err());
    }
}
