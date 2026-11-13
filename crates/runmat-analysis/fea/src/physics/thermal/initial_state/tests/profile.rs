use super::*;
use runmat_canonical_codec::{encode_contract, profile_serialized_shape};

#[test]
fn projected_initial_evidence_uses_exact_selected_ids_and_covers_realized_values() {
    let (mut model, _, projection) = thermal_model();
    let long_id = "温度".repeat(2_000);
    let active = uniform_condition(&long_id, InitialConditionTarget::WholeModel, 315.0);
    let mut inactive = uniform_condition("inactive", InitialConditionTarget::WholeModel, 350.0);
    let InitialCondition::Temperature(value) = &mut inactive else {
        unreachable!()
    };
    value.applies_before_step_id = "another_step".into();
    let other_physics = InitialCondition::ElectricPotential(InitialConditionValue {
        initial_condition_id: "not_thermal".into(),
        target: InitialConditionTarget::WholeModel,
        frame: CoordinateFrame::Global,
        value: InitialValue::Uniform(runmat_analysis_core::ElectricPotential::new(1.0)),
        applies_before_step_id: "thermal_step".into(),
    });
    model.initial_conditions = vec![active, inactive, other_physics];
    let step = &model.sequence.steps[0];
    let projected = evidence_profile(&model, step).unwrap();
    let realized = realize(
        &model,
        &projection,
        step.step_id.as_str(),
        293.15,
        &vec![None; projection.nodes.len()],
        &[],
    )
    .unwrap();
    assert_eq!(realized.evidence.applied_initial_condition_ids, [long_id]);
    assert_eq!(
        selected_condition_count(&model, step),
        realized.evidence.applied_initial_condition_ids.len()
    );
    let actual = profile_serialized_shape(
        &realized.evidence,
        crate::execution::THERMAL_WORKLOAD_LIMITS,
    )
    .unwrap();
    assert_eq!(actual, projected);
    model.initial_conditions.truncate(1);
    assert_eq!(
        projected,
        evidence_profile(&model, &model.sequence.steps[0]).unwrap()
    );
    model.sequence.steps[0].activations = vec![runmat_analysis_core::StepObjectActivation {
        object: runmat_analysis_core::StepObjectRef::InitialCondition(
            model.initial_conditions[0].initial_condition_id().into(),
        ),
        lifecycle: runmat_analysis_core::ActivationLifecycle::Deactivate {
            from_step: runmat_analysis_core::ProcedureStepId("prior".into()),
        },
    }];
    let inactive_profile = evidence_profile(&model, &model.sequence.steps[0]).unwrap();
    let inactive_state = realize(
        &model,
        &projection,
        "thermal_step",
        293.15,
        &vec![None; projection.nodes.len()],
        &[],
    )
    .unwrap();
    assert!(inactive_state
        .evidence
        .applied_initial_condition_ids
        .is_empty());
    assert_eq!(
        selected_condition_count(&model, &model.sequence.steps[0]),
        inactive_state.evidence.applied_initial_condition_ids.len()
    );
    assert_eq!(
        inactive_profile,
        profile_serialized_shape(
            &inactive_state.evidence,
            crate::execution::THERMAL_WORKLOAD_LIMITS
        )
        .unwrap()
    );
    model.initial_conditions.clear();
    assert_eq!(
        inactive_profile,
        evidence_profile(&model, &model.sequence.steps[0]).unwrap()
    );
    assert!(
        evidence_profile(&model, &model.sequence.steps[0])
            .unwrap()
            .encoded_bytes_upper_bound
            < projected.encoded_bytes_upper_bound
    );
}

#[test]
fn borrowed_initial_evidence_preserves_derived_json_and_canonical_wire() {
    #[derive(Serialize)]
    struct ThermalInitialStateEvidence<'a> {
        procedure_step_id: &'a str,
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        reference_temperature_k: runmat_types::physical::Temperature,
        applied_initial_condition_ids: &'a [String],
        authored_node_count: usize,
        prescribed_node_count: usize,
        authored_and_prescribed_node_count: usize,
        reference_node_count: usize,
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        minimum_temperature_k: runmat_types::physical::Temperature,
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        maximum_temperature_k: runmat_types::physical::Temperature,
    }
    let (mut model, _, projection) = thermal_model();
    model.initial_conditions = vec![uniform_condition(
        "warm",
        InitialConditionTarget::WholeModel,
        315.0,
    )];
    let actual = realize(
        &model,
        &projection,
        "thermal_step",
        293.15,
        &vec![None; projection.nodes.len()],
        &[],
    )
    .unwrap()
    .evidence;
    let old_wire = ThermalInitialStateEvidence {
        procedure_step_id: &actual.procedure_step_id,
        reference_temperature_k: runmat_types::physical::Temperature::new(
            actual.reference_temperature_k.canonical_value(),
        ),
        applied_initial_condition_ids: &actual.applied_initial_condition_ids,
        authored_node_count: actual.authored_node_count,
        prescribed_node_count: actual.prescribed_node_count,
        authored_and_prescribed_node_count: actual.authored_and_prescribed_node_count,
        reference_node_count: actual.reference_node_count,
        minimum_temperature_k: runmat_types::physical::Temperature::new(
            actual.minimum_temperature_k.canonical_value(),
        ),
        maximum_temperature_k: runmat_types::physical::Temperature::new(
            actual.maximum_temperature_k.canonical_value(),
        ),
    };
    assert_eq!(
        serde_json::to_vec(&actual).unwrap(),
        serde_json::to_vec(&old_wire).unwrap()
    );
    let limits = crate::execution::THERMAL_WORKLOAD_LIMITS;
    assert_eq!(
        encode_contract(b"test\0", "initial-evidence", &actual, limits).unwrap(),
        encode_contract(b"test\0", "initial-evidence", &old_wire, limits).unwrap()
    );
    let decoded: crate::ThermalInitialStateEvidence =
        serde_json::from_slice(&serde_json::to_vec(&actual).unwrap()).unwrap();
    assert_eq!(decoded, actual);
}

#[test]
fn initial_evidence_profile_enforces_shared_string_limit_on_selected_ids() {
    let (mut model, _, _) = thermal_model();
    let oversized = "x".repeat(crate::execution::THERMAL_WORKLOAD_LIMITS.maximum_string_bytes + 1);
    model.initial_conditions = vec![uniform_condition(
        &oversized,
        InitialConditionTarget::WholeModel,
        315.0,
    )];
    assert!(evidence_profile(&model, &model.sequence.steps[0]).is_err());
    let InitialCondition::Temperature(value) = &mut model.initial_conditions[0] else {
        unreachable!()
    };
    value.applies_before_step_id = "another_step".into();
    assert!(evidence_profile(&model, &model.sequence.steps[0]).is_ok());
}
