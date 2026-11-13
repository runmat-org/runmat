use runmat_analysis_core::{
    ActivationLifecycle, AnalysisSequence, AngularVelocity, CoordinateFrame, Displacement,
    InitialCondition, InitialConditionTarget, InitialConditionValue, InitialFieldArtifactRef,
    InitialValue, PhysicalSignal, ProcedureStep, ProcedureStepId, Signal,
    SignalApproximationEvidence, SignalExtrapolation, SignalId, SignalInterpolation,
    SignalPrecision, SignalProvenance, SignalRepresentation, SignalSampleApproach, SignalShape,
    SignalValue, StepObjectActivation, StepObjectRef, TabulatedSignal, TabulatedSignalDomain,
    Temperature, Velocity,
};
use runmat_execution::Digest;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

use crate::{
    assembly::dofs::{StructuralDofKind, StructuralDofLayout, StructuralNodeDofSet},
    fixtures::{fixture_model, FixtureId},
};

use super::{
    realize as realize_with_constraints, StructuralInitialConstraintView,
    StructuralInitialStateError, StructuralInitialStateEvidenceBasis,
    StructuralInitialStateRealizationBasis,
};

const STEP_ID: &str = "transient";

fn realize(
    model: &runmat_analysis_core::AnalysisModel,
    mesh: Option<&runmat_meshing_core::SolverMeshArtifact>,
    layout: &StructuralDofLayout,
    step_id: &str,
) -> Result<super::RealizedStructuralInitialState, StructuralInitialStateError> {
    realize_with_constraints(
        model,
        mesh,
        layout,
        StructuralInitialStateRealizationBasis::CanonicalTopology,
        StructuralInitialConstraintView::unconstrained(),
        step_id,
    )
}

fn beam_model() -> runmat_analysis_core::AnalysisModel {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    model.sequence = AnalysisSequence {
        sequence_id: "initial-state-test".into(),
        steps: vec![ProcedureStep::structural_transient(STEP_ID)],
        dependencies: Vec::new(),
        transitions: Vec::new(),
        branches: Vec::new(),
    };
    model.initial_conditions.clear();
    model
}

fn beam_layout() -> StructuralDofLayout {
    StructuralDofLayout::from_node_sets(vec![
        StructuralNodeDofSet::translational_rotational(),
        StructuralNodeDofSet::translational_rotational(),
    ])
    .with_reference_points(1)
}

fn displacement(id: &str, target: InitialConditionTarget, value: [f64; 3]) -> InitialCondition {
    InitialCondition::Displacement(InitialConditionValue {
        initial_condition_id: id.into(),
        target,
        frame: CoordinateFrame::Global,
        value: InitialValue::UniformVector3(value.map(Displacement::new)),
        applies_before_step_id: STEP_ID.into(),
    })
}

fn velocity(id: &str, target: InitialConditionTarget, value: [f64; 3]) -> InitialCondition {
    InitialCondition::Velocity(InitialConditionValue {
        initial_condition_id: id.into(),
        target,
        frame: CoordinateFrame::Global,
        value: InitialValue::UniformVector3(value.map(Velocity::new)),
        applies_before_step_id: STEP_ID.into(),
    })
}

fn angular_velocity(id: &str, target: InitialConditionTarget, value: [f64; 3]) -> InitialCondition {
    InitialCondition::AngularVelocity(InitialConditionValue {
        initial_condition_id: id.into(),
        target,
        frame: CoordinateFrame::Global,
        value: InitialValue::UniformVector3(value.map(AngularVelocity::new)),
        applies_before_step_id: STEP_ID.into(),
    })
}

#[test]
fn implicit_synthetic_basis_realizes_only_an_evidenced_zero_state() {
    let mut model = beam_model();
    let mut inactive = velocity(
        "other-step",
        InitialConditionTarget::WholeModel,
        [1.0, 2.0, 3.0],
    );
    let InitialCondition::Velocity(condition) = &mut inactive else {
        unreachable!()
    };
    condition.applies_before_step_id = "later".into();
    model.initial_conditions = vec![inactive];
    let layout = StructuralDofLayout::reduced_translational_rows(6);
    let operator_identity = Digest::sha256(b"synthetic-operator");

    let state = realize_with_constraints(
        &model,
        None,
        &layout,
        StructuralInitialStateRealizationBasis::OperatorBoundZeroDefault {
            dof_count: 6,
            operator_identity,
        },
        StructuralInitialConstraintView::unconstrained(),
        STEP_ID,
    )
    .expect("an explicitly admitted operator-bound state may default to zero");

    assert_eq!(state.displacement, vec![0.0; 6]);
    assert_eq!(state.velocity, vec![0.0; 6]);
    assert_eq!(
        state.evidence.basis,
        StructuralInitialStateEvidenceBasis::OperatorBoundZeroDefault {
            dof_count: 6,
            operator_identity,
        }
    );
    assert!(state.evidence.applied_conditions.is_empty());
}

#[test]
fn implicit_synthetic_basis_rejects_authored_state_and_dof_mismatch() {
    let mut model = beam_model();
    model.initial_conditions = vec![velocity(
        "authored",
        InitialConditionTarget::WholeModel,
        [0.0, 0.0, 0.0],
    )];
    let layout = StructuralDofLayout::reduced_translational_rows(6);
    let operator_identity = Digest::sha256(b"synthetic-operator");
    let basis = StructuralInitialStateRealizationBasis::OperatorBoundZeroDefault {
        dof_count: 6,
        operator_identity,
    };

    assert!(matches!(
        realize_with_constraints(
            &model,
            None,
            &layout,
            basis,
            StructuralInitialConstraintView::unconstrained(),
            STEP_ID,
        ),
        Err(StructuralInitialStateError::OperatorBoundInitialConditionRequiresTopology {
            initial_condition_id,
        }) if initial_condition_id == "authored"
    ));

    model.initial_conditions.clear();
    assert!(matches!(
        realize_with_constraints(
            &model,
            None,
            &layout,
            StructuralInitialStateRealizationBasis::OperatorBoundZeroDefault {
                dof_count: 5,
                operator_identity,
            },
            StructuralInitialConstraintView::unconstrained(),
            STEP_ID,
        ),
        Err(StructuralInitialStateError::OperatorBoundDofCountMismatch {
            supplied_dof_count: 5,
            layout_dof_count: 6,
        })
    ));
}

#[test]
fn implicit_synthetic_basis_rejects_a_nonzero_constraint_default() {
    let model = beam_model();
    let layout = StructuralDofLayout::reduced_translational_rows(6);
    let mut constrained = vec![false; 6];
    constrained[0] = true;
    let mut prescribed = vec![0.0; 6];
    prescribed[0] = 0.5;

    assert!(matches!(
        realize_with_constraints(
            &model,
            None,
            &layout,
            StructuralInitialStateRealizationBasis::OperatorBoundZeroDefault {
                dof_count: 6,
                operator_identity: Digest::sha256(b"synthetic-operator"),
            },
            StructuralInitialConstraintView {
                displacement_constraints: None,
                velocity_constraints: None,
                affine_reduction: None,
                direct_constrained_dofs: &constrained,
                direct_prescribed_displacement: &prescribed,
            },
            STEP_ID,
        ),
        Err(StructuralInitialStateError::OperatorBoundZeroStateConstraintConflict)
    ));
}

#[test]
fn whole_model_maps_translation_and_rotation_rows_and_leaves_auxiliary_rows_zero() {
    let mut model = beam_model();
    model.initial_conditions = vec![
        displacement("u0", InitialConditionTarget::WholeModel, [1.0, 2.0, 3.0]),
        angular_velocity(
            "omega0",
            InitialConditionTarget::WholeModel,
            [4.0, 5.0, 6.0],
        ),
    ];
    let state = realize(&model, None, &beam_layout(), STEP_ID).expect("initial state");

    assert_eq!(
        state.displacement,
        vec![
            1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0
        ]
    );
    assert_eq!(
        state.velocity,
        vec![
            0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0
        ]
    );
    assert_eq!(state.evidence.authored_displacement_dof_count, 6);
    assert_eq!(state.evidence.authored_velocity_dof_count, 6);
    assert_eq!(state.evidence.applied_conditions.len(), 2);
    assert!(matches!(
        state.evidence.basis,
        StructuralInitialStateEvidenceBasis::CanonicalTopology { .. }
    ));

    let encoded = serde_json::to_value(&state.evidence).expect("serialize initial-state evidence");
    let decoded: crate::contracts::StructuralInitialStateEvidence =
        serde_json::from_value(encoded.clone()).expect("round-trip valid initial-state evidence");
    assert_eq!(decoded, state.evidence);

    let mut duplicate = encoded.clone();
    let conditions = duplicate["applied_conditions"]
        .as_array_mut()
        .expect("applied conditions array");
    conditions[1]["initial_condition_id"] = conditions[0]["initial_condition_id"].clone();
    assert!(
        serde_json::from_value::<crate::contracts::StructuralInitialStateEvidence>(duplicate)
            .is_err()
    );

    let mut out_of_order = encoded.clone();
    out_of_order["applied_conditions"]
        .as_array_mut()
        .expect("applied conditions array")
        .reverse();
    assert!(
        serde_json::from_value::<crate::contracts::StructuralInitialStateEvidence>(out_of_order)
            .is_err()
    );

    let mut negative_residual = encoded;
    negative_residual["maximum_displacement_constraint_residual_bits"] =
        serde_json::json!((-1.0_f64).to_bits());
    assert!(
        serde_json::from_value::<crate::contracts::StructuralInitialStateEvidence>(
            negative_residual
        )
        .is_err()
    );
}

#[test]
fn explicit_region_uses_exact_canonical_node_mapping() {
    let mut model = beam_model();
    model.initial_conditions = vec![velocity(
        "tip-v",
        InitialConditionTarget::Region(runmat_analysis_core::RegionTarget::point("node:2")),
        [7.0, 8.0, 9.0],
    )];
    let state = realize(&model, None, &beam_layout(), STEP_ID).expect("initial state");

    assert_eq!(&state.velocity[0..6], &[0.0; 6]);
    assert_eq!(&state.velocity[6..9], &[7.0, 8.0, 9.0]);
    assert_eq!(
        state.evidence.applied_conditions[0]
            .resolved_node_identities
            .len(),
        1
    );
}

#[test]
fn applies_before_and_step_deactivation_are_both_honored() {
    let mut model = beam_model();
    let mut wrong_step = displacement(
        "wrong-step",
        InitialConditionTarget::WholeModel,
        [1.0, 1.0, 1.0],
    );
    let InitialCondition::Displacement(condition) = &mut wrong_step else {
        unreachable!()
    };
    condition.applies_before_step_id = "later".into();
    model.initial_conditions = vec![
        wrong_step,
        velocity(
            "disabled",
            InitialConditionTarget::WholeModel,
            [2.0, 2.0, 2.0],
        ),
    ];
    model.sequence.steps[0].activations = vec![StepObjectActivation {
        object: StepObjectRef::InitialCondition("disabled".into()),
        lifecycle: ActivationLifecycle::Deactivate {
            from_step: ProcedureStepId("prior".into()),
        },
    }];

    let state = realize(&model, None, &beam_layout(), STEP_ID).expect("zero default state");
    assert!(state.displacement.iter().all(|value| *value == 0.0));
    assert!(state.velocity.iter().all(|value| *value == 0.0));
    assert!(state.evidence.applied_conditions.is_empty());
}

#[test]
fn identical_overlap_is_allowed_but_conflicting_overlap_is_rejected() {
    let mut model = beam_model();
    model.initial_conditions = vec![
        displacement("first", InitialConditionTarget::WholeModel, [1.0, 2.0, 3.0]),
        displacement(
            "same-tip",
            InitialConditionTarget::Region(runmat_analysis_core::RegionTarget::point("node:2")),
            [1.0, 2.0, 3.0],
        ),
    ];
    realize(&model, None, &beam_layout(), STEP_ID).expect("identical overlap");

    model.initial_conditions.push(displacement(
        "conflict",
        InitialConditionTarget::Region(runmat_analysis_core::RegionTarget::point("node:2")),
        [9.0, 2.0, 3.0],
    ));
    assert!(matches!(
        realize(&model, None, &beam_layout(), STEP_ID),
        Err(StructuralInitialStateError::ConflictingValue {
            first_initial_condition_id,
            second_initial_condition_id,
            row: 6,
        }) if first_initial_condition_id == "first" && second_initial_condition_id == "conflict"
    ));
}

#[test]
fn samples_typed_vector_signal_and_rejects_unmaterialized_fields() {
    let mut model = beam_model();
    let signal_id = SignalId("initial-velocity".into());
    model.signals.push(PhysicalSignal::Velocity(Signal {
        signal_id: signal_id.clone(),
        shape: SignalShape::Vector3,
        frame: Some(CoordinateFrame::Global),
        representation: SignalRepresentation::Constant(SignalValue::Vector3(
            [1.0, 2.0, 3.0].map(Velocity::new),
        )),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    }));
    model.initial_conditions = vec![InitialCondition::Velocity(InitialConditionValue {
        initial_condition_id: "sampled".into(),
        target: InitialConditionTarget::WholeModel,
        frame: CoordinateFrame::Global,
        value: InitialValue::SignalSample {
            signal_id,
            abscissa: 0.0,
            approach: SignalSampleApproach::At,
        },
        applies_before_step_id: STEP_ID.into(),
    })];
    let state = realize(&model, None, &beam_layout(), STEP_ID).expect("sampled state");
    assert_eq!(&state.velocity[0..3], &[1.0, 2.0, 3.0]);

    let encoded = serde_json::to_value(&state.evidence).expect("serialize sampled evidence");
    for invalid_abscissa in [(-0.0_f64).to_bits(), f64::INFINITY.to_bits()] {
        let mut tampered = encoded.clone();
        tampered["applied_conditions"][0]["source"]["abscissa_bits"] =
            serde_json::json!(invalid_abscissa);
        assert!(
            serde_json::from_value::<crate::contracts::StructuralInitialStateEvidence>(tampered)
                .is_err()
        );
    }

    match &mut model.initial_conditions[0] {
        InitialCondition::Velocity(condition) => {
            condition.value = InitialValue::SignalSample {
                signal_id: SignalId("initial-velocity".into()),
                abscissa: -0.0,
                approach: SignalSampleApproach::At,
            };
        }
        _ => unreachable!(),
    }
    assert!(matches!(
        realize(&model, None, &beam_layout(), STEP_ID),
        Err(StructuralInitialStateError::InvalidRealizedValue { .. })
    ));
    match &mut model.initial_conditions[0] {
        InitialCondition::Velocity(condition) => {
            condition.value = InitialValue::SignalSample {
                signal_id: SignalId("initial-velocity".into()),
                abscissa: 0.0,
                approach: SignalSampleApproach::At,
            };
        }
        _ => unreachable!(),
    }

    match model.signals.last_mut().expect("velocity signal") {
        PhysicalSignal::Velocity(signal) => signal.frame = Some(CoordinateFrame::Material),
        _ => unreachable!(),
    }
    assert!(matches!(
        realize(&model, None, &beam_layout(), STEP_ID),
        Err(StructuralInitialStateError::UnsupportedSignalFrame { .. })
    ));
    match model.signals.last_mut().expect("velocity signal") {
        PhysicalSignal::Velocity(signal) => {
            signal.frame = Some(CoordinateFrame::Global);
            signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
                domain: TabulatedSignalDomain::FrequencyHertz,
                knots: Vec::new(),
                interpolation: SignalInterpolation::Linear,
                extrapolation: SignalExtrapolation::Reject,
            });
        }
        _ => unreachable!(),
    }
    assert!(matches!(
        realize(&model, None, &beam_layout(), STEP_ID),
        Err(StructuralInitialStateError::UnsupportedSignalDomain { .. })
    ));

    model.initial_conditions = vec![InitialCondition::Displacement(InitialConditionValue {
        initial_condition_id: "field".into(),
        target: InitialConditionTarget::WholeModel,
        frame: CoordinateFrame::Global,
        value: InitialValue::FieldArtifact(InitialFieldArtifactRef {
            artifact_id: "artifact".into(),
            content_digest: "content".into(),
            topology_digest: "topology".into(),
            field_id: "u".into(),
        }),
        applies_before_step_id: STEP_ID.into(),
    })];
    assert!(matches!(
        realize(&model, None, &beam_layout(), STEP_ID),
        Err(StructuralInitialStateError::UnmaterializedFieldArtifact { .. })
    ));
}

#[test]
fn solid_translation_layout_rejects_angular_velocity_and_incompatible_initial_conditions() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.sequence.steps = vec![ProcedureStep::structural_transient(STEP_ID)];
    model.structural = None;
    model.initial_conditions = vec![angular_velocity(
        "omega",
        InitialConditionTarget::WholeModel,
        [1.0, 2.0, 3.0],
    )];
    let layout = StructuralDofLayout::from_node_sets(vec![
        StructuralNodeDofSet::translational();
        mesh.topology.nodes.len()
    ]);
    assert!(matches!(
        realize(&model, Some(&mesh), &layout, STEP_ID),
        Err(StructuralInitialStateError::MissingDof {
            dof: StructuralDofKind::Rx,
            ..
        })
    ));

    model.initial_conditions = vec![InitialCondition::Temperature(InitialConditionValue {
        initial_condition_id: "temperature".into(),
        target: InitialConditionTarget::WholeModel,
        frame: CoordinateFrame::Global,
        value: InitialValue::Uniform(Temperature::new(300.0)),
        applies_before_step_id: STEP_ID.into(),
    })];
    assert!(matches!(
        realize(&model, Some(&mesh), &layout, STEP_ID),
        Err(StructuralInitialStateError::UnsupportedInitialCondition { .. })
    ));
}

#[test]
fn rejects_non_global_frames_scalar_values_and_unmapped_regions() {
    let mut model = beam_model();
    let mut framed = displacement(
        "framed",
        InitialConditionTarget::WholeModel,
        [1.0, 2.0, 3.0],
    );
    let InitialCondition::Displacement(condition) = &mut framed else {
        unreachable!()
    };
    condition.frame = CoordinateFrame::Material;
    model.initial_conditions = vec![framed];
    assert!(matches!(
        realize(&model, None, &beam_layout(), STEP_ID),
        Err(StructuralInitialStateError::UnsupportedFrame { .. })
    ));

    model.initial_conditions = vec![InitialCondition::Displacement(InitialConditionValue {
        initial_condition_id: "scalar".into(),
        target: InitialConditionTarget::WholeModel,
        frame: CoordinateFrame::Global,
        value: InitialValue::Uniform(Displacement::new(1.0)),
        applies_before_step_id: STEP_ID.into(),
    })];
    assert!(matches!(
        realize(&model, None, &beam_layout(), STEP_ID),
        Err(StructuralInitialStateError::InvalidValueShape { .. })
    ));

    model.initial_conditions = vec![velocity(
        "unmapped",
        InitialConditionTarget::Region(runmat_analysis_core::RegionTarget::point("node:999")),
        [1.0, 2.0, 3.0],
    )];
    assert!(matches!(
        realize(&model, None, &beam_layout(), STEP_ID),
        Err(StructuralInitialStateError::InvalidTarget { .. })
    ));
}

#[test]
fn signal_references_fail_closed_for_missing_and_incompatible_quantities() {
    let mut model = beam_model();
    model.initial_conditions = vec![InitialCondition::Velocity(InitialConditionValue {
        initial_condition_id: "missing".into(),
        target: InitialConditionTarget::WholeModel,
        frame: CoordinateFrame::Global,
        value: InitialValue::SignalSample {
            signal_id: SignalId("does-not-exist".into()),
            abscissa: 0.0,
            approach: SignalSampleApproach::At,
        },
        applies_before_step_id: STEP_ID.into(),
    })];
    assert!(matches!(
        realize(&model, None, &beam_layout(), STEP_ID),
        Err(StructuralInitialStateError::MissingSignal { .. })
    ));

    let force_signal_id = model.signals[0].signal_id().clone();
    let InitialCondition::Velocity(condition) = &mut model.initial_conditions[0] else {
        unreachable!()
    };
    let InitialValue::SignalSample { signal_id, .. } = &mut condition.value else {
        unreachable!()
    };
    *signal_id = force_signal_id;
    assert!(matches!(
        realize(&model, None, &beam_layout(), STEP_ID),
        Err(StructuralInitialStateError::IncompatibleSignal { .. })
    ));
}
mod constraints;

mod fields;
