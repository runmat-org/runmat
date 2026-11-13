use crate::fixtures::{
    fixture_model, replace_fixture_excitations, FixtureExcitation, FixtureExcitationKind, FixtureId,
};

mod acoustic_incident_pressure;
mod charge_density;
mod magnetic_volume_current;
mod prescribed_fluid;
mod volume_velocity;

use super::{
    realize::{realize_excitations, realize_step_excitations, RealizationError},
    sample::ExcitationSamplePoint,
    RealizedExcitationKind, RealizedExcitations,
};

fn realize_static(
    model: &runmat_analysis_core::AnalysisModel,
) -> Result<RealizedExcitations, RealizationError> {
    realize_step_excitations(
        model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::Static,
    )
}

fn tabulated_force_model(
    domain: runmat_analysis_core::TabulatedSignalDomain,
    axis: Vec<f64>,
) -> runmat_analysis_core::AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let runmat_analysis_core::PhysicalSignal::Force(signal) = &mut model.signals[0] else {
        panic!("cantilever fixture should carry a force signal");
    };
    signal.representation = runmat_analysis_core::SignalRepresentation::Tabulated(
        runmat_analysis_core::TabulatedSignal {
            domain,
            knots: axis
                .into_iter()
                .zip([[0.0, 0.0, 0.0], [20.0, 40.0, 60.0]])
                .map(|(abscissa, value)| {
                    runmat_analysis_core::SignalKnot::continuous(
                        abscissa,
                        runmat_analysis_core::SignalOrdinate::Real(
                            runmat_analysis_core::SignalValue::Vector3(
                                value.map(runmat_analysis_core::Force::new),
                            ),
                        ),
                    )
                })
                .collect(),
            interpolation: runmat_analysis_core::SignalInterpolation::Linear,
            extrapolation: runmat_analysis_core::SignalExtrapolation::Reject,
        },
    );
    signal.evidence.sample_count = 2;
    model
}

fn set_table_policy(
    model: &mut runmat_analysis_core::AnalysisModel,
    interpolation: runmat_analysis_core::SignalInterpolation,
    extrapolation: runmat_analysis_core::SignalExtrapolation,
) {
    let runmat_analysis_core::PhysicalSignal::Force(signal) = &mut model.signals[0] else {
        panic!("cantilever fixture should carry a force signal");
    };
    let runmat_analysis_core::SignalRepresentation::Tabulated(table) = &mut signal.representation
    else {
        panic!("test signal should be tabulated");
    };
    table.interpolation = interpolation;
    table.extrapolation = extrapolation;
}

fn force_ordinate(x: f64) -> runmat_analysis_core::SignalOrdinate<runmat_analysis_core::Force> {
    runmat_analysis_core::SignalOrdinate::Real(runmat_analysis_core::SignalValue::Vector3([
        runmat_analysis_core::Force::new(x),
        runmat_analysis_core::Force::new(0.0),
        runmat_analysis_core::Force::new(0.0),
    ]))
}

fn jump_force_model(
    continuity: runmat_analysis_core::SignalPointContinuity,
) -> runmat_analysis_core::AnalysisModel {
    let mut model = tabulated_force_model(
        runmat_analysis_core::TabulatedSignalDomain::TimeSeconds,
        vec![0.0, 1.0],
    );
    let runmat_analysis_core::PhysicalSignal::Force(signal) = &mut model.signals[0] else {
        unreachable!()
    };
    let runmat_analysis_core::SignalRepresentation::Tabulated(table) = &mut signal.representation
    else {
        unreachable!()
    };
    table.knots = vec![
        runmat_analysis_core::SignalKnot::continuous(0.0, force_ordinate(0.0)),
        runmat_analysis_core::SignalKnot::jump(
            0.5,
            force_ordinate(10.0),
            force_ordinate(30.0),
            continuity,
        ),
        runmat_analysis_core::SignalKnot::continuous(1.0, force_ordinate(40.0)),
    ];
    signal.evidence.sample_count = 3;
    model
}

fn force_x(model: &runmat_analysis_core::AnalysisModel, sample: ExcitationSamplePoint<'_>) -> f64 {
    let step = &model.sequence.steps[0].step_id.0;
    match &realize_excitations(model, Some(step), sample)
        .unwrap()
        .iter()
        .next()
        .unwrap()
        .kind
    {
        RealizedExcitationKind::Force(value) => value.real[0],
        _ => unreachable!(),
    }
}

#[test]
fn realizes_typed_force_signal() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let realized = realize_static(&model).expect("typed force should realize");
    let excitation = realized.iter().next().expect("one excitation");

    assert_eq!(excitation.excitation_id.0, "tip_load");
    assert!(matches!(
        excitation.kind,
        RealizedExcitationKind::Force(value)
            if value.real == [0.0, -1000.0, 0.0] && value.imaginary == [0.0; 3]
    ));
}

// witness: FEA-SHARED-EXCITATION-001/conservative_spatial_realization @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Realization preserves exact multi-entity region targets.
#[test]
fn realization_preserves_exact_multi_entity_region_targets() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let runmat_analysis_core::ExcitationTarget::Region(target) =
        &mut model.excitations[0].excitation_mut().target
    else {
        panic!("fixture force must be region-scoped");
    };
    let entities = ["face:0", "face:1"]
        .map(|source_topology_id| {
            runmat_geometry_core::PersistentEntityId::new(
                runmat_geometry_core::PersistentEntityKind::Face,
                source_topology_id.into(),
                vec!["root".into()],
            )
        })
        .to_vec();
    target.bind_resolved_entities(entities.clone()).unwrap();

    let realized = realize_static(&model).expect("typed force should realize");
    let super::RealizedExcitationTarget::Region(realized_target) =
        &realized.iter().next().unwrap().target
    else {
        panic!("realized force must remain region-scoped");
    };
    assert_eq!(realized_target.resolved_entities(), entities);
}

#[test]
fn realizes_one_action_with_every_active_case_membership() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut second_case = model.excitation_cases[0].clone();
    second_case.case_id = runmat_analysis_core::ExcitationCaseId("proof_case".into());
    model.excitation_cases.push(second_case);

    let realized = realize_static(&model).expect("shared action should realize once");
    let entries = realized.iter().collect::<Vec<_>>();

    assert_eq!(entries.len(), 1);
    assert_eq!(
        entries[0].case_ids,
        [
            runmat_analysis_core::ExcitationCaseId("tip_load.case".into()),
            runmat_analysis_core::ExcitationCaseId("proof_case".into()),
        ]
    );
}

#[test]
fn static_realization_does_not_implicitly_take_the_first_time_sample() {
    let model = tabulated_force_model(
        runmat_analysis_core::TabulatedSignalDomain::TimeSeconds,
        vec![0.0, 1.0],
    );

    let error = realize_static(&model).unwrap_err();

    assert_eq!(
        error.to_string(),
        "static excitation requires constant tabulated values"
    );
}

#[test]
fn mutually_exclusive_cases_require_an_explicit_selection() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitation_cases[0].activation =
        runmat_analysis_core::ExcitationCaseActivation::MutuallyExclusiveGroup {
            group_id: "load_variant".into(),
        };

    let error = realize_static(&model).unwrap_err();

    assert!(error.to_string().contains("no case selection was supplied"));
}

#[test]
fn malformed_tabulated_shape_fails_before_ordinate_indexing() {
    let model = tabulated_force_model(
        runmat_analysis_core::TabulatedSignalDomain::TimeSeconds,
        vec![0.0, 0.0],
    );
    let step = model.sequence.steps[0].step_id.clone();

    let error =
        realize_excitations(&model, Some(&step.0), ExcitationSamplePoint::time(0.25)).unwrap_err();

    assert_eq!(
        error.to_string(),
        "tabulated knots must have finite, strictly increasing abscissae"
    );
}

#[test]
fn explicit_time_and_parameter_samples_use_the_matching_domain() {
    let time_model = tabulated_force_model(
        runmat_analysis_core::TabulatedSignalDomain::TimeSeconds,
        vec![0.0, 1.0],
    );
    let time_step = time_model.sequence.steps[0].step_id.clone();
    let time = realize_excitations(
        &time_model,
        Some(&time_step.0),
        ExcitationSamplePoint::time(0.25),
    )
    .unwrap();
    assert!(matches!(
        time.iter().next().unwrap().kind,
        RealizedExcitationKind::Force(value) if value.real == [5.0, 10.0, 15.0]
    ));

    let parameter_model = tabulated_force_model(
        runmat_analysis_core::TabulatedSignalDomain::StudyParameter {
            parameter_id: "load_scale".into(),
            quantity: Some(runmat_types::PhysicalDimension::DIMENSIONLESS.into()),
        },
        vec![0.0, 1.0],
    );
    let parameter_step = parameter_model.sequence.steps[0].step_id.clone();
    let parameter = realize_excitations(
        &parameter_model,
        Some(&parameter_step.0),
        ExcitationSamplePoint::StudyParameter {
            parameter_id: "load_scale",
            canonical_value: 0.5,
            approach: runmat_analysis_core::SignalSampleApproach::At,
        },
    )
    .unwrap();
    assert!(matches!(
        parameter.iter().next().unwrap().kind,
        RealizedExcitationKind::Force(value) if value.real == [10.0, 20.0, 30.0]
    ));
}

// witness: FEA-SHARED-EXCITATION-001/governed_excitation_corpus @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Static excitation accepts only coordinate-independent tables; the constant, step and
//   frequency-dependent cases of the corpus are not all covered.
#[test]
fn static_excitation_accepts_only_coordinate_independent_tables() {
    for domain in [
        runmat_analysis_core::TabulatedSignalDomain::TimeSeconds,
        runmat_analysis_core::TabulatedSignalDomain::StudyParameter {
            parameter_id: "load_scale".into(),
            quantity: Some(runmat_types::PhysicalDimension::DIMENSIONLESS.into()),
        },
    ] {
        let mut model = tabulated_force_model(domain, vec![0.0, 1.0]);
        assert_eq!(
            realize_static(&model).unwrap_err().to_string(),
            "static excitation requires constant tabulated values"
        );
        let runmat_analysis_core::PhysicalSignal::Force(signal) = &mut model.signals[0] else {
            unreachable!()
        };
        let runmat_analysis_core::SignalRepresentation::Tabulated(table) =
            &mut signal.representation
        else {
            unreachable!()
        };
        table.knots[0].value = table.knots[1].value.clone();
        let realized = realize_static(&model).unwrap();
        assert!(matches!(
            realized.iter().next().unwrap().kind,
            RealizedExcitationKind::Force(value) if value.real == [20.0, 40.0, 60.0]
        ));
    }
    let jump = jump_force_model(runmat_analysis_core::SignalPointContinuity::Right);
    assert_eq!(
        realize_static(&jump).unwrap_err().to_string(),
        "static excitation cannot select a jump in a tabulated signal"
    );
}

// witness: FEA-SHARED-EXCITATION-001/dimensioned_signal_contract @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Typed tabulated force signal at time 0.5 selects distinct left and right endpoint
//   vectors; no generator freezing, spatial integration or placement equivalence.
#[test]
fn step_interpolation_selects_the_documented_side_of_an_interval() {
    let mut model = tabulated_force_model(
        runmat_analysis_core::TabulatedSignalDomain::TimeSeconds,
        vec![0.0, 1.0],
    );
    let step = model.sequence.steps[0].step_id.clone();
    set_table_policy(
        &mut model,
        runmat_analysis_core::SignalInterpolation::LeftEndpoint,
        runmat_analysis_core::SignalExtrapolation::Reject,
    );
    let left =
        realize_excitations(&model, Some(&step.0), ExcitationSamplePoint::time(0.5)).unwrap();
    assert!(matches!(
        left.iter().next().unwrap().kind,
        RealizedExcitationKind::Force(value) if value.real == [0.0; 3]
    ));

    set_table_policy(
        &mut model,
        runmat_analysis_core::SignalInterpolation::RightEndpoint,
        runmat_analysis_core::SignalExtrapolation::Reject,
    );
    let right =
        realize_excitations(&model, Some(&step.0), ExcitationSamplePoint::time(0.5)).unwrap();
    assert!(matches!(
        right.iter().next().unwrap().kind,
        RealizedExcitationKind::Force(value) if value.real == [20.0, 40.0, 60.0]
    ));
}

#[test]
fn jump_knots_have_distinct_point_and_one_sided_values() {
    let left = jump_force_model(runmat_analysis_core::SignalPointContinuity::Left);
    assert_eq!(force_x(&left, ExcitationSamplePoint::time(0.5)), 10.0);
    assert_eq!(
        force_x(
            &left,
            ExcitationSamplePoint::TimeSeconds {
                value: 0.5,
                approach: runmat_analysis_core::SignalSampleApproach::LeftLimit,
            },
        ),
        10.0
    );
    assert_eq!(
        force_x(
            &left,
            ExcitationSamplePoint::TimeSeconds {
                value: 0.5,
                approach: runmat_analysis_core::SignalSampleApproach::RightLimit,
            },
        ),
        30.0
    );

    let right = jump_force_model(runmat_analysis_core::SignalPointContinuity::Right);
    assert_eq!(force_x(&right, ExcitationSamplePoint::time(0.5)), 30.0);
    assert_eq!(force_x(&right, ExcitationSamplePoint::time(0.25)), 5.0);
    assert_eq!(force_x(&right, ExcitationSamplePoint::time(0.75)), 35.0);
}

#[test]
fn one_sided_sampling_rejects_missing_endpoint_limits() {
    let model = jump_force_model(runmat_analysis_core::SignalPointContinuity::Right);
    let step = &model.sequence.steps[0].step_id.0;
    let error = realize_excitations(
        &model,
        Some(step),
        ExcitationSamplePoint::TimeSeconds {
            value: 0.0,
            approach: runmat_analysis_core::SignalSampleApproach::LeftLimit,
        },
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("does not exist at the domain endpoint"));
}

#[test]
fn event_inventory_ignores_unreferenced_signal_jumps() {
    let mut model = jump_force_model(runmat_analysis_core::SignalPointContinuity::Right);
    model.sequence.steps[0].activations = vec![runmat_analysis_core::StepObjectActivation {
        object: runmat_analysis_core::StepObjectRef::Excitation("tip_load".into()),
        lifecycle: runmat_analysis_core::ActivationLifecycle::Activate,
    }];
    let mut unused = model.signals[0].clone();
    let runmat_analysis_core::PhysicalSignal::Force(signal) = &mut unused else {
        unreachable!()
    };
    signal.signal_id = runmat_analysis_core::SignalId("unused.force".into());
    let runmat_analysis_core::SignalRepresentation::Tabulated(table) = &mut signal.representation
    else {
        unreachable!()
    };
    table.knots[1].abscissa = 0.25;
    model.signals.push(unused);

    let displacement_id = runmat_analysis_core::SignalId("inactive.displacement".into());
    let displacement = |value: f64| {
        runmat_analysis_core::SignalOrdinate::Real(runmat_analysis_core::SignalValue::Scalar(
            runmat_analysis_core::Displacement::new(value),
        ))
    };
    model
        .signals
        .push(runmat_analysis_core::PhysicalSignal::Displacement(
            runmat_analysis_core::Signal {
                signal_id: displacement_id.clone(),
                shape: runmat_analysis_core::SignalShape::Scalar,
                frame: None,
                representation: runmat_analysis_core::SignalRepresentation::Tabulated(
                    runmat_analysis_core::TabulatedSignal {
                        domain: runmat_analysis_core::TabulatedSignalDomain::TimeSeconds,
                        knots: vec![
                            runmat_analysis_core::SignalKnot::continuous(0.0, displacement(0.0)),
                            runmat_analysis_core::SignalKnot::jump(
                                0.25,
                                displacement(0.0),
                                displacement(1.0),
                                runmat_analysis_core::SignalPointContinuity::Right,
                            ),
                            runmat_analysis_core::SignalKnot::continuous(1.0, displacement(1.0)),
                        ],
                        interpolation: runmat_analysis_core::SignalInterpolation::LeftEndpoint,
                        extrapolation: runmat_analysis_core::SignalExtrapolation::Reject,
                    },
                ),
                precision: runmat_analysis_core::SignalPrecision::Real64,
                evidence: runmat_analysis_core::SignalApproximationEvidence {
                    max_absolute_error: Some(0.0),
                    max_relative_error: Some(0.0),
                    derivative_max_absolute_error: None,
                    sample_count: 3,
                },
                provenance: runmat_analysis_core::SignalProvenance::Authored,
            },
        ));
    model
        .constraints
        .push(runmat_analysis_core::AnalysisConstraint::Structural(
            runmat_analysis_core::StructuralConstraint {
                constraint_id: "inactive_constraint".into(),
                frame: runmat_analysis_core::CoordinateFrame::Global,
                kind: runmat_analysis_core::StructuralConstraintKind::NormalDisplacement {
                    target: runmat_analysis_core::RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_face("fixed"),
                    ),
                    value: runmat_analysis_core::PrescribedValue::Signal(displacement_id),
                },
            },
        ));

    let events = super::time_breakpoints(&model, &model.sequence.steps[0], 0.0, 1.0).unwrap();
    assert_eq!(events, vec![0.5]);
}

#[test]
fn prescribed_motion_only_marks_actual_acceleration_jumps_as_discontinuities() {
    let mut model = fixture_model(FixtureId::StructuralTransientTet4SupportMotionReference);
    let step = &model.sequence.steps[0];
    assert!(super::time_breakpoints(&model, step, 0.0, 1.0)
        .unwrap()
        .is_empty());

    let runmat_analysis_core::PhysicalSignal::Acceleration(signal) = &mut model.signals[0] else {
        panic!("support-motion fixture should carry an acceleration signal");
    };
    let runmat_analysis_core::SignalRepresentation::Tabulated(table) = &mut signal.representation
    else {
        panic!("support-motion acceleration should be tabulated");
    };
    let runmat_analysis_core::SignalKnotValue::Continuous(left_limit) =
        table.knots[1].value.clone()
    else {
        panic!("support-motion midpoint should initially be continuous");
    };
    let right_limit =
        runmat_analysis_core::SignalOrdinate::Real(runmat_analysis_core::SignalValue::Vector3([
            runmat_analysis_core::Acceleration::new(2.0),
            runmat_analysis_core::Acceleration::new(0.0),
            runmat_analysis_core::Acceleration::new(0.0),
        ]));
    table.knots[1] = runmat_analysis_core::SignalKnot::jump(
        0.5,
        left_limit,
        right_limit,
        runmat_analysis_core::SignalPointContinuity::Right,
    );

    let events = super::time_breakpoints(&model, &model.sequence.steps[0], 0.0, 1.0).unwrap();
    assert_eq!(events, vec![0.5]);
}

#[test]
fn extrapolation_policy_is_applied_without_numeric_sentinels() {
    let mut model = tabulated_force_model(
        runmat_analysis_core::TabulatedSignalDomain::TimeSeconds,
        vec![0.0, 1.0],
    );
    let step = model.sequence.steps[0].step_id.clone();
    set_table_policy(
        &mut model,
        runmat_analysis_core::SignalInterpolation::Linear,
        runmat_analysis_core::SignalExtrapolation::HoldEndpoint,
    );
    let held =
        realize_excitations(&model, Some(&step.0), ExcitationSamplePoint::time(2.0)).unwrap();
    assert!(matches!(
        held.iter().next().unwrap().kind,
        RealizedExcitationKind::Force(value) if value.real == [20.0, 40.0, 60.0]
    ));

    set_table_policy(
        &mut model,
        runmat_analysis_core::SignalInterpolation::Linear,
        runmat_analysis_core::SignalExtrapolation::Zero,
    );
    let zero =
        realize_excitations(&model, Some(&step.0), ExcitationSamplePoint::time(-1.0)).unwrap();
    assert!(matches!(
        zero.iter().next().unwrap().kind,
        RealizedExcitationKind::Force(value) if value.real == [0.0; 3]
    ));

    set_table_policy(
        &mut model,
        runmat_analysis_core::SignalInterpolation::Linear,
        runmat_analysis_core::SignalExtrapolation::Reject,
    );
    let error =
        realize_excitations(&model, Some(&step.0), ExcitationSamplePoint::time(-1.0)).unwrap_err();
    assert_eq!(error.to_string(), "sample lies outside signal domain");
}

#[test]
fn rejects_nonfinite_and_mismatched_parameter_samples() {
    let time_model = tabulated_force_model(
        runmat_analysis_core::TabulatedSignalDomain::TimeSeconds,
        vec![0.0, 1.0],
    );
    let step = time_model.sequence.steps[0].step_id.clone();
    let error = realize_excitations(
        &time_model,
        Some(&step.0),
        ExcitationSamplePoint::time(f64::NAN),
    )
    .unwrap_err();
    assert_eq!(error.to_string(), "excitation sample point must be finite");

    let parameter_model = tabulated_force_model(
        runmat_analysis_core::TabulatedSignalDomain::StudyParameter {
            parameter_id: "load_scale".into(),
            quantity: Some(runmat_types::PhysicalDimension::DIMENSIONLESS.into()),
        },
        vec![0.0, 1.0],
    );
    let step = parameter_model.sequence.steps[0].step_id.clone();
    let error = realize_excitations(
        &parameter_model,
        Some(&step.0),
        ExcitationSamplePoint::StudyParameter {
            parameter_id: "temperature_scale",
            canonical_value: 0.5,
            approach: runmat_analysis_core::SignalSampleApproach::At,
        },
    )
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "excitation sample point does not match the signal domain"
    );
}

#[test]
fn projects_scalar_force_along_its_typed_direction() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let runmat_analysis_core::PhysicalSignal::Force(signal) = &mut model.signals[0] else {
        panic!("cantilever fixture should carry a force signal");
    };
    signal.shape = runmat_analysis_core::SignalShape::Scalar;
    signal.representation = runmat_analysis_core::SignalRepresentation::Constant(
        runmat_analysis_core::SignalValue::Scalar(runmat_analysis_core::Force::new(250.0)),
    );
    model.excitations[0].excitation_mut().direction = Some(runmat_analysis_core::Direction {
        vector: [0.0, -1.0, 0.0],
    });

    let realized = realize_static(&model).expect("directional scalar should realize");

    assert!(matches!(
        realized.iter().next().expect("one force").kind,
        RealizedExcitationKind::Force(value)
            if value.real == [0.0, -250.0, 0.0] && value.imaginary == [0.0; 3]
    ));
}

#[test]
fn rejects_scalar_force_without_a_direction() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let runmat_analysis_core::PhysicalSignal::Force(signal) = &mut model.signals[0] else {
        panic!("cantilever fixture should carry a force signal");
    };
    signal.shape = runmat_analysis_core::SignalShape::Scalar;
    signal.representation = runmat_analysis_core::SignalRepresentation::Constant(
        runmat_analysis_core::SignalValue::Scalar(runmat_analysis_core::Force::new(250.0)),
    );
    model.excitations[0].excitation_mut().direction = None;

    let error = realize_static(&model).expect_err("direction should be required");

    assert_eq!(
        error.to_string(),
        "scalar directional excitation requires a direction"
    );
}

// witness: FEA-SHARED-EXCITATION-001/conservative_spatial_realization @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Reference-point force and moment action identity is preserved.
#[test]
fn preserves_reference_point_force_and_moment_action_identity() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "tip_wrench".into(),
            region_id: "tip".into(),
            kind: FixtureExcitationKind::Wrench {
                force: [1.0, 2.0, 3.0],
                moment: [4.0, 5.0, 6.0],
                point_m: [0.1, 0.2, 0.3],
            },
        }],
    );

    let realized = realize_static(&model).expect("wrench should realize");
    assert_eq!(realized.len(), 2);
    let entries = realized.iter().collect::<Vec<_>>();
    assert_eq!(entries[0].excitation_id.0, "tip_wrench");
    assert!(matches!(
        entries[0].kind,
        RealizedExcitationKind::Force(force) if force.real == [1.0, 2.0, 3.0]
    ));
    assert_eq!(entries[1].excitation_id.0, "tip_wrench.moment");
    assert!(matches!(
        entries[1].kind,
        RealizedExcitationKind::Moment(moment) if moment.real == [4.0, 5.0, 6.0]
    ));
    assert_eq!(entries[0].target, entries[1].target);
}

#[test]
fn preserves_complex_frequency_domain_current_density() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.electromagnetic = Some(runmat_analysis_core::ElectromagneticDomain::Harmonic {
        reference_frequency: runmat_analysis_core::Frequency::new(60.0),
    });
    let signal_id = runmat_analysis_core::SignalId("coil.current_density".into());
    model.signals = vec![
        runmat_analysis_core::PhysicalSignal::ElectricCurrentDensity(
            runmat_analysis_core::Signal {
                signal_id: signal_id.clone(),
                shape: runmat_analysis_core::SignalShape::Vector3,
                frame: Some(runmat_analysis_core::CoordinateFrame::Global),
                representation: runmat_analysis_core::SignalRepresentation::Tabulated(
                    runmat_analysis_core::TabulatedSignal {
                        domain: runmat_analysis_core::TabulatedSignalDomain::FrequencyHertz,
                        knots: vec![runmat_analysis_core::SignalKnot::continuous(
                            60.0,
                            runmat_analysis_core::SignalOrdinate::Complex {
                                real: runmat_analysis_core::SignalValue::Vector3(
                                    [1.0, 2.0, 3.0]
                                        .map(runmat_analysis_core::ElectricCurrentDensity::new),
                                ),
                                imaginary: runmat_analysis_core::SignalValue::Vector3(
                                    [4.0, 5.0, 6.0]
                                        .map(runmat_analysis_core::ElectricCurrentDensity::new),
                                ),
                            },
                        )],
                        interpolation: runmat_analysis_core::SignalInterpolation::Linear,
                        extrapolation: runmat_analysis_core::SignalExtrapolation::HoldEndpoint,
                    },
                ),
                precision: runmat_analysis_core::SignalPrecision::Complex64,
                evidence: runmat_analysis_core::SignalApproximationEvidence {
                    max_absolute_error: Some(0.0),
                    max_relative_error: Some(0.0),
                    derivative_max_absolute_error: None,
                    sample_count: 1,
                },
                provenance: runmat_analysis_core::SignalProvenance::Authored,
            },
        ),
    ];
    let mut action = model.excitations[0].excitation().clone();
    action.signal_id = signal_id;
    model.excitations = vec![runmat_analysis_core::ExcitationAction::CurrentDensity(
        action,
    )];

    let realized = realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::frequency(60.0),
    )
    .expect("complex current should realize");
    assert!(matches!(
        realized.iter().next().expect("one current").kind,
        RealizedExcitationKind::CurrentDensity(value)
            if value.real == [1.0, 2.0, 3.0] && value.imaginary == [4.0, 5.0, 6.0]
    ));
    assert!(matches!(
        realized
            .component(super::ComplexComponent::Imaginary)
            .iter()
            .next()
            .expect("one imaginary current")
            .kind,
        RealizedExcitationKind::CurrentDensity(value)
            if value.real == [4.0, 5.0, 6.0] && value.imaginary == [0.0; 3]
    ));
}
