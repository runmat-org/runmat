use crate::fixtures::{
    fixture_model, replace_fixture_excitations, FixtureExcitation, FixtureExcitationKind, FixtureId,
};

use super::{
    realize::{realize_excitations, ExcitationSamplePoint},
    realize_model_excitations, RealizedExcitationKind,
};

fn tabulated_force_model(
    axis: runmat_analysis_core::SignalAxis,
) -> runmat_analysis_core::AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let runmat_analysis_core::PhysicalSignal::Force(signal) = &mut model.signals[0] else {
        panic!("cantilever fixture should carry a force signal");
    };
    signal.representation = runmat_analysis_core::SignalRepresentation::Tabulated(
        runmat_analysis_core::TabulatedSignal {
            axis,
            values: [[0.0, 0.0, 0.0], [20.0, 40.0, 60.0]]
                .into_iter()
                .map(|value| {
                    runmat_analysis_core::SignalOrdinate::Real(
                        runmat_analysis_core::SignalValue::Vector3(
                            value.map(runmat_analysis_core::Force::new),
                        ),
                    )
                })
                .collect(),
            interpolation: runmat_analysis_core::SignalInterpolation::Linear,
            extrapolation: runmat_analysis_core::SignalExtrapolation::Reject,
            breakpoints: Vec::new(),
        },
    );
    signal.evidence.sample_count = 2;
    model
}

#[test]
fn realizes_typed_force_signal() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let realized = realize_model_excitations(&model).expect("typed force should realize");
    let excitation = realized.iter().next().expect("one excitation");

    assert_eq!(excitation.excitation_id.0, "tip_load");
    assert!(matches!(
        excitation.kind,
        RealizedExcitationKind::Force(value)
            if value.real == [0.0, -1000.0, 0.0] && value.imaginary == [0.0; 3]
    ));
}

#[test]
fn static_realization_does_not_implicitly_take_the_first_time_sample() {
    let model = tabulated_force_model(runmat_analysis_core::SignalAxis::TimeSeconds(vec![
        0.0, 1.0,
    ]));

    let error = realize_model_excitations(&model).unwrap_err();

    assert_eq!(
        error.to_string(),
        "excitation sample point does not match the signal domain"
    );
}

#[test]
fn mutually_exclusive_cases_require_an_explicit_selection() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitation_cases[0].activation =
        runmat_analysis_core::ExcitationCaseActivation::MutuallyExclusiveGroup {
            group_id: "load_variant".into(),
        };

    let error = realize_model_excitations(&model).unwrap_err();

    assert!(error.to_string().contains("no case selection was supplied"));
}

#[test]
fn malformed_tabulated_shape_fails_before_ordinate_indexing() {
    let model = tabulated_force_model(runmat_analysis_core::SignalAxis::TimeSeconds(vec![
        0.0, 0.5, 1.0,
    ]));
    let step = model.sequence.steps[0].step_id.clone();

    let error = realize_excitations(
        &model,
        Some(&step.0),
        ExcitationSamplePoint::TimeSeconds(0.25),
    )
    .unwrap_err();

    assert_eq!(
        error.to_string(),
        "tabulated axis must be finite, strictly increasing, and match values"
    );
}

#[test]
fn explicit_time_and_parameter_samples_use_the_matching_domain() {
    let time_model = tabulated_force_model(runmat_analysis_core::SignalAxis::TimeSeconds(vec![
        0.0, 1.0,
    ]));
    let time_step = time_model.sequence.steps[0].step_id.clone();
    let time = realize_excitations(
        &time_model,
        Some(&time_step.0),
        ExcitationSamplePoint::TimeSeconds(0.25),
    )
    .unwrap();
    assert!(matches!(
        time.iter().next().unwrap().kind,
        RealizedExcitationKind::Force(value) if value.real == [5.0, 10.0, 15.0]
    ));

    let parameter_model = tabulated_force_model(runmat_analysis_core::SignalAxis::StudyParameter {
        parameter_id: "load_scale".into(),
        canonical_values: vec![0.0, 1.0],
    });
    let parameter_step = parameter_model.sequence.steps[0].step_id.clone();
    let parameter = realize_excitations(
        &parameter_model,
        Some(&parameter_step.0),
        ExcitationSamplePoint::StudyParameter {
            parameter_id: "load_scale",
            canonical_value: 0.5,
        },
    )
    .unwrap();
    assert!(matches!(
        parameter.iter().next().unwrap().kind,
        RealizedExcitationKind::Force(value) if value.real == [10.0, 20.0, 30.0]
    ));
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

    let realized = realize_model_excitations(&model).expect("directional scalar should realize");

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

    let error = realize_model_excitations(&model).expect_err("direction should be required");

    assert_eq!(
        error.to_string(),
        "scalar directional excitation requires a direction"
    );
}

#[test]
fn coalesces_reference_point_force_and_moment_into_wrench() {
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

    let realized = realize_model_excitations(&model).expect("wrench should realize");
    assert_eq!(realized.len(), 1);
    assert!(matches!(
        &realized.iter().next().expect("one wrench").kind,
        RealizedExcitationKind::Wrench { force, moment, point }
            if force.real == [1.0, 2.0, 3.0]
                && moment.real == [4.0, 5.0, 6.0]
                && point.meters == [0.1, 0.2, 0.3]
    ));
}

#[test]
fn preserves_complex_frequency_domain_current_density() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.electromagnetic = Some(runmat_analysis_core::ElectromagneticDomain {
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
                        axis: runmat_analysis_core::SignalAxis::FrequencyHertz(vec![60.0]),
                        values: vec![runmat_analysis_core::SignalOrdinate::Complex {
                            real: runmat_analysis_core::SignalValue::Vector3(
                                [1.0, 2.0, 3.0]
                                    .map(runmat_analysis_core::ElectricCurrentDensity::new),
                            ),
                            imaginary: runmat_analysis_core::SignalValue::Vector3(
                                [4.0, 5.0, 6.0]
                                    .map(runmat_analysis_core::ElectricCurrentDensity::new),
                            ),
                        }],
                        interpolation: runmat_analysis_core::SignalInterpolation::Linear,
                        extrapolation: runmat_analysis_core::SignalExtrapolation::HoldEndpoint,
                        breakpoints: Vec::new(),
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

    let realized = realize_model_excitations(&model).expect("complex current should realize");
    assert!(matches!(
        realized.iter().next().expect("one current").kind,
        RealizedExcitationKind::CurrentDensity(value)
            if value.real == [1.0, 2.0, 3.0] && value.imaginary == [4.0, 5.0, 6.0]
    ));
}
