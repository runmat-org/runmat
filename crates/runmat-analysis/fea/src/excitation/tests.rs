use crate::fixtures::{
    fixture_model, replace_fixture_excitations, FixtureExcitation, FixtureExcitationKind, FixtureId,
};

use super::{realize_model_excitations, RealizedExcitationKind};

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
