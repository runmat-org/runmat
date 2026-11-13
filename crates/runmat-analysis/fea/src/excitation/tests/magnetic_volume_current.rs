use super::*;
use runmat_analysis_core::{
    ElectricCurrentDensity, ExcitationAction, ExcitationDistribution, PhysicalSignal, Signal,
    SignalId, SignalPrecision, SignalRepresentation, SignalShape, SignalValue,
};

fn model(value: SignalValue<ElectricCurrentDensity>) -> runmat_analysis_core::AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let PhysicalSignal::Force(original) = &model.signals[0] else {
        unreachable!()
    };
    let id = SignalId("coil.volume_current".into());
    model.signals = vec![PhysicalSignal::ElectricCurrentDensity(Signal {
        signal_id: id.clone(),
        shape: SignalShape::Vector3,
        frame: None,
        representation: SignalRepresentation::Constant(value),
        precision: SignalPrecision::Real64,
        evidence: original.evidence.clone(),
        provenance: runmat_analysis_core::SignalProvenance::Authored,
    })];
    let mut action = model.excitations[0].excitation().clone();
    action.signal_id = id;
    action.target = runmat_analysis_core::ExcitationTarget::WholeModel;
    action.direction = None;
    action.distribution = ExcitationDistribution::IntensityPerVolume;
    action.projection.preserve_resultant = false;
    action.projection.preserve_moment = false;
    model.excitations = vec![ExcitationAction::MagneticVolumeCurrentDensity(action)];
    model
}

#[test]
fn magnetic_volume_current_realizes_signed_vector_without_surface_reinterpretation() {
    let vector = [1.0, -2.0, 3.0];
    let input = model(SignalValue::Vector3(
        vector.map(ElectricCurrentDensity::new),
    ));
    let realized = realize_static(&input).unwrap();
    let load = realized.iter().next().unwrap();
    assert_eq!(
        load.policy.distribution,
        ExcitationDistribution::IntensityPerVolume
    );
    assert!(
        matches!(load.kind, RealizedExcitationKind::MagneticVolumeCurrentDensity(value) if value.real == vector && value.imaginary == [0.0; 3])
    );
    assert!(!load.kind.is_structural_load());
    for value in [
        SignalValue::Scalar(ElectricCurrentDensity::new(2.0)),
        SignalValue::Vector3([ElectricCurrentDensity::new(f64::NAN); 3]),
    ] {
        assert!(realize_static(&model(value)).is_err());
    }
}

#[test]
fn magnetic_volume_current_preserves_complex_components_for_consumer_admission() {
    use runmat_analysis_core::{
        SignalExtrapolation, SignalInterpolation, SignalKnot, SignalOrdinate, TabulatedSignal,
        TabulatedSignalDomain,
    };
    let vector = |values: [f64; 3]| SignalValue::Vector3(values.map(ElectricCurrentDensity::new));
    let mut input = model(vector([1.0, -2.0, 3.0]));
    let PhysicalSignal::ElectricCurrentDensity(signal) = &mut input.signals[0] else {
        unreachable!()
    };
    signal.precision = SignalPrecision::Complex64;
    signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
        domain: TabulatedSignalDomain::FrequencyHertz,
        knots: vec![SignalKnot::continuous(
            60.0,
            SignalOrdinate::Complex {
                real: vector([1.0, -2.0, 3.0]),
                imaginary: vector([4.0, 5.0, -6.0]),
            },
        )],
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::HoldEndpoint,
    });
    let realized = realize_step_excitations(
        &input,
        &input.sequence.steps[0],
        ExcitationSamplePoint::frequency(60.0),
    )
    .unwrap();
    assert!(
        matches!(realized.iter().next().unwrap().kind, RealizedExcitationKind::MagneticVolumeCurrentDensity(value)
        if value.real == [1.0, -2.0, 3.0] && value.imaginary == [4.0, 5.0, -6.0])
    );
    assert!(
        matches!(realized.component(super::super::ComplexComponent::Imaginary).iter().next().unwrap().kind,
        RealizedExcitationKind::MagneticVolumeCurrentDensity(value) if value.real == [4.0, 5.0, -6.0] && value.imaginary == [0.0; 3])
    );
}
