use runmat_analysis_core::{
    CoordinateFrame, GeneratorProvenance, SignalApproximationEvidence, SignalExtrapolation,
    SignalId, SignalInterpolation, SignalPrecision,
};

use crate::execution::{
    CanonicalSignalKnot, CanonicalSignalRealizationInput, SignalMaterializationIdentity,
};

use super::*;

fn realization(times: &[f64], accelerations: &[f64]) -> CanonicalSignalRealization {
    let materialization = SignalMaterializationIdentity::new(
        Digest::sha256(b"srs-test-history"),
        None,
        Digest::sha256(b"srs-test-policy"),
    )
    .unwrap();
    CanonicalSignalRealization::new(CanonicalSignalRealizationInput {
        materialization,
        signal_id: SignalId("base_acceleration".into()),
        quantity: PhysicalQuantityKind::Acceleration,
        shape: SignalShape::Scalar,
        topology_entity_count: None,
        domain: GeneratorDomain::TimeSeconds,
        knots: times
            .iter()
            .copied()
            .map(CanonicalSignalKnot::continuous)
            .collect(),
        precision: SignalPrecision::Real64,
        values: CanonicalSignalValues::real(accelerations.iter().copied()),
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::Reject,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: Some(0.0),
            sample_count: times.len(),
        },
        provenance: GeneratorProvenance {
            project_revision: "revision".into(),
            callable_id: "fixture".into(),
            frozen_input_digest: "sha256:fixture".into(),
        },
        input_admissions: Vec::new(),
    })
    .unwrap()
}

fn request(response: SpectrumResponse, envelope: SpectrumEnvelope) -> ShockResponseSpectrumRequest {
    ShockResponseSpectrumRequest {
        spectrum_id: ResponseSpectrumId("unit_step".into()),
        abscissa: SpectrumAbscissa::FrequencyHertz(runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Frequency,
            _,
        >::new(vec![1.0])),
        damping: SpectrumDamping::Ratio(0.0),
        response,
        envelope,
        interpolation: SpectrumInterpolation::LinearFrequency,
        extrapolation: SpectrumExtrapolation::Reject,
        direction: SpectrumDirection {
            frame: CoordinateFrame::Global,
            vector: [1.0, 0.0, 0.0],
        },
        modal_combination: ModalCombination::Srss,
        directional_combination: DirectionalCombination::Srss,
        truncation: ModalTruncationPolicy {
            minimum_cumulative_mass_ratio: 1.0,
            correction: runmat_analysis_core::ModalTruncationCorrection::None,
        },
        rigid_response: runmat_analysis_core::RigidResponsePolicy::Disabled,
        method: ShockResponseSpectrumMethod::ExactPiecewiseLinear,
    }
}

#[test]
fn exact_piecewise_linear_step_matches_undamped_closed_form() {
    let source = realization(&[0.0, 1.0], &[1.0, 1.0]);
    let displacement = construct_shock_response_spectrum(
        &source,
        request(
            SpectrumResponse::RelativeDisplacement,
            SpectrumEnvelope::Negative,
        ),
    )
    .unwrap();
    let expected = 2.0 / (2.0 * std::f64::consts::PI).powi(2);
    assert!((displacement.resolved_ordinates().unwrap()[0] - expected).abs() <= 1.0e-11);

    let acceleration = construct_shock_response_spectrum(
        &source,
        request(
            SpectrumResponse::AbsoluteAcceleration,
            SpectrumEnvelope::Positive,
        ),
    )
    .unwrap();
    assert!((acceleration.resolved_ordinates().unwrap()[0] - 2.0).abs() <= 1.0e-10);
    assert_eq!(
        acceleration.construction,
        SpectrumConstruction::DerivedFromAccelerationHistory {
            source_signal_id: SignalId("base_acceleration".into()),
            method: ShockResponseSpectrumMethod::ExactPiecewiseLinear,
        }
    );
    assert!(acceleration
        .ordinates
        .numerical_digest()
        .unwrap()
        .starts_with("sha256:"));
}

#[test]
fn zero_history_produces_zero_spectrum_for_every_response_quantity() {
    let source = realization(&[0.0, 0.01, 0.03], &[0.0, 0.0, 0.0]);
    for response in [
        SpectrumResponse::RelativeDisplacement,
        SpectrumResponse::RelativeVelocity,
        SpectrumResponse::RelativeAcceleration,
        SpectrumResponse::AbsoluteAcceleration,
        SpectrumResponse::PseudoVelocity,
        SpectrumResponse::PseudoAcceleration,
    ] {
        let spectrum = construct_shock_response_spectrum(
            &source,
            request(response, SpectrumEnvelope::Maximax),
        )
        .unwrap();
        assert_eq!(spectrum.resolved_ordinates().unwrap(), &[0.0]);
    }
}

#[test]
fn exact_piecewise_linear_transition_matches_undamped_ramp_solution() {
    let omega = 2.0 * std::f64::consts::PI;
    let duration = 0.37;
    let state =
        advance_piecewise_linear(OscillatorState::default(), 0.0, 1.0, duration, omega, 0.0)
            .unwrap();
    let expected_displacement = (omega * duration).sin() / omega.powi(3) - duration / omega.powi(2);
    let expected_velocity = (omega * duration).cos() / omega.powi(2) - 1.0 / omega.powi(2);
    assert!((state.displacement - expected_displacement).abs() <= 1.0e-13);
    assert!((state.velocity - expected_velocity).abs() <= 1.0e-13);
}

#[test]
fn damped_step_peak_matches_closed_form_overshoot() {
    let damping = 0.05;
    let source = realization(&[0.0, 2.0], &[1.0, 1.0]);
    let mut spectrum_request = request(
        SpectrumResponse::RelativeDisplacement,
        SpectrumEnvelope::Negative,
    );
    spectrum_request.damping = SpectrumDamping::Ratio(damping);
    let spectrum = construct_shock_response_spectrum(&source, spectrum_request).unwrap();
    let omega = 2.0 * std::f64::consts::PI;
    let overshoot = (-damping * std::f64::consts::PI / (1.0 - damping * damping).sqrt()).exp();
    let expected = (1.0 + overshoot) / omega.powi(2);
    assert!((spectrum.resolved_ordinates().unwrap()[0] - expected).abs() <= 1.0e-11);
}

#[test]
fn construction_rejects_unimplemented_method_and_wrong_signal_quantity() {
    let source = realization(&[0.0, 1.0], &[0.0, 0.0]);
    let mut recursive = request(
        SpectrumResponse::AbsoluteAcceleration,
        SpectrumEnvelope::Maximax,
    );
    recursive.method = ShockResponseSpectrumMethod::RecursiveFilter;
    assert!(construct_shock_response_spectrum(&source, recursive).is_err());

    let mut wrong = source;
    wrong.quantity = PhysicalQuantityKind::Velocity;
    assert!(construct_shock_response_spectrum(
        &wrong,
        request(
            SpectrumResponse::AbsoluteAcceleration,
            SpectrumEnvelope::Maximax,
        ),
    )
    .is_err());
}
