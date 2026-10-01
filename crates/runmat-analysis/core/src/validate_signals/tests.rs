use crate::{
    Force, GeneratorDomain, GeneratorEffectPolicy, GeneratorProvenance, PhysicalSignal, Signal,
    SignalApproximationEvidence, SignalExtrapolation, SignalGenerator, SignalId,
    SignalInterpolation, SignalKnot, SignalOrdinate, SignalPointContinuity, SignalPrecision,
    SignalProvenance, SignalRepresentation, SignalSamplingPolicy, SignalShape, SignalValue,
    TabulatedSignal, TabulatedSignalDomain,
};

use super::validate_signals;

fn generated_force(provenance: SignalProvenance) -> PhysicalSignal {
    PhysicalSignal::Force(Signal::<Force> {
        signal_id: SignalId("loads.launch_force".to_string()),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Generator(SignalGenerator {
            invocation: crate::GeneratorInvocation {
                callable: crate::GeneratorCallable::ProjectFunction {
                    qualified_name: "loads.launch_force".to_string(),
                },
                arguments: Vec::new(),
                effects: GeneratorEffectPolicy::Pure,
            },
            domain: GeneratorDomain::TimeSeconds,
            sampling: SignalSamplingPolicy::Uniform {
                start: 0.0,
                end: 0.1,
                sample_count: 1001,
            },
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count: 0,
        },
        provenance,
    })
}

#[test]
fn authored_generator_declaration_is_valid() {
    let signal = generated_force(SignalProvenance::Authored);

    validate_signals(&[signal]).expect("authored generator should validate");
}

#[test]
fn bound_function_identity_is_typed_and_requires_a_display_name() {
    let mut signal = generated_force(SignalProvenance::Authored);
    set_bound_generator(&mut signal, 17, "pressureHistory");
    validate_signals(&[signal.clone()]).expect("semantic callable identity should validate");

    set_bound_generator(&mut signal, 17, "");
    assert!(validate_signals(&[signal.clone()]).is_err());
}

fn set_bound_generator(signal: &mut PhysicalSignal, function: u32, display_name: &str) {
    let PhysicalSignal::Force(signal) = signal else {
        unreachable!()
    };
    let SignalRepresentation::Generator(generator) = &mut signal.representation else {
        unreachable!()
    };
    generator.invocation.callable = crate::GeneratorCallable::BoundFunction {
        function: runmat_types::ProgramFunctionId(function),
        display_name: display_name.into(),
    };
}

#[test]
fn generator_cannot_claim_materialized_provenance() {
    let signal = generated_force(SignalProvenance::Generated(GeneratorProvenance {
        project_revision: "sha256:project".to_string(),
        callable_id: "loads.launch_force".to_string(),
        frozen_input_digest: "sha256:inputs".to_string(),
    }));

    assert!(validate_signals(&[signal]).is_err());
}

#[test]
fn jump_knots_are_typed_and_must_be_interior() {
    let mut signal = generated_force(SignalProvenance::Authored);
    let PhysicalSignal::Force(signal) = &mut signal else {
        unreachable!()
    };
    signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
        domain: TabulatedSignalDomain::TimeSeconds,
        knots: vec![
            SignalKnot::continuous(0.0, force_ordinate(0.0)),
            SignalKnot::jump(
                0.5,
                force_ordinate(1.0),
                force_ordinate(2.0),
                SignalPointContinuity::Right,
            ),
            SignalKnot::continuous(1.0, force_ordinate(3.0)),
        ],
        interpolation: SignalInterpolation::RightEndpoint,
        extrapolation: SignalExtrapolation::Reject,
    });
    signal.evidence.sample_count = 3;
    validate_signals(&[PhysicalSignal::Force(signal.clone())])
        .expect("interior jump should validate");

    let SignalRepresentation::Tabulated(table) = &mut signal.representation else {
        unreachable!()
    };
    table.knots.swap(0, 1);
    let error = validate_signals(&[PhysicalSignal::Force(signal.clone())]).unwrap_err();
    assert!(error.to_string().contains("strictly increasing"));

    let SignalRepresentation::Tabulated(table) = &mut signal.representation else {
        unreachable!()
    };
    table.knots.swap(0, 1);
    table.knots[0] = SignalKnot::jump(
        0.0,
        force_ordinate(0.0),
        force_ordinate(1.0),
        SignalPointContinuity::Left,
    );
    let error = validate_signals(&[PhysicalSignal::Force(signal.clone())]).unwrap_err();
    assert!(error.to_string().contains("jump knots must be interior"));
}

#[test]
fn authored_knots_reject_negative_zero_abscissae() {
    let mut signal = generated_force(SignalProvenance::Authored);
    let PhysicalSignal::Force(signal) = &mut signal else {
        unreachable!()
    };
    signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
        domain: TabulatedSignalDomain::TimeSeconds,
        knots: vec![
            SignalKnot::continuous(-0.0, force_ordinate(0.0)),
            SignalKnot::continuous(1.0, force_ordinate(1.0)),
        ],
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::Reject,
    });
    signal.evidence.sample_count = 2;
    assert!(validate_signals(&[PhysicalSignal::Force(signal.clone())])
        .unwrap_err()
        .to_string()
        .contains("finite, strictly increasing"));
}

fn force_ordinate(value: f64) -> SignalOrdinate<Force> {
    SignalOrdinate::Real(SignalValue::Scalar(Force::new(value)))
}
