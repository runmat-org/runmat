//! Complex frequency samples use the shared chunk/value/table authority.
use super::*;
use runmat_analysis_core::{SignalKnot, SignalValue, TabulatedSignal, TabulatedSignalDomain};

#[test]
fn chunked_complex_frequency_materialization_matches_explicit_table() {
    let mut generated = signal(SignalShape::Scalar, SignalPrecision::Complex128);
    let PhysicalSignal::Force(value) = &mut generated else {
        unreachable!()
    };
    let SignalRepresentation::Generator(generator) = &mut value.representation else {
        unreachable!()
    };
    generator.domain = GeneratorDomain::FrequencyHertz;
    generator.sampling = SignalSamplingPolicy::CanonicalPoints(vec![10.0, 17.0, 20.0]);
    let mut admitted = admitted();
    admitted.sample_points = runmat_types::physical::Quantity::try_from_descriptor(
        GeneratorDomain::FrequencyHertz
            .axis_descriptor(None)
            .unwrap(),
        vec![10.0, 17.0, 20.0],
    )
    .unwrap();
    let expected = [(1.0, 0.4), (2.0, -0.3), (-1.0, 0.8)];
    let mut explicit = generated.clone();
    let PhysicalSignal::Force(value) = &mut explicit else {
        unreachable!()
    };
    value.representation = SignalRepresentation::Tabulated(TabulatedSignal {
        domain: TabulatedSignalDomain::FrequencyHertz,
        knots: admitted
            .sample_points
            .canonical_values()
            .iter()
            .zip(expected)
            .map(|(f, (real, imaginary))| {
                SignalKnot::continuous(
                    *f,
                    SignalOrdinate::Complex {
                        real: SignalValue::Scalar(Force::new(real)),
                        imaginary: SignalValue::Scalar(Force::new(imaginary)),
                    },
                )
            })
            .collect(),
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::Reject,
    });
    value.evidence.sample_count = 3;
    // Both one-chunk and split-chunk execution must retain exactly the same table.
    for count in [1, 2, 3] {
        let chunks = plan_signal_generator_chunks(
            &generated,
            &admitted,
            None,
            RESULT_ENVELOPE_BYTES + count * 16,
            |_| panic!("no artifact argument expected"),
        )
        .unwrap()
        .chunks;
        let outputs = chunks
            .iter()
            .map(|chunk| {
                let samples = chunk
                    .sample_points()
                    .map(|f| {
                        let index = admitted
                            .sample_points
                            .canonical_values()
                            .iter()
                            .position(|point| *point == f)
                            .unwrap();
                        expected[index]
                    })
                    .collect::<Vec<_>>();
                let bytes = samples
                    .iter()
                    .flat_map(|(real, imaginary)| {
                        [*real, *imaginary]
                            .into_iter()
                            .flat_map(|v| v.to_bits().to_le_bytes())
                    })
                    .collect();
                output(
                    chunk,
                    ValuePayload::Inline(Box::new(InlineValue::Dense(DenseValue {
                        element_type: ElementType::ComplexF64,
                        shape: vec![samples.len() as u64, 1],
                        little_endian_data: bytes,
                    }))),
                )
            })
            .collect::<Vec<_>>();
        let realization = canonicalize_signal_generator_outputs(
            &generated,
            &admitted,
            None,
            &[],
            &chunks,
            &outputs,
        )
        .unwrap();
        let materialized =
            materialize_signal_for_solver(&generated, &admitted, &realization).unwrap();
        assert_eq!(
            canonical_tabulated_signal_content_digest(&materialized).unwrap(),
            canonical_tabulated_signal_content_digest(&explicit).unwrap()
        );
        let PhysicalSignal::Force(value) = materialized else {
            unreachable!()
        };
        let PhysicalSignal::Force(reference) = &explicit else {
            unreachable!()
        };
        assert_eq!(value.representation, reference.representation);
        assert!(matches!(value.provenance, SignalProvenance::Generated(_)));
        // Real-only declarations cannot silently discard the imaginary component.
        let mut real_only = generated.clone();
        let PhysicalSignal::Force(value) = &mut real_only else {
            unreachable!()
        };
        value.precision = SignalPrecision::Real64;
        assert!(canonicalize_signal_generator_outputs(
            &real_only,
            &admitted,
            None,
            &[],
            &chunks,
            &outputs
        )
        .is_err());
        let mut substituted = admitted.clone();
        substituted
            .sample_points
            .try_replace(
                1,
                runmat_types::physical::Quantity::try_from_descriptor(
                    substituted.sample_points.descriptor(),
                    18.0,
                )
                .unwrap(),
            )
            .unwrap();
        assert!(validate_signal_generator_chunks(&generated, &substituted, &chunks).is_err());
    }
}
