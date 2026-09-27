use super::*;

fn request(shape: [usize; 2], controls: Vec<LiteralValue>) -> CallRequest {
    let signal = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape.into_iter().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    );
    let mut arguments = vec![signal];
    arguments.extend(
        controls
            .iter()
            .map(|_| ValueFact::unknown(DynamicReason::RuntimeValue)),
    );
    let mut literals = vec![LiteralValue::Unknown];
    literals.extend(controls);
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(4)),
    }
}

#[test]
fn periodogram_retains_channels_and_normalizes_row_vectors() {
    let matrix = request([8, 3], vec![]);
    assert_eq!(
        geometry::output_shape(&matrix, Spectrum::Periodogram, 0),
        ShapeFact::from(vec![Some(129), Some(3)])
    );
    let row = request([1, 8], vec![]);
    assert_eq!(
        geometry::output_shape(&row, Spectrum::Periodogram, 0),
        ShapeFact::from(vec![Some(129), Some(1)])
    );
    let unknown = request([8, 3], vec![LiteralValue::Unknown]);
    assert_eq!(
        geometry::output_shape(&unknown, Spectrum::Periodogram, 0),
        ShapeFact::from(vec![None, Some(3)])
    );
}

#[test]
fn known_fft_range_frequency_and_frame_controls_refine_geometry() {
    let uniform = request(
        [8, 3],
        vec![
            LiteralValue::Empty,
            LiteralValue::Number(8.0),
            LiteralValue::String("centered".into()),
        ],
    );
    assert_eq!(
        geometry::output_shape(&uniform, Spectrum::Periodogram, 0),
        ShapeFact::from(vec![Some(8), Some(3)])
    );
    let explicit = request(
        [8, 3],
        vec![
            LiteralValue::Empty,
            LiteralValue::Vector(vec![
                LiteralValue::Number(0.0),
                LiteralValue::Number(0.3),
                LiteralValue::Number(0.7),
            ]),
        ],
    );
    assert_eq!(
        geometry::output_shape(&explicit, Spectrum::Periodogram, 1),
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    let frames = request(
        [1, 8],
        vec![
            LiteralValue::Number(4.0),
            LiteralValue::Number(2.0),
            LiteralValue::Number(8.0),
            LiteralValue::String("twosided".into()),
        ],
    );
    assert_eq!(
        geometry::output_shape(&frames, Spectrum::Spectrogram, 0),
        ShapeFact::from(vec![Some(8), Some(3)])
    );
    assert_eq!(
        geometry::output_shape(&frames, Spectrum::Spectrogram, 2),
        ShapeFact::from(vec![Some(1), Some(3)])
    );
    let complex = {
        let mut r = request([8, 1], vec![]);
        r.arguments[0].kind = ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        });
        r
    };
    assert_eq!(
        geometry::output_shape(&complex, Spectrum::Periodogram, 1),
        ShapeFact::from(vec![Some(256), Some(1)])
    );
}

#[test]
fn both_identities_diagnose_invalid_arity_and_non_numeric_signals() {
    for (entry, spectrum, maximum) in [
        (
            &super::super::periodogram::PERIODOGRAM_CATALOG_ENTRY,
            Spectrum::Periodogram,
            7,
        ),
        (
            &super::super::spectrogram::SPECTROGRAM_CATALOG_ENTRY,
            Spectrum::Spectrogram,
            10,
        ),
    ] {
        let mut empty = request([8, 1], vec![]);
        empty.arguments.clear();
        assert!(!infer(&empty, entry, spectrum).diagnostics.is_empty());
        let excessive = request([8, 1], vec![LiteralValue::Unknown; maximum]);
        assert!(!infer(&excessive, entry, spectrum).diagnostics.is_empty());
        let mut text = request([8, 1], vec![]);
        text.arguments[0].kind = ValueKindFact::String;
        let result = infer(&text, entry, spectrum);
        assert!(!result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    }
}
