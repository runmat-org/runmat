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

#[test]
fn rejected_literal_grids_never_infer_a_success_cardinality() {
    for (entry, spectrum, controls) in [
        (
            &super::super::periodogram::PERIODOGRAM_CATALOG_ENTRY,
            Spectrum::Periodogram,
            vec![LiteralValue::Empty, LiteralValue::Number(0.0)],
        ),
        (
            &super::super::spectrogram::SPECTROGRAM_CATALOG_ENTRY,
            Spectrum::Spectrogram,
            vec![
                LiteralValue::Empty,
                LiteralValue::Empty,
                LiteralValue::Number(0.0),
            ],
        ),
    ] {
        let request = request([8, 1], controls);
        let result = infer(&request, entry, spectrum);
        assert!(result
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-SPECTRAL-GRID"));
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
        assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
    }
    let too_large = request(
        [8, 1],
        vec![
            LiteralValue::Empty,
            LiteralValue::Number(((1 << 22) + 1) as f64),
        ],
    );
    let result = infer(
        &too_large,
        &super::super::periodogram::PERIODOGRAM_CATALOG_ENTRY,
        Spectrum::Periodogram,
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-SPECTRAL-GRID"));
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn singleton_numeric_vector_grid_is_scalar_uniform_nfft_for_both_identities() {
    for (entry, spectrum, controls) in [
        (
            &super::super::periodogram::PERIODOGRAM_CATALOG_ENTRY,
            Spectrum::Periodogram,
            vec![
                LiteralValue::Empty,
                LiteralValue::Vector(vec![LiteralValue::Number(2.0)]),
            ],
        ),
        (
            &super::super::spectrogram::SPECTROGRAM_CATALOG_ENTRY,
            Spectrum::Spectrogram,
            vec![
                LiteralValue::Empty,
                LiteralValue::Empty,
                LiteralValue::Vector(vec![LiteralValue::Number(2.0)]),
            ],
        ),
    ] {
        let result = infer(&request([8, 1], controls), entry, spectrum);
        assert!(!result
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-SPECTRAL-GRID"));
        assert_eq!(
            result.outputs[0].numeric().unwrap().class,
            NumericClass::Single
        );
        let columns = if matches!(spectrum, Spectrum::Periodogram) {
            1
        } else {
            8
        };
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(columns)])
        );
    }
}

#[test]
fn empty_numeric_vector_grid_uses_default_fft_length_for_both_identities() {
    for (entry, spectrum, controls, columns) in [
        (
            &super::super::periodogram::PERIODOGRAM_CATALOG_ENTRY,
            Spectrum::Periodogram,
            vec![LiteralValue::Empty, LiteralValue::Vector(vec![])],
            1,
        ),
        (
            &super::super::spectrogram::SPECTROGRAM_CATALOG_ENTRY,
            Spectrum::Spectrogram,
            vec![
                LiteralValue::Empty,
                LiteralValue::Empty,
                LiteralValue::Vector(vec![]),
            ],
            8,
        ),
    ] {
        let result = infer(&request([8, 1], controls), entry, spectrum);
        assert!(!result
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-SPECTRAL-GRID"));
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(129), Some(columns)])
        );
    }
}

#[test]
fn singleton_numeric_spectrogram_window_uses_scalar_hamming_length() {
    let controls = vec![
        LiteralValue::Vector(vec![LiteralValue::Number(4.0)]),
        LiteralValue::Number(2.0),
        LiteralValue::Empty,
    ];
    let result = infer(
        &request([8, 1], controls),
        &super::super::spectrogram::SPECTROGRAM_CATALOG_ENTRY,
        Spectrum::Spectrogram,
    );
    assert!(result.diagnostics.is_empty());
    // Hamming length 4, 50% overlap: starts 0, 2, 4.
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(129), Some(3)])
    );
}

#[test]
fn source_proven_invalid_shapes_and_options_have_no_named_success_fact() {
    let mut invalid = vec![
        (
            request([8, 3], vec![]),
            Spectrum::Spectrogram,
            "RM-CATALOG-SPECTRAL-VECTOR",
        ),
        (
            request(
                [8, 1],
                vec![LiteralValue::Vector(vec![
                    LiteralValue::Number(1.0),
                    LiteralValue::Number(1.0),
                ])],
            ),
            Spectrum::Periodogram,
            "RM-CATALOG-SPECTRAL-WINDOW",
        ),
        (
            request([8, 1], vec![LiteralValue::Number(0.0)]),
            Spectrum::Spectrogram,
            "RM-CATALOG-SPECTRAL-WINDOW",
        ),
        (
            request(
                [8, 1],
                vec![LiteralValue::Number(4.0), LiteralValue::Number(4.0)],
            ),
            Spectrum::Spectrogram,
            "RM-CATALOG-SPECTRAL-OVERLAP",
        ),
        (
            request(
                [8, 1],
                vec![
                    LiteralValue::Empty,
                    LiteralValue::Vector(vec![
                        LiteralValue::Number(f64::NAN),
                        LiteralValue::Number(0.5),
                    ]),
                ],
            ),
            Spectrum::Periodogram,
            "RM-CATALOG-SPECTRAL-GRID",
        ),
        (
            request(
                [8, 1],
                vec![
                    LiteralValue::Empty,
                    LiteralValue::Empty,
                    LiteralValue::Vector(vec![
                        LiteralValue::Number(0.0),
                        LiteralValue::Complex {
                            real: "0.0".into(),
                            imaginary: "1.0".into(),
                            class: NumericClass::Double,
                        },
                    ]),
                ],
            ),
            Spectrum::Spectrogram,
            "RM-CATALOG-SPECTRAL-GRID",
        ),
        (
            request([8, 1], vec![LiteralValue::String("reassigned".into())]),
            Spectrum::Spectrogram,
            "RM-CATALOG-SPECTRAL-OPTION",
        ),
        (
            request(
                [8, 1],
                vec![
                    LiteralValue::Empty,
                    LiteralValue::Vector(vec![
                        LiteralValue::Number(0.0),
                        LiteralValue::Number(0.5),
                    ]),
                    LiteralValue::String("centered".into()),
                ],
            ),
            Spectrum::Periodogram,
            "RM-CATALOG-SPECTRAL-OPTION",
        ),
    ];
    let mut complex = request([8, 1], vec![LiteralValue::String("onesided".into())]);
    complex.arguments[0].kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Complex,
    });
    invalid.push((complex, Spectrum::Periodogram, "RM-CATALOG-SPECTRAL-OPTION"));
    for (request, spectrum, code) in invalid {
        let entry = match spectrum {
            Spectrum::Periodogram => &super::super::periodogram::PERIODOGRAM_CATALOG_ENTRY,
            Spectrum::Spectrogram => &super::super::spectrogram::SPECTROGRAM_CATALOG_ENTRY,
        };
        let result = infer(&request, entry, spectrum);
        assert!(
            result.diagnostics.iter().any(|d| d.code == code),
            "{code}: {:?}",
            result.diagnostics
        );
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown, "{code}");
        assert_eq!(result.outputs[0].shape, ShapeFact::Unknown, "{code}");
    }
}

#[test]
fn matrix_periodogram_window_uses_row_count_not_total_elements() {
    let wrong = request(
        [8, 3],
        vec![LiteralValue::Vector(vec![
            LiteralValue::Number(1.0),
            LiteralValue::Number(1.0),
        ])],
    );
    let invalid = infer(
        &wrong,
        &super::super::periodogram::PERIODOGRAM_CATALOG_ENTRY,
        Spectrum::Periodogram,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-SPECTRAL-WINDOW"));
    assert_eq!(invalid.outputs[0].kind, ValueKindFact::Unknown);
    let valid = request(
        [8, 3],
        vec![LiteralValue::Vector(vec![LiteralValue::Number(1.0); 8])],
    );
    let admitted = infer(
        &valid,
        &super::super::periodogram::PERIODOGRAM_CATALOG_ENTRY,
        Spectrum::Periodogram,
    );
    assert!(admitted.diagnostics.is_empty());
    assert_eq!(
        admitted.outputs[0].shape,
        ShapeFact::from(vec![Some(129), Some(3)])
    );
}

#[test]
fn proven_invalid_numeric_controls_never_publish_named_success_facts() {
    let cases = [
        (
            Spectrum::Periodogram,
            vec![LiteralValue::Vector(
                vec![LiteralValue::Number(1.0); 7]
                    .into_iter()
                    .chain([LiteralValue::Number(f64::NAN)])
                    .collect(),
            )],
            "RM-CATALOG-SPECTRAL-WINDOW",
        ),
        (
            Spectrum::Spectrogram,
            vec![LiteralValue::Number(-1.0)],
            "RM-CATALOG-SPECTRAL-WINDOW",
        ),
        (
            Spectrum::Spectrogram,
            vec![LiteralValue::Number(4.0), LiteralValue::Number(1.5)],
            "RM-CATALOG-SPECTRAL-OVERLAP",
        ),
        (
            Spectrum::Periodogram,
            vec![LiteralValue::Empty, LiteralValue::Number(-2.0)],
            "RM-CATALOG-SPECTRAL-GRID",
        ),
        (
            Spectrum::Spectrogram,
            vec![
                LiteralValue::Empty,
                LiteralValue::Empty,
                LiteralValue::Number(2.5),
            ],
            "RM-CATALOG-SPECTRAL-GRID",
        ),
    ];
    for (spectrum, controls, code) in cases {
        let entry = match spectrum {
            Spectrum::Periodogram => &super::super::periodogram::PERIODOGRAM_CATALOG_ENTRY,
            Spectrum::Spectrogram => &super::super::spectrogram::SPECTROGRAM_CATALOG_ENTRY,
        };
        let result = infer(&request([8, 1], controls), entry, spectrum);
        assert!(
            result.diagnostics.iter().any(|d| d.code == code),
            "{code}: {:?}",
            result.diagnostics
        );
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown, "{code}");
    }
}
