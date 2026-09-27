use super::*;

#[test]
fn represented_complex_single_imaginary_value_controls_reality() {
    for (class, imaginary, rejected) in [
        (NumericClass::Single, "1e-50", false),
        (NumericClass::Single, "-1e-50", false),
        (NumericClass::Single, "1e-40", true),
        (NumericClass::Double, "1e-50", true),
        (NumericClass::Double, "0", false),
    ] {
        let request = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Complex,
            }))],
            literals: LiteralContext::new(vec![LiteralValue::Complex {
                class,
                real: "1".into(),
                imaginary: imaginary.into(),
            }]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(0)),
        };
        let result = infer(&request, &crate::MUST_BE_REAL_CATALOG_ENTRY);
        assert_eq!(
            !result.diagnostics.is_empty(),
            rejected,
            "{class:?}/{imaginary}"
        );
        assert!(result.outputs.is_empty());
    }
}

#[test]
fn unknown_complex_contents_do_not_imply_nonzero_imaginary_part() {
    let request = CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        }))],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(0)),
    };
    assert!(infer(&request, &crate::MUST_BE_REAL_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
