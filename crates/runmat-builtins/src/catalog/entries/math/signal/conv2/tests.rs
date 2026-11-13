use super::*;
use runmat_types::*;

fn numeric(class: NumericClass, domain: NumericDomain, dims: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::from(dims.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

#[test]
fn direct_shape_and_single_complex_class_are_proven() {
    let signal = numeric(NumericClass::Double, NumericDomain::Real, &[3, 4]);
    let kernel = numeric(NumericClass::Single, NumericDomain::Complex, &[2, 3]);
    let full = inference::infer(
        &request(
            vec![signal.clone(), kernel.clone()],
            vec![LiteralValue::Unknown; 2],
        ),
        &CONV2_CATALOG_ENTRY,
    );
    assert!(full.diagnostics.is_empty());
    assert_eq!(
        full.outputs[0].shape,
        ShapeFact::from(vec![Some(4), Some(6)])
    );
    assert_eq!(
        full.outputs[0].numeric(),
        Some(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex
        })
    );
    let same = inference::infer(
        &request(
            vec![signal, kernel, ValueFact::scalar(ValueKindFact::Character)],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Character("same".into()),
            ],
        ),
        &CONV2_CATALOG_ENTRY,
    );
    assert_eq!(
        same.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(4)])
    );
}

#[test]
fn separable_shape_uses_both_kernel_lengths_and_signal_matrix() {
    let column = numeric(NumericClass::Double, NumericDomain::Real, &[3, 1]);
    let row = numeric(NumericClass::Double, NumericDomain::Real, &[1, 2]);
    let signal = numeric(NumericClass::Double, NumericDomain::Real, &[5, 6]);
    let fact = inference::infer(
        &request(vec![column, row, signal], vec![LiteralValue::Unknown; 3]),
        &CONV2_CATALOG_ENTRY,
    );
    assert!(fact.diagnostics.is_empty());
    assert_eq!(
        fact.outputs[0].shape,
        ShapeFact::from(vec![Some(7), Some(7)])
    );
}

#[test]
fn separable_numeric_and_logical_scalar_signals_are_not_text_selectors() {
    let column = numeric(NumericClass::Double, NumericDomain::Real, &[2, 1]);
    let row = numeric(NumericClass::Double, NumericDomain::Real, &[1, 2]);
    for signal in [
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        ValueFact::scalar(ValueKindFact::Logical),
    ] {
        let result = inference::infer(
            &request(
                vec![column.clone(), row.clone(), signal],
                vec![LiteralValue::Unknown; 3],
            ),
            &CONV2_CATALOG_ENTRY,
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(2)])
        );
    }
}

#[test]
fn fourth_input_and_nd_character_selector_cannot_prove_shape() {
    let column = numeric(NumericClass::Double, NumericDomain::Real, &[2, 1]);
    let row = numeric(NumericClass::Double, NumericDomain::Real, &[1, 2]);
    let signal = numeric(NumericClass::Double, NumericDomain::Real, &[1, 1]);
    for selector in [
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        ValueFact::scalar(ValueKindFact::Logical),
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), Some(4), Some(1)]),
            StorageFact::Dense,
        ),
    ] {
        let result = inference::infer(
            &request(
                vec![column.clone(), row.clone(), signal.clone(), selector],
                vec![LiteralValue::Unknown; 4],
            ),
            &CONV2_CATALOG_ENTRY,
        );
        assert!(!result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    }
}

#[test]
fn empty_direct_full_has_canonical_empty_matrix_shape() {
    let left = numeric(NumericClass::Double, NumericDomain::Real, &[0, 1]);
    let right = numeric(NumericClass::Double, NumericDomain::Real, &[1, 3]);
    let result = inference::infer(
        &request(vec![left, right], vec![LiteralValue::Unknown; 2]),
        &CONV2_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(0), Some(0)])
    );
}

#[test]
fn malformed_selector_or_kernel_geometry_cannot_prove_output() {
    let matrix = numeric(NumericClass::Double, NumericDomain::Real, &[2, 2]);
    let signal = numeric(NumericClass::Double, NumericDomain::Real, &[4, 4]);
    let bad_kernel = inference::infer(
        &request(
            vec![matrix.clone(), matrix, signal.clone()],
            vec![LiteralValue::Unknown; 3],
        ),
        &CONV2_CATALOG_ENTRY,
    );
    assert!(!bad_kernel.diagnostics.is_empty());
    assert_eq!(bad_kernel.outputs[0].kind, ValueKindFact::Unknown);
    let bad_selector = inference::infer(
        &request(
            vec![
                signal.clone(),
                signal,
                ValueFact::scalar(ValueKindFact::String),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("circular".into()),
            ],
        ),
        &CONV2_CATALOG_ENTRY,
    );
    assert!(!bad_selector.diagnostics.is_empty());
    assert_eq!(bad_selector.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn canonical_metadata_retains_four_forms_and_integer_classes() {
    let labels = CONV2_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect::<Vec<_>>();
    assert_eq!(
        labels,
        vec![
            "C = conv2(A, B)",
            "C = conv2(A, B, shape)",
            "C = conv2(hcol, hrow, A)",
            "C = conv2(hcol, hrow, A, shape)"
        ]
    );
    assert!(CONV2_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == "RM.CONV2.SHAPE_INVALID"));
    assert_eq!(CONV2_INTEGER_CAPABILITIES.len(), 2);
    assert!(CONV2_INTEGER_CAPABILITIES
        .iter()
        .flat_map(|capability| capability.inputs)
        .all(|input| input.classes.len() == 8));
}
