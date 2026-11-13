use super::*;
use runmat_types::*;

fn numeric(class: NumericClass, domain: NumericDomain, dims: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::from(dims.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}

fn request(left: ValueFact, right: ValueFact, shape: Option<&str>) -> CallRequest {
    let mut arguments = vec![left, right];
    let mut literals = vec![LiteralValue::Unknown, LiteralValue::Unknown];
    if let Some(shape) = shape {
        arguments.push(ValueFact::scalar(ValueKindFact::Character));
        literals.push(LiteralValue::Character(shape.into()));
    }
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

#[test]
fn convolution_geometry_and_class_follow_both_operands_and_typed_selector() {
    let column = numeric(NumericClass::Double, NumericDomain::Real, &[3, 1]);
    let single = numeric(NumericClass::Single, NumericDomain::Complex, &[2, 1]);
    let full = inference::infer(
        &request(column.clone(), single.clone(), None),
        &CONV_CATALOG_ENTRY,
    );
    assert!(full.diagnostics.is_empty());
    assert_eq!(
        full.outputs[0].shape,
        ShapeFact::from(vec![Some(4), Some(1)])
    );
    assert_eq!(
        full.outputs[0].numeric(),
        Some(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex
        })
    );
    let same = inference::infer(
        &request(column.clone(), single.clone(), Some("same")),
        &CONV_CATALOG_ENTRY,
    );
    assert_eq!(
        same.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    let valid = inference::infer(&request(column, single, Some("valid")), &CONV_CATALOG_ENTRY);
    assert_eq!(
        valid.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
}

#[test]
fn invalid_matrix_and_selector_do_not_prove_output() {
    let row = numeric(NumericClass::Double, NumericDomain::Real, &[1, 3]);
    let matrix = numeric(NumericClass::Double, NumericDomain::Real, &[2, 2]);
    let bad_matrix = inference::infer(&request(row.clone(), matrix, None), &CONV_CATALOG_ENTRY);
    assert!(!bad_matrix.diagnostics.is_empty());
    assert_eq!(bad_matrix.outputs[0].kind, ValueKindFact::Unknown);
    let bad_selector = inference::infer(
        &request(row.clone(), row, Some("circular")),
        &CONV_CATALOG_ENTRY,
    );
    assert!(!bad_selector.diagnostics.is_empty());
    assert_eq!(bad_selector.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn valid_empty_second_vector_keeps_first_orientation_and_length() {
    let column = numeric(NumericClass::Double, NumericDomain::Real, &[4, 1]);
    let empty = numeric(NumericClass::Double, NumericDomain::Real, &[0, 1]);
    let fact = inference::infer(&request(column, empty, Some("valid")), &CONV_CATALOG_ENTRY);
    assert!(fact.diagnostics.is_empty());
    assert_eq!(
        fact.outputs[0].shape,
        ShapeFact::from(vec![Some(4), Some(1)])
    );
}

#[test]
fn canonical_descriptor_retains_original_signatures_errors_and_integer_scope() {
    let labels = CONV_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect::<Vec<_>>();
    assert_eq!(labels, vec!["C = conv(A, B)", "C = conv(A, B, shape)"]);
    assert!(CONV_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == "RM.CONV.SHAPE_INVALID"));
    assert_eq!(CONV_INTEGER_CAPABILITIES.len(), 1);
    assert!(CONV_INTEGER_CAPABILITIES[0]
        .inputs
        .iter()
        .all(|input| input.classes.len() == 8));
}
