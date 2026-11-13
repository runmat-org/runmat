use super::*;
use crate::BuiltinOutputMode;
use runmat_types::*;

fn numeric(class: NumericClass, domain: NumericDomain, dims: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact { class, domain }),
        ShapeFact::from(dims.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}

fn request(left: ValueFact, right: ValueFact, outputs: usize) -> CallRequest {
    CallRequest {
        arguments: vec![left, right],
        literals: LiteralContext::new(vec![LiteralValue::Unknown; 2]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn quotient_unknown_degree_but_remainder_retains_numerator_shape_and_class() {
    let numerator = numeric(NumericClass::Single, NumericDomain::Complex, &[1, 4]);
    let denominator = numeric(NumericClass::Double, NumericDomain::Real, &[1, 2]);
    let facts = inference::infer(&request(numerator, denominator, 2), &DECONV_CATALOG_ENTRY);
    assert!(facts.diagnostics.is_empty());
    assert_eq!(facts.outputs.len(), 2);
    assert_eq!(facts.outputs[0].shape, ShapeFact::from(vec![Some(1), None]));
    assert_eq!(
        facts.outputs[1].shape,
        ShapeFact::from(vec![Some(1), Some(4)])
    );
    assert_eq!(
        facts.outputs[0].numeric(),
        Some(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex
        })
    );
}

#[test]
fn integer_or_logical_coefficient_selects_double_and_invalid_forms_do_not_prove() {
    let numerator = numeric(NumericClass::Single, NumericDomain::Real, &[4, 1]);
    let logical = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(2), Some(1)]),
        StorageFact::Dense,
    );
    let facts = inference::infer(
        &request(numerator.clone(), logical, 2),
        &DECONV_CATALOG_ENTRY,
    );
    assert_eq!(
        facts.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        facts.outputs[1].shape,
        ShapeFact::from(vec![Some(4), Some(1)])
    );
    let matrix = numeric(NumericClass::Double, NumericDomain::Real, &[2, 2]);
    let invalid = inference::infer(&request(numerator, matrix, 1), &DECONV_CATALOG_ENTRY);
    assert!(!invalid.diagnostics.is_empty());
    assert_eq!(invalid.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn known_zero_denominator_and_excess_outputs_are_diagnosed() {
    let mut zero = request(
        numeric(NumericClass::Double, NumericDomain::Real, &[1, 2]),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        2,
    );
    zero.literals = LiteralContext::new(vec![LiteralValue::Unknown, LiteralValue::Number(0.0)]);
    let invalid = inference::infer(&zero, &DECONV_CATALOG_ENTRY);
    assert!(!invalid.diagnostics.is_empty());
    assert!(invalid
        .outputs
        .iter()
        .all(|fact| fact.kind == ValueKindFact::Unknown));
    let excess = inference::infer(
        &request(
            numeric(NumericClass::Double, NumericDomain::Real, &[1, 2]),
            numeric(NumericClass::Double, NumericDomain::Real, &[1, 2]),
            3,
        ),
        &DECONV_CATALOG_ENTRY,
    );
    assert!(!excess.diagnostics.is_empty());
    // The shared call contract retains a fact for each requested slot even
    // when the output count is invalid; no slot may claim a produced value.
    assert_eq!(excess.outputs.len(), 3);
    assert!(excess
        .outputs
        .iter()
        .all(|fact| fact.kind == ValueKindFact::Unknown));
    assert!(excess
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-DECONV-OUTPUTS"));
}

#[test]
fn known_typed_complex_integer_coefficients_are_not_proven() {
    let real = numeric(NumericClass::Double, NumericDomain::Real, &[1, 2]);
    let complex_integer = numeric(NumericClass::Int32, NumericDomain::Complex, &[1, 2]);
    for (numerator, denominator) in [
        (complex_integer.clone(), real.clone()),
        (real.clone(), complex_integer.clone()),
    ] {
        let facts = inference::infer(&request(numerator, denominator, 2), &DECONV_CATALOG_ENTRY);
        assert!(!facts.diagnostics.is_empty());
        assert!(facts
            .outputs
            .iter()
            .all(|fact| fact.kind == ValueKindFact::Unknown));
    }
}

#[test]
fn canonical_metadata_retains_requested_output_and_error_contract() {
    let labels = DECONV_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect::<Vec<_>>();
    assert_eq!(
        labels,
        vec![
            "Q = deconv(numerator, denominator)",
            "[Q, R] = deconv(numerator, denominator)"
        ]
    );
    assert_eq!(
        DECONV_DESCRIPTOR.output_mode,
        BuiltinOutputMode::ByRequestedOutputCount
    );
    assert!(DECONV_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == "RM.DECONV.DENOMINATOR_INVALID"));
    assert!(DECONV_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.code == "RM.DECONV.TOO_MANY_OUTPUTS"));
    assert!(DECONV_DESCRIPTOR.errors.iter().any(|error| {
        error.code == "RM.DECONV.LATE_UPLOAD_ADMISSION"
            && error.identifier == Some("RunMat:Placement:LateUploadAdmissionFailed")
    }));
}
