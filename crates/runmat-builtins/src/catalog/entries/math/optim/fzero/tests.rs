use super::*;
use runmat_types::*;

fn request(class: NumericClass, count: usize, outputs: usize) -> CallRequest {
    CallRequest {
        arguments: vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(1), Some(count)]),
                StorageFact::Dense,
            ),
        ],
        literals: LiteralContext::new(vec![]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn bracket_and_scalar_precision_facts_follow_initial_input() {
    let single = inference::infer(&request(NumericClass::Single, 2, 4), &FZERO_CATALOG_ENTRY);
    assert!(single.diagnostics.is_empty());
    for output in &single.outputs[..3] {
        assert_eq!(output.numeric().unwrap().class, NumericClass::Single);
        assert_eq!(output.shape, ShapeFact::Scalar);
    }
    assert!(matches!(single.outputs[3].kind, ValueKindFact::Struct(_)));
    let double = inference::infer(&request(NumericClass::Double, 1, 1), &FZERO_CATALOG_ENTRY);
    assert_eq!(
        double.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn malformed_initial_geometry_and_excessive_outputs_do_not_prove_success() {
    let invalid = inference::infer(&request(NumericClass::Double, 3, 5), &FZERO_CATALOG_ENTRY);
    assert!(!invalid.diagnostics.is_empty());
}
