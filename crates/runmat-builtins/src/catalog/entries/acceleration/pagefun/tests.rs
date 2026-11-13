use super::*;
use runmat_types::*;

fn request(left: Vec<Option<usize>>, right: Vec<Option<usize>>) -> CallRequest {
    let array = |shape| {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(shape),
            StorageFact::Dense,
        )
    };
    CallRequest {
        arguments: vec![
            ValueFact::unknown(DynamicReason::RuntimeValue),
            array(left),
            array(right),
        ],
        literals: LiteralContext::new(vec![LiteralValue::String("mtimes".into())]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}
#[test]
fn page_broadcast_preserves_numeric_facts_and_empty_extents() {
    let req = request(
        vec![Some(2), Some(3), Some(0)],
        vec![Some(3), Some(4), Some(1)],
    );
    let result = inference::infer(&req, &PAGEFUN_CATALOG_ENTRY);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(4), Some(0)])
    );
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Single
    );
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
}
#[test]
fn unknown_shapes_classes_and_resident_results_stay_conservative() {
    let mut req = request(vec![None, Some(1)], vec![Some(2), Some(2)]);
    req.arguments[1].kind = ValueKindFact::Unknown;
    req.arguments[2].residency = ResidencyFact::Device { provider: None };
    let result = inference::infer(&req, &PAGEFUN_CATALOG_ENTRY);
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
}
#[test]
fn unsupported_selector_page_mismatch_and_output_arity_are_diagnosed() {
    let mut req = request(
        vec![Some(2), Some(2), Some(0)],
        vec![Some(2), Some(2), Some(2)],
    );
    assert!(!inference::infer(&req, &PAGEFUN_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
    req.literals.literal_args[0] = LiteralValue::String("plus".into());
    req.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
    assert!(
        inference::infer(&req, &PAGEFUN_CATALOG_ENTRY)
            .diagnostics
            .len()
            >= 3
    );
}
