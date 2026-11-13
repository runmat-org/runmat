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
        assert_eq!(output.storage, StorageFact::Dense);
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
    assert!(invalid
        .outputs
        .iter()
        .all(|fact| matches!(fact.kind, ValueKindFact::Unknown)));

    let mut sparse = request(NumericClass::Double, 2, 1);
    sparse.arguments[1].storage = StorageFact::Sparse;
    let invalid = inference::infer(&sparse, &FZERO_CATALOG_ENTRY);
    assert!(!invalid.diagnostics.is_empty());
    assert!(matches!(invalid.outputs[0].kind, ValueKindFact::Unknown));
}

#[test]
fn callback_capabilities_and_known_text_geometry_are_preserved() {
    let mut known = request(NumericClass::Double, 1, 1);
    known.arguments[0] = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: CapabilitySet(std::collections::BTreeSet::from([
            CapabilityRequirement::Filesystem,
        ])),
        parameters: vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }))],
        parameters_complete: true,
        outputs: vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }))],
        outputs_complete: true,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: vec![],
        captures_complete: true,
    }));
    let inference = inference::infer(&known, &FZERO_CATALOG_ENTRY);
    assert!(inference.diagnostics.is_empty());
    assert!(inference
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));

    let mut unknown_initial = known.clone();
    unknown_initial.arguments[1] = ValueFact::unknown(DynamicReason::RuntimeValue);
    let ValueKindFact::Callable(callable) = &mut unknown_initial.arguments[0].kind else {
        unreachable!("known callable fixture");
    };
    callable.parameters[0] = super::super::scalar_facts::host_scalar(NumericClass::Single);
    let unknown = inference::infer(&unknown_initial, &FZERO_CATALOG_ENTRY);
    assert!(unknown.diagnostics.is_empty());
    assert!(matches!(unknown.outputs[0].kind, ValueKindFact::Unknown));
    assert!(unknown
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));

    known.arguments[0] = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let invalid = inference::infer(&known, &FZERO_CATALOG_ENTRY);
    assert!(!invalid.diagnostics.is_empty());
    assert!(matches!(invalid.outputs[0].kind, ValueKindFact::Unknown));

    known.arguments[0] = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    assert!(!inference::infer(&known, &FZERO_CATALOG_ENTRY)
        .diagnostics
        .is_empty());
}
