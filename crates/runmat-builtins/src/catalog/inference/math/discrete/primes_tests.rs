use crate::builtin_catalog_entry_by_name;
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

use super::primes::infer;

fn request(input: ValueFact) -> CallRequest {
    CallRequest {
        arguments: vec![input],
        literals: LiteralContext::new(Vec::new()),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn primes_preserves_class_and_returns_a_dynamic_row() {
    let input = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    let entry = builtin_catalog_entry_by_name("primes").expect("primes entry");
    let inferred = infer(&request(input), entry);
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].numeric().unwrap().class,
        NumericClass::UInt64
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
    assert_eq!(inferred.outputs[0].storage, StorageFact::Dense);
}

#[test]
fn primes_rejects_a_known_nonscalar_input() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    let entry = builtin_catalog_entry_by_name("primes").expect("primes entry");
    assert!(!infer(&request(input), entry).diagnostics.is_empty());
}
