use super::*;
use crate::{infer_catalog_call, BuiltinDocumentationAuthority};
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[test]
fn seven_degree_inverse_identities_have_catalog_examples_and_typed_shapes() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let request = CallRequest {
        arguments: vec![input.clone()],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    for entry in ENTRIES {
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert!(!entry.documentation.examples.is_empty());
        let result = infer_catalog_call(entry, &request);
        assert!(result.diagnostics.is_empty(), "{}", entry.identity.name);
        assert_eq!(
            result.outputs[0].shape, input.shape,
            "{}",
            entry.identity.name
        );
    }
    let binary = CallRequest {
        arguments: vec![input.clone(), input],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = infer_catalog_call(&crate::ATAN2D_CATALOG_ENTRY, &binary);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
}
