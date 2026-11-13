use super::*;
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[test]
fn ss2tf_catalog_exposes_selected_input_and_two_outputs() {
    let entry = &SS2TF_CATALOG_ENTRY;
    assert_eq!(entry.identity.name, "ss2tf");
    assert_eq!(entry.descriptor.signatures.len(), 2);
    let request = CallRequest {
        arguments: Vec::new(),
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
    };
    assert!(!inference::infer(&request, entry).diagnostics.is_empty());
}

#[test]
fn all_single_matrix_facts_retained_for_selected_input_outputs() {
    let matrix = |rows, cols| {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(rows), Some(cols)]),
            StorageFact::Dense,
        )
    };
    let request = CallRequest {
        arguments: vec![matrix(2, 2), matrix(2, 3), matrix(4, 2), matrix(4, 3)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    let result = inference::infer(&request, &SS2TF_CATALOG_ENTRY);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(4), Some(3)])
    );
    assert_eq!(
        result.outputs[1].shape,
        ShapeFact::from(vec![Some(1), Some(3)])
    );
    assert_eq!(
        result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })
    );
}
