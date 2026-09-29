use super::*;
use runmat_types::{
    standard, LiteralContext, NumericClass, NumericDomain, NumericFact, ObjectFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn dlarray(data: ValueFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::DL_ARRAY.owned()),
            properties: BTreeMap::from([(crate::DLARRAY_DATA_PROPERTY.into(), data.clone())]),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        data.shape,
        StorageFact::Opaque,
    )
}

#[test]
fn traced_dlarray_target_preserves_known_data_class_and_shape() {
    let data = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let loss = dlarray(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Real,
    })));
    let request = CallRequest {
        arguments: vec![loss, dlarray(data.clone())],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    };
    let result = crate::infer_catalog_call(&DLGRADIENT_CATALOG_ENTRY, &request);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs.len(), 1);
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("dlarray gradient")
    };
    assert_eq!(object.runtime_class, Some(standard::DL_ARRAY.owned()));
    assert_eq!(object.properties[crate::DLARRAY_DATA_PROPERTY], data);
}

#[test]
fn gradient_rejects_excess_outputs_and_known_non_dlarray_loss() {
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let request = CallRequest {
        arguments: vec![scalar.clone(), dlarray(scalar)],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    let result = crate::infer_catalog_call(&DLGRADIENT_CATALOG_ENTRY, &request);
    assert!(result
        .diagnostics
        .iter()
        .any(|diag| diag.code == "RM-DL-DLGRADIENT-OUTPUTS"));
    assert!(result
        .diagnostics
        .iter()
        .any(|diag| diag.code == "RM-DL-DLGRADIENT-LOSS"));
}
