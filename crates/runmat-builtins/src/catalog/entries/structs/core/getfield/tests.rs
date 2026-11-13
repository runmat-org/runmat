use super::GETFIELD_CATALOG_ENTRY;
use crate::catalog::infer_catalog_call;
use runmat_types::{
    CallRequest, CellFact, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, StructFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

fn numeric(class: NumericClass, shape: Vec<Option<usize>>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape),
        StorageFact::Dense,
    )
}

fn selector(element: ValueFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element.clone()),
            elements: vec![element],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Dense,
    )
}
fn number() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

#[test]
fn infers_literal_nested_field() {
    let mut nested = BTreeMap::new();
    nested.insert("value".to_string(), number());
    let mut root = BTreeMap::new();
    root.insert(
        "inner".to_string(),
        ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(nested, true))),
    );
    let target = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(root, true)));
    let text = ValueFact::scalar(ValueKindFact::String);
    let inferred = infer_catalog_call(
        &GETFIELD_CATALOG_ENTRY,
        &request(
            vec![target, text.clone(), text],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("inner".into()),
                LiteralValue::String("value".into()),
            ],
        ),
    );
    assert_eq!(inferred.outputs, vec![number()]);
    assert!(inferred.diagnostics.is_empty());
}

#[test]
fn reports_a_missing_literal_field_on_complete_schema() {
    let target = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        BTreeMap::new(),
        true,
    )));
    let text = ValueFact::scalar(ValueKindFact::String);
    let inferred = infer_catalog_call(
        &GETFIELD_CATALOG_ENTRY,
        &request(
            vec![target, text],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("missing".into()),
            ],
        ),
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-MEMBER-MISSING"));
}

#[test]
fn typed_scalar_selector_preserves_the_selected_field_class() {
    let field = numeric(NumericClass::UInt16, vec![Some(4), Some(1)]);
    let mut root = BTreeMap::new();
    root.insert("samples".to_string(), field);
    let target = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(root, true)));
    let text = ValueFact::scalar(ValueKindFact::String);
    let index = selector(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt8,
        domain: NumericDomain::Real,
    })));
    let inferred = infer_catalog_call(
        &GETFIELD_CATALOG_ENTRY,
        &request(
            vec![target, text, index],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("samples".into()),
                LiteralValue::Unknown,
            ],
        ),
    );
    assert_eq!(
        inferred.outputs[0].numeric().map(|fact| fact.class),
        Some(NumericClass::UInt16)
    );
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Scalar);
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
}
