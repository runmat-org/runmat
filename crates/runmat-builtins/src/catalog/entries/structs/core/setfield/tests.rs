use super::SETFIELD_CATALOG_ENTRY;
use crate::catalog::infer_catalog_call;
use runmat_types::{
    CallRequest, CellFact, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, StructFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

fn number() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
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
fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn adds_a_literal_field_to_a_complete_structure_fact() {
    let target = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        BTreeMap::new(),
        true,
    )));
    let text = ValueFact::scalar(ValueKindFact::String);
    let inferred = infer_catalog_call(
        &SETFIELD_CATALOG_ENTRY,
        &request(
            vec![target, text, number()],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("answer".into()),
                LiteralValue::Unknown,
            ],
        ),
    );
    let ValueKindFact::Struct(structure) = &inferred.outputs[0].kind else {
        panic!("expected struct fact")
    };
    assert_eq!(structure.fields.get("answer"), Some(&number()));
    assert!(inferred.diagnostics.is_empty());
}

#[test]
fn preserves_known_nested_schema() {
    let mut nested = BTreeMap::new();
    nested.insert("old".to_string(), number());
    let mut root = BTreeMap::new();
    root.insert(
        "inner".to_string(),
        ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(nested, true))),
    );
    let target = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(root, true)));
    let text = ValueFact::scalar(ValueKindFact::String);
    let inferred = infer_catalog_call(
        &SETFIELD_CATALOG_ENTRY,
        &request(
            vec![target, text.clone(), text, number()],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("inner".into()),
                LiteralValue::String("new".into()),
                LiteralValue::Unknown,
            ],
        ),
    );
    let ValueKindFact::Struct(root) = &inferred.outputs[0].kind else {
        panic!("expected struct")
    };
    let ValueKindFact::Struct(inner) = &root.fields["inner"].kind else {
        panic!("expected nested struct")
    };
    assert!(inner.fields.contains_key("old"));
    assert_eq!(inner.fields.get("new"), Some(&number()));
}

#[test]
fn typed_selector_updates_the_known_field_without_erasing_its_class() {
    let samples = numeric(NumericClass::UInt16, vec![Some(4), Some(1)]);
    let mut root = BTreeMap::new();
    root.insert("samples".to_string(), samples);
    let target = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(root, true)));
    let text = ValueFact::scalar(ValueKindFact::String);
    let index = selector(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt8,
        domain: NumericDomain::Real,
    })));
    let replacement = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt16,
        domain: NumericDomain::Real,
    }));
    let inferred = infer_catalog_call(
        &SETFIELD_CATALOG_ENTRY,
        &request(
            vec![target, text, index, replacement],
            vec![
                LiteralValue::Unknown,
                LiteralValue::String("samples".into()),
                LiteralValue::Unknown,
                LiteralValue::Unknown,
            ],
        ),
    );
    let ValueKindFact::Struct(output) = &inferred.outputs[0].kind else {
        panic!("expected structure")
    };
    assert_eq!(
        output.fields["samples"].numeric().map(|fact| fact.class),
        Some(NumericClass::UInt16)
    );
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
}
