use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, LiteralValue, ObjectFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn asset() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(super::super::GEOMETRY_ASSET_CLASS.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

fn codes(request: &CallRequest) -> Vec<String> {
    inference::infer(request, &GEOMETRY_REGION_CATALOG_ENTRY)
        .diagnostics
        .into_iter()
        .map(|diagnostic| diagnostic.code)
        .collect()
}

#[test]
fn region_returns_a_typed_revision_bound_object() {
    let result = inference::infer(
        &request(
            vec![asset(), ValueFact::scalar(ValueKindFact::String)],
            vec![],
        ),
        &GEOMETRY_REGION_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("expected a geometry.Region object");
    };
    assert_eq!(
        object.runtime_class.as_ref(),
        Some(&GEOMETRY_REGION_CLASS.owned())
    );
    for property in ["geometry_id", "geometry_revision", "region_id", "name"] {
        assert!(object.properties.contains_key(property), "{property}");
    }
    assert!(!object.properties_complete);
}

#[test]
fn region_rejects_wrong_arity_kinds_and_empty_literals() {
    assert!(codes(&request(vec![asset()], vec![]))
        .contains(&"RM-CATALOG-GEOMETRY-REGION-ARITY".to_string()));
    assert!(codes(&request(
        vec![
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String)
        ],
        vec![]
    ))
    .contains(&"RM-CATALOG-GEOMETRY-REGION-ASSET".to_string()));
    assert!(codes(&request(
        vec![asset(), ValueFact::scalar(ValueKindFact::Logical)],
        vec![]
    ))
    .contains(&"RM-CATALOG-GEOMETRY-REGION-SELECTOR".to_string()));
    assert!(codes(&request(
        vec![asset(), ValueFact::scalar(ValueKindFact::String)],
        vec![LiteralValue::Unknown, LiteralValue::String(String::new())]
    ))
    .contains(&"RM-CATALOG-GEOMETRY-REGION-SELECTOR".to_string()));
}

#[test]
fn region_accepts_unknown_inputs() {
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert!(codes(&request(vec![unknown.clone(), unknown], vec![])).is_empty());
}

#[test]
fn region_entry_is_valid_catalog_content() {
    let errors = crate::validate_builtin_catalog(ENTRIES);
    assert!(errors.is_empty(), "{errors:?}");
}
