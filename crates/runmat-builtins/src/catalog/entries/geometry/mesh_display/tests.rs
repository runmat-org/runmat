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

fn codes(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> Vec<String> {
    infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
        &GEOMETRY_MESH_METHOD_CATALOG_ENTRY,
    )
    .diagnostics
    .into_iter()
    .map(|diagnostic| diagnostic.code)
    .collect()
}

#[test]
fn display_accepts_asset_and_logical_options() {
    assert!(codes(
        vec![
            asset(),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical)
        ],
        vec![LiteralValue::Unknown, LiteralValue::String("XRay".into())]
    )
    .is_empty());
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    assert!(codes(vec![unknown.clone(), unknown.clone(), unknown], vec![]).is_empty());
}

#[test]
fn display_rejects_missing_asset_bad_pairs_names_and_values() {
    assert!(codes(vec![], vec![]).contains(&"RM-CATALOG-GEOMETRY-MESH-ARITY".to_string()));
    assert!(
        codes(vec![ValueFact::scalar(ValueKindFact::String)], vec![])
            .contains(&"RM-CATALOG-GEOMETRY-MESH-ASSET".to_string())
    );
    assert!(codes(
        vec![asset(), ValueFact::scalar(ValueKindFact::String)],
        vec![]
    )
    .contains(&"RM-CATALOG-GEOMETRY-MESH-OPTIONS".to_string()));
    assert!(codes(
        vec![
            asset(),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical)
        ],
        vec![LiteralValue::Unknown, LiteralValue::String("Color".into())]
    )
    .contains(&"RM-CATALOG-GEOMETRY-MESH-OPTION".to_string()));
    assert!(codes(
        vec![
            asset(),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String)
        ],
        vec![]
    )
    .contains(&"RM-CATALOG-GEOMETRY-MESH-LOGICAL".to_string()));
}

#[test]
fn display_entry_is_method_only_and_valid() {
    assert_eq!(
        GEOMETRY_MESH_METHOD_DESCRIPTOR.completion_policy,
        BuiltinCompletionPolicy::MethodOnly
    );
    let errors = crate::validate_builtin_catalog(ENTRIES);
    assert!(errors.is_empty(), "{errors:?}");
}
