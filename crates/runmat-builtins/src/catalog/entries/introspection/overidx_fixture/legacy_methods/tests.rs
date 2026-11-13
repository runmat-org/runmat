use super::*;
use runmat_types::*;

fn infer(entry: &BuiltinCatalogEntry, arguments: Vec<ValueFact>, count: usize) -> CallInference {
    let BuiltinInferenceRule::Identity(rule) = entry.contract.inference_rule else {
        panic!("fixture owns identity inference")
    };
    rule.infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        },
        entry,
    )
}

fn object() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(StaticClassIdentity::new("OverIdx").owned()),
        properties: Default::default(),
        properties_complete: true,
        handle_semantics: Some(false),
    }))
}

#[test]
fn every_fixture_keeps_method_policy_and_single_outer_output() {
    for entry in ENTRIES {
        assert_eq!(entry.audience, BuiltinCatalogAudience::Fixture);
        assert_eq!(
            entry.descriptor.completion_policy,
            BuiltinCompletionPolicy::MethodOnly
        );
        let arguments = vec![
            ValueFact::unknown(DynamicReason::DynamicDispatch);
            entry.descriptor.signatures[0].inputs.len()
        ];
        assert!(
            infer(entry, arguments.clone(), 1).diagnostics.is_empty(),
            "{}",
            entry.identity.name
        );
        assert!(
            !infer(entry, arguments, 2).diagnostics.is_empty(),
            "{}",
            entry.identity.name
        );
    }
}

#[test]
fn nargs_property_is_not_falsely_inferred_numeric() {
    let mut receiver = object();
    let text = ValueFact::scalar(ValueKindFact::String);
    if let ValueKindFact::Object(object) = &mut receiver.kind {
        object.properties.insert("nargs".into(), text.clone());
    }
    let result = infer(
        &OVERIDX_NUM_ARGUMENTS_CATALOG_ENTRY,
        vec![receiver, text.clone(), text.clone()],
        1,
    );
    assert_eq!(result.outputs[0], text);
}

#[test]
fn unary_identity_preserves_arbitrary_fact_and_comparisons_are_double() {
    let text = ValueFact::scalar(ValueKindFact::String);
    assert_eq!(
        infer(&OVERIDX_UPLUS_CATALOG_ENTRY, vec![text.clone()], 1).outputs[0],
        text
    );
    let result = infer(
        &OVERIDX_EQ_CATALOG_ENTRY,
        vec![
            object(),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            })),
        ],
        1,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn loadobj_accepts_nonstruct_and_saveobj_preserves_property_class() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let result = infer(&OVERIDX_LOADOBJ_CATALOG_ENTRY, vec![text.clone()], 1);
    assert!(result.diagnostics.is_empty());
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("OverIdx object")
    };
    assert!(object.properties_complete);
    assert!(object.properties.contains_key("loaded_by"));
    let mut receiver = self::object();
    if let ValueKindFact::Object(object) = &mut receiver.kind {
        object.properties.insert("nargs".into(), text.clone());
    }
    let saved = infer(&OVERIDX_SAVEOBJ_CATALOG_ENTRY, vec![receiver], 1);
    let ValueKindFact::Struct(payload) = &saved.outputs[0].kind else {
        panic!("serialized struct")
    };
    assert_eq!(payload.fields["nargs"], text);
}

#[test]
fn semantic_routes_do_not_depend_on_entry_spelling() {
    for entry in ENTRIES {
        let arguments = vec![
            ValueFact::unknown(DynamicReason::DynamicDispatch);
            entry.descriptor.signatures[0].inputs.len()
        ];
        let mut renamed = **entry;
        renamed.identity = BuiltinCatalogIdentity {
            name: "deliberately_not_a_fixture_identity",
        };
        assert_eq!(
            infer(entry, arguments.clone(), 1),
            infer(&renamed, arguments, 1)
        );
    }
}
