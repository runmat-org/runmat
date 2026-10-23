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

#[test]
fn every_fixture_owns_fixed_input_and_output_counts() {
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
fn shape_area_ignores_receiver_and_origin_owns_exact_properties() {
    let area = infer(
        &SHAPE_AREA_CATALOG_ENTRY,
        vec![ValueFact::scalar(ValueKindFact::String)],
        1,
    );
    assert!(area.diagnostics.is_empty());
    assert_eq!(
        area.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    let origin = infer(&POINT_ORIGIN_CATALOG_ENTRY, vec![], 1);
    let ValueKindFact::Object(object) = &origin.outputs[0].kind else {
        panic!("Point object")
    };
    assert_eq!(
        object.runtime_class,
        Some(StaticClassIdentity::new("Point").owned())
    );
    assert!(object.properties_complete);
    assert_eq!(object.properties.len(), 2);
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
