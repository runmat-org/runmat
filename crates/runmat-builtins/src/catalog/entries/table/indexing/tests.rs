use super::*;
use runmat_types::*;

fn call(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn receiver(class: StaticClassIdentity) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn path() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Struct(StructFact::array(
            Default::default(),
            false,
            Vec::new(),
            false,
        )),
        ShapeFact::Unknown,
        StorageFact::Opaque,
    )
}

#[test]
fn four_methods_have_distinct_canonical_bindings_and_signatures() {
    assert_eq!(ENTRIES.len(), 4);
    for entry in ENTRIES {
        assert_eq!(entry.bindings, &REQUIRED_DEFAULT_BINDING);
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(
            entry.descriptor.completion_policy,
            BuiltinCompletionPolicy::MethodOnly
        );
        assert!(entry.descriptor.signatures[0]
            .label
            .contains(entry.identity.name));
        assert!(entry.documentation.examples.iter().all(|example| matches!(
            example.verification,
            BuiltinExampleVerification::Assertions { .. }
        )));
    }
}

#[test]
fn read_paths_remain_dynamic_and_assignment_keeps_only_known_class() {
    for (read, write, class) in [
        (
            &table_subsref::ENTRY,
            &table_subsasgn::ENTRY,
            standard::TABLE,
        ),
        (
            &dictionary_subsref::ENTRY,
            &dictionary_subsasgn::ENTRY,
            standard::DICTIONARY,
        ),
    ] {
        let input = receiver(class);
        let read_result = infer_catalog_call(read, &call(vec![input.clone(), path()], 1));
        assert!(read_result.diagnostics.is_empty());
        assert_eq!(read_result.outputs[0].kind, ValueKindFact::Unknown);
        assert_eq!(read_result.outputs[0].residency, ResidencyFact::Unknown);

        let write_result = infer_catalog_call(
            write,
            &call(
                vec![
                    input,
                    path(),
                    ValueFact::unknown(DynamicReason::RuntimeValue),
                ],
                1,
            ),
        );
        assert!(write_result.diagnostics.is_empty());
        let ValueKindFact::Object(object) = &write_result.outputs[0].kind else {
            panic!("assignment should preserve known receiver class");
        };
        assert_eq!(object.runtime_class, Some(class.owned()));
        assert!(object.properties.is_empty());
        assert!(!object.properties_complete);
        assert_eq!(write_result.outputs[0].shape, ShapeFact::Scalar);
    }
}

#[test]
fn unknown_and_invalid_inputs_do_not_produce_invented_output_facts() {
    for entry in ENTRIES {
        let assignment = entry.identity.name.ends_with("subsasgn");
        let mut unknown_args = vec![ValueFact::unknown(DynamicReason::RuntimeValue), path()];
        if assignment {
            unknown_args.push(ValueFact::unknown(DynamicReason::RuntimeValue));
        }
        let unknown = infer_catalog_call(entry, &call(unknown_args, 1));
        assert_eq!(unknown.outputs[0].kind, ValueKindFact::Unknown);
        assert!(!infer_catalog_call(entry, &call(Vec::new(), 1))
            .diagnostics
            .is_empty());

        let class = if entry.identity.name.starts_with("table.") {
            standard::TABLE
        } else {
            standard::DICTIONARY
        };
        let mut invalid_args = vec![
            receiver(class),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        ];
        if assignment {
            invalid_args.push(ValueFact::scalar(ValueKindFact::String));
        }
        let invalid = infer_catalog_call(entry, &call(invalid_args, 1));
        assert!(invalid
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-TABULAR-SUBSCRIPT-PATH"));
        assert!(!infer_catalog_call(entry, &call(vec![receiver(class)], 1))
            .diagnostics
            .is_empty());
    }
}

#[test]
fn table_indexing_admits_timetable_receiver_without_changing_its_identity() {
    let result = infer_catalog_call(
        &table_subsasgn::ENTRY,
        &call(
            vec![
                receiver(standard::TIMETABLE),
                path(),
                ValueFact::scalar(ValueKindFact::String),
            ],
            1,
        ),
    );
    assert!(result.diagnostics.is_empty());
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("expected timetable");
    };
    assert_eq!(object.runtime_class, Some(standard::TIMETABLE.owned()));
    assert!(!infer_catalog_call(
        &dictionary_subsref::ENTRY,
        &call(vec![receiver(standard::TABLE), path()], 1),
    )
    .diagnostics
    .is_empty());
}
