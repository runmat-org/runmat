use super::*;
use runmat_types::*;

fn call(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

fn numeric() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(4), Some(2)]),
        StorageFact::Dense,
    )
}

fn table() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::TABLE.owned()),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

#[test]
fn grpstats_has_one_canonical_binding_and_asserted_examples() {
    assert_eq!(ENTRY.identity.name, "grpstats");
    assert_eq!(ENTRY.category, "stats/summary");
    assert_eq!(ENTRY.bindings, &REQUIRED_DEFAULT_BINDING);
    assert_eq!(
        ENTRY.documentation.authority,
        BuiltinDocumentationAuthority::Catalog
    );
    assert_eq!(
        ENTRY.descriptor.output_mode,
        BuiltinOutputMode::ByRequestedOutputCount
    );
    assert_eq!(ENTRY.extensions.len(), 4);
    assert_eq!(ENTRY.integer_capabilities.len(), 3);
    assert_eq!(
        ENTRY.placement.accelerator,
        BuiltinAcceleratorPolicy::Forbidden
    );
    assert_eq!(ENTRY.placement.fusion, BuiltinFusionPolicy::Boundary);
    assert!(ENTRY.documentation.examples.iter().all(|example| matches!(
        example.verification,
        BuiltinExampleVerification::Assertions { .. }
    )));
}

#[test]
fn table_input_infers_table_identity_without_schema() {
    let result = infer_catalog_call(&ENTRY, &call(vec![table(), unknown()], vec![], 1));
    assert!(result.diagnostics.is_empty());
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("expected grouped table");
    };
    assert_eq!(object.runtime_class, Some(standard::TABLE.owned()));
    assert!(!object.properties_complete);
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    let extra = infer_catalog_call(&ENTRY, &call(vec![table(), unknown()], vec![], 2));
    assert!(!extra.diagnostics.is_empty());
}

#[test]
fn matrix_default_and_literal_statistics_have_precise_output_classes() {
    let default = infer_catalog_call(&ENTRY, &call(vec![numeric(), unknown()], vec![], 1));
    assert!(default.diagnostics.is_empty());
    assert!(matches!(
        default.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            ..
        })
    ));
    let interval = infer_catalog_call(
        &ENTRY,
        &call(
            vec![numeric(), unknown(), unknown()],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("meanci".into()),
            ],
            1,
        ),
    );
    assert!(interval.diagnostics.is_empty());
    assert_eq!(
        interval.outputs[0].shape.known_dims(),
        Some(vec![None, None, Some(2)])
    );
    let names = infer_catalog_call(
        &ENTRY,
        &call(
            vec![numeric(), unknown(), unknown()],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("gname".into()),
            ],
            1,
        ),
    );
    assert!(names.diagnostics.is_empty());
    assert!(matches!(names.outputs[0].kind, ValueKindFact::Cell(_)));
}

#[test]
fn known_statistic_list_checks_output_count_without_guessing_unknown_lists() {
    let args = vec![numeric(), unknown(), unknown()];
    let literals = vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::Vector(vec![
            LiteralValue::String("mean".into()),
            LiteralValue::String("gname".into()),
        ]),
    ];
    let good = infer_catalog_call(&ENTRY, &call(args.clone(), literals.clone(), 2));
    assert!(good.diagnostics.is_empty());
    assert!(matches!(good.outputs[0].kind, ValueKindFact::Numeric(_)));
    assert!(matches!(good.outputs[1].kind, ValueKindFact::Cell(_)));
    let bad = infer_catalog_call(&ENTRY, &call(args.clone(), literals, 1));
    assert!(bad
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-GRPSTATS-OUTPUT-COUNT"));
    let dynamic = infer_catalog_call(&ENTRY, &call(args, vec![], 2));
    assert!(dynamic.diagnostics.is_empty());
    assert!(dynamic
        .outputs
        .iter()
        .all(|output| matches!(output.kind, ValueKindFact::Unknown)));
}

#[test]
fn missing_data_argument_is_diagnosed() {
    let result = infer_catalog_call(&ENTRY, &call(vec![numeric()], vec![], 1));
    assert!(!result.diagnostics.is_empty());
}
