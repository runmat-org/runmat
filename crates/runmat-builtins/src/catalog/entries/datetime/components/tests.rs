use super::*;
use crate::*;
use runmat_types::{
    CallRequest, DimensionFact, EffectKind, LiteralContext, LiteralValue, NumericClass,
    NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> runmat_types::CallInference {
    let BuiltinInferenceRule::Identity(rule) = entry.contract.inference_rule else {
        panic!("calendar entries require identity-owned inference");
    };
    rule.infer(request, entry)
}

#[test]
fn calendar_inference_uses_its_bound_kind_not_a_display_name() {
    let mut entry = NOW_CATALOG_ENTRY;
    entry.identity.name = "month";
    let result = infer(&request(vec![], vec![]), &entry);
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert!(matches!(result.outputs[0].kind, ValueKindFact::Numeric(_)));
}

fn shape(rows: usize, cols: usize) -> ShapeFact {
    ShapeFact::Shaped {
        dims: vec![DimensionFact::Known(rows), DimensionFact::Known(cols)],
    }
}

#[test]
fn calendar_entries_keep_their_bindings_and_argument_dependent_clock_effects() {
    let mut registered = Vec::new();
    extend_entries(&mut registered);
    assert_eq!(registered.len(), 9);
    for entry in declarations::ENTRIES.iter().copied() {
        assert_eq!(entry.bindings, REQUIRED_DEFAULT_BINDING.as_slice());
        assert!(entry.contract.effects.contains(&EffectKind::Clock));
        assert_eq!(entry.placement.fusion, BuiltinFusionPolicy::Boundary);
        assert!(!entry.documentation.examples.is_empty());
    }
    for entry in second::ENTRIES.iter().copied() {
        assert_eq!(entry.bindings, REQUIRED_DEFAULT_BINDING.as_slice());
        assert!(!entry.contract.effects.contains(&EffectKind::Clock));
        assert!(!entry.documentation.examples.is_empty());
    }
    assert_eq!(YEAR_CATALOG_ENTRY.descriptor.signatures.len(), 2);
    assert_eq!(YEAR_CATALOG_ENTRY.extensions, &YEAR_EXTENSIONS);
    assert_eq!(
        YEAR_CATALOG_ENTRY.integer_capabilities[0].form,
        YEAR_INTEGER_CAPABILITIES[0].form
    );
    assert_eq!(DAY_CATALOG_ENTRY.extensions.len(), 2);
    for entry in [
        &NOW_CATALOG_ENTRY,
        &TODAY_CATALOG_ENTRY,
        &CLOCK_CATALOG_ENTRY,
    ] {
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
    }
    for entry in [
        &YEAR_CATALOG_ENTRY,
        &MONTH_CATALOG_ENTRY,
        &HOUR_CATALOG_ENTRY,
        &MINUTE_CATALOG_ENTRY,
    ] {
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(entry.extensions.len(), 3);
        assert_eq!(entry.integer_capabilities.len(), 1);
        assert_eq!(entry.integer_capabilities[0].inputs[0].classes.len(), 8);
        assert_eq!(
            entry.integer_capabilities[0].inputs[0].availability,
            BuiltinIntegerInputAvailability::RunMatOnly
        );
    }
    assert_eq!(
        DAY_INTEGER_CAPABILITIES[0].inputs[0].availability,
        BuiltinIntegerInputAvailability::Documented
    );
    assert_eq!(
        DAY_CATALOG_ENTRY.documentation.authority,
        BuiltinDocumentationAuthority::Catalog
    );
}

#[test]
fn current_calendar_output_class_shape_and_arity_are_exact() {
    let empty = request(vec![], vec![]);
    let now = infer(&empty, &NOW_CATALOG_ENTRY);
    assert_eq!(now.outputs[0].shape, ShapeFact::Scalar);
    assert!(matches!(
        now.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    ));
    let clock = infer(&empty, &CLOCK_CATALOG_ENTRY);
    assert_eq!(clock.outputs[0].shape, shape(1, 6));
    let today = infer(&empty, &TODAY_CATALOG_ENTRY);
    assert_eq!(today.outputs[0].shape, ShapeFact::Scalar);
    assert!(matches!(
        today.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    ));
    let text = ValueFact::scalar(ValueKindFact::String);
    let today = infer(
        &request(
            vec![text.clone()],
            vec![LiteralValue::String("datetime".into())],
        ),
        &TODAY_CATALOG_ENTRY,
    );
    assert_eq!(today.outputs[0].shape, shape(1, 1));
    assert!(
        matches!(&today.outputs[0].kind, ValueKindFact::Object(object)
        if object.runtime_class.as_ref() == Some(&runmat_types::standard::DATETIME.owned()))
    );
    for entry in [&NOW_CATALOG_ENTRY, &CLOCK_CATALOG_ENTRY] {
        let invalid = request(vec![ValueFact::scalar(ValueKindFact::Unknown)], vec![]);
        assert!(infer(&invalid, entry)
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-DATETIME-COMPONENT-ARITY"));
    }
    let dynamic = infer(&request(vec![text.clone()], vec![]), &TODAY_CATALOG_ENTRY);
    assert!(matches!(dynamic.outputs[0].kind, ValueKindFact::Unknown));
    assert!(dynamic.diagnostics.is_empty());
    let serial = infer(
        &request(
            vec![text.clone()],
            vec![LiteralValue::Character("datenum".into())],
        ),
        &TODAY_CATALOG_ENTRY,
    );
    assert!(matches!(serial.outputs[0].kind, ValueKindFact::Numeric(_)));
    let invalid = infer(
        &request(
            vec![text.clone()],
            vec![LiteralValue::String("unsupported".into())],
        ),
        &TODAY_CATALOG_ENTRY,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TODAY-OUTPUT-TYPE"));
    let invalid = infer(
        &request(vec![text.clone(), text], vec![]),
        &TODAY_CATALOG_ENTRY,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DATETIME-COMPONENT-ARITY"));
}

#[test]
fn migrated_component_documentation_keeps_original_examples_and_asserts_each_program() {
    for (entry, count) in [
        (&YEAR_CATALOG_ENTRY, 3),
        (&MONTH_CATALOG_ENTRY, 3),
        (&DAY_CATALOG_ENTRY, 5),
        (&HOUR_CATALOG_ENTRY, 3),
        (&MINUTE_CATALOG_ENTRY, 3),
    ] {
        assert_eq!(entry.documentation.examples.len(), count);
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert!(!entry.documentation.evidence.verification.is_empty());
        for example in entry.documentation.examples {
            assert_eq!(example.harness, BuiltinExampleHarness::Portable);
            assert_eq!(example.fixture, BuiltinExampleFixture::None);
            assert!(!example.program.trim().is_empty());
            let BuiltinExampleVerification::Assertions { source } = example.verification else {
                panic!("component examples require executable assertions");
            };
            assert!(!source.trim().is_empty());
        }
    }
}

#[test]
fn component_shapes_and_name_modes_do_not_claim_numeric_outputs() {
    let today = infer(
        &request(
            vec![ValueFact::scalar(ValueKindFact::String)],
            vec![LiteralValue::String("datetime".into())],
        ),
        &TODAY_CATALOG_ENTRY,
    )
    .outputs
    .remove(0);
    let name = ValueFact::scalar(ValueKindFact::String);
    for entry in [&MONTH_CATALOG_ENTRY, &DAY_CATALOG_ENTRY] {
        let result = infer(
            &request(
                vec![today.clone(), name.clone()],
                vec![LiteralValue::Unknown, LiteralValue::String("name".into())],
            ),
            entry,
        );
        assert!(matches!(result.outputs[0].kind, ValueKindFact::Cell(_)));
        assert_eq!(result.outputs[0].shape, shape(1, 1));
        let unknown = infer(&request(vec![today.clone(), name.clone()], vec![]), entry);
        assert!(matches!(unknown.outputs[0].kind, ValueKindFact::Unknown));
    }
    let serials = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape(2, 3),
        StorageFact::Dense,
    );
    let result = infer(&request(vec![serials], vec![]), &YEAR_CATALOG_ENTRY);
    assert_eq!(result.outputs[0].shape, shape(2, 3));
    let text = ValueFact::proven(ValueKindFact::Character, shape(4, 10), StorageFact::Dense);
    let result = infer(&request(vec![text], vec![]), &YEAR_CATALOG_ENTRY);
    assert_eq!(result.outputs[0].shape, shape(4, 1));
}
