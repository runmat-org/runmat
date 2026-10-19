use super::*;
use crate::{catalog::inference::infer_catalog_call, BuiltinCatalogEntry};
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, count: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    }
}

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_catalog_call(entry, request)
}

fn text() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}

#[test]
fn all_java_public_identities_have_distinct_catalog_entries_and_default_bindings() {
    let names: std::collections::BTreeSet<_> =
        ENTRIES.iter().map(|entry| entry.identity.name).collect();
    assert_eq!(names.len(), 11);
    for name in [
        "isjava",
        "javaaddpath",
        "javaArray",
        "javaclasspath",
        "javaMethod",
        "javaMethodEDT",
        "javaObject",
        "javaObjectEDT",
        "javarmpath",
        "jenv",
        "usejava",
    ] {
        assert!(names.contains(name), "missing {name}");
    }
    assert!(ENTRIES.iter().all(|entry| !entry.bindings.is_empty()));
}

#[test]
fn java_handle_and_environment_facts_preserve_session_identity() {
    let object = infer(&request(vec![text()], 1), &JAVA_OBJECT_CATALOG_ENTRY);
    assert!(object.diagnostics.is_empty(), "{:?}", object.diagnostics);
    let ValueKindFact::Foreign(reference) = &object.outputs[0].kind else {
        panic!("expected Java foreign reference")
    };
    assert_eq!(reference.family, "java");
    assert_eq!(reference.affinity, ForeignAffinityFact::OriginProcess);
    assert_eq!(reference.lifetime, ForeignLifetimeFact::Session);
    assert_eq!(object.outputs[0].alias, AliasFact::Identity);

    let environment = infer(&request(vec![], 1), &JENV_CATALOG_ENTRY);
    assert!(
        environment.diagnostics.is_empty(),
        "{:?}",
        environment.diagnostics
    );
    let ValueKindFact::Object(status) = &environment.outputs[0].kind else {
        panic!("expected Java environment object")
    };
    assert!(status
        .runtime_class
        .as_ref()
        .is_some_and(|identity| identity.is(JAVA_ENVIRONMENT_CLASS)));
    for property in ["Version", "Home", "Library", "Status", "Configuration"] {
        assert!(status.properties.contains_key(property));
    }
    assert!(!environment
        .effects
        .0
        .contains(&EffectKind::EnvironmentWrite));
}

#[test]
fn array_dimension_shape_and_classpath_forms_are_not_optimistically_inferred() {
    let matrix = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(2)]),
        StorageFact::Dense,
    );
    let invalid = infer(&request(vec![text(), matrix], 1), &JAVA_ARRAY_CATALOG_ENTRY);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-JAVA-DIMENSION"));

    let unknown = infer(&request(vec![text()], 1), &JAVA_CLASSPATH_CATALOG_ENTRY);
    assert!(unknown
        .outputs
        .iter()
        .all(|output| output.kind == ValueKindFact::Unknown));
    let query = infer(&request(vec![], 1), &JAVA_CLASSPATH_CATALOG_ENTRY);
    assert!(matches!(query.outputs[0].kind, ValueKindFact::Cell(_)));
    let bad_path = infer(
        &request(vec![ValueFact::scalar(ValueKindFact::Logical)], 0),
        &JAVA_ADD_PATH_CATALOG_ENTRY,
    );
    assert!(bad_path
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-JAVA-PATH"));
}

#[test]
fn local_isjava_and_feature_probe_do_not_invent_java_results() {
    let local = infer(&request(vec![text()], 1), &ISJAVA_CATALOG_ENTRY);
    assert!(local.capabilities.0.is_empty());
    assert!(matches!(local.outputs[0].kind, ValueKindFact::Logical));
    let feature = infer(&request(vec![text()], 1), &USEJAVA_CATALOG_ENTRY);
    assert!(matches!(feature.outputs[0].kind, ValueKindFact::Logical));
    assert!(feature
        .capabilities
        .0
        .contains(&CapabilityRequirement::ForeignRuntime));
}

#[test]
fn text_controls_reject_proven_matrix_pages_and_numeric_cell_entries() {
    let pages = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(1), Some(4)]),
        StorageFact::Dense,
    );
    let class = infer(&request(vec![pages], 1), &JAVA_OBJECT_CATALOG_ENTRY);
    assert!(class
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-JAVA-TEXT"));

    let number = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let cell = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::Unspecified)),
            elements: vec![number],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(1)]),
        StorageFact::Opaque,
    );
    let path = infer(&request(vec![cell], 0), &JAVA_ADD_PATH_CATALOG_ENTRY);
    assert!(path
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-JAVA-PATH"));
}

#[test]
fn feature_literals_use_the_same_typed_identity_as_runtime() {
    assert_eq!(JavaFeature::parse("sWiNg"), Some(JavaFeature::Swing));
    assert_eq!(JavaFeature::parse("unknown"), None);
}

#[test]
fn classpath_public_and_wire_controls_have_distinct_typed_boundaries() {
    assert_eq!(
        JavaClasspathControl::parse_public("-StAtIc"),
        JavaClasspathControl::Query(JavaClasspathQuery::Static)
    );
    assert_eq!(
        JavaClasspathControl::parse_public("-v1"),
        JavaClasspathControl::Verbosity
    );
    assert_eq!(
        JavaClasspathControl::parse_public("plugin.jar"),
        JavaClasspathControl::Path
    );
    assert_eq!(
        JavaClasspathQuery::parse_wire("static"),
        Some(JavaClasspathQuery::Static)
    );
    assert_eq!(
        JavaClasspathQuery::parse_wire("bootstrap"),
        Some(JavaClasspathQuery::Static)
    );
    assert_eq!(JavaClasspathQuery::parse_wire("-static"), None);
    assert_eq!(
        JavaInsertionPosition::from_public_terminal("-end"),
        Some(JavaInsertionPosition::End)
    );
    assert_eq!(JavaInsertionPosition::from_public_terminal("-END"), None);
    assert_eq!(
        JavaInsertionPosition::parse_wire("end"),
        Some(JavaInsertionPosition::End)
    );
    assert_eq!(JavaInsertionPosition::parse_wire("-end"), None);
}

#[test]
fn environment_control_names_share_the_runtime_boundary() {
    assert_eq!(
        JavaEnvironmentControl::parse("vErSiOn"),
        Some(JavaEnvironmentControl::Version)
    );
    assert_eq!(
        JavaEnvironmentControl::parse("EXECUTIONMODE"),
        Some(JavaEnvironmentControl::ExecutionMode)
    );
    assert_eq!(JavaEnvironmentControl::parse("Mode"), None);
    assert_eq!(
        JavaExecutionMode::parse("inPROCESS"),
        Some(JavaExecutionMode::InProcess)
    );
    assert_eq!(JavaExecutionMode::parse("OutOfProcess"), None);

    let mut valid = request(vec![text(), text()], 1);
    valid.literals = LiteralContext::new(vec![
        LiteralValue::String("EXECUTIONMODE".into()),
        LiteralValue::Unknown,
    ]);
    assert!(infer(&valid, &JENV_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .all(|diagnostic| diagnostic.code != "RM-JAVA-OPTION"));
    valid.literals = LiteralContext::new(vec![
        LiteralValue::String("Mode".into()),
        LiteralValue::Unknown,
    ]);
    assert!(infer(&valid, &JENV_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-JAVA-OPTION"));

    valid.literals = LiteralContext::new(vec![
        LiteralValue::String("ExecutionMode".into()),
        LiteralValue::String("OutOfProcess".into()),
    ]);
    assert!(infer(&valid, &JENV_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-JAVA-EXECUTION-MODE"));
    valid.literals = LiteralContext::new(vec![
        LiteralValue::String("ExecutionMode".into()),
        LiteralValue::String("inPROCESS".into()),
    ]);
    assert!(infer(&valid, &JENV_CATALOG_ENTRY)
        .diagnostics
        .iter()
        .all(|diagnostic| diagnostic.code != "RM-JAVA-EXECUTION-MODE"));
}

#[test]
fn feature_startup_effect_is_retained_unless_literal_proves_read_only_probe() {
    let unknown = infer(&request(vec![text()], 1), &USEJAVA_CATALOG_ENTRY);
    assert!(unknown.effects.0.contains(&EffectKind::EnvironmentWrite));
    for (name, writes) in [
        ("jvm", false),
        ("DESKTOP", false),
        ("awt", true),
        ("swing", true),
    ] {
        let mut call = request(vec![text()], 1);
        call.literals = LiteralContext::new(vec![LiteralValue::String(name.into())]);
        let result = infer(&call, &USEJAVA_CATALOG_ENTRY);
        assert_eq!(
            result.effects.0.contains(&EffectKind::EnvironmentWrite),
            writes,
            "{name}"
        );
        assert_eq!(
            result.effects.0.contains(&EffectKind::FilesystemRead),
            !name.eq_ignore_ascii_case("desktop"),
            "{name}"
        );
    }
}
