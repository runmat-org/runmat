use super::*;
use crate::{catalog::inference::infer_catalog_call, BuiltinCatalogEntry, BuiltinPortability};
use runmat_types::*;

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_catalog_call(entry, request)
}

#[test]
fn python_effect_and_capability_declarations_are_canonical_sets() {
    for entry in [
        &PYARGS_CATALOG_ENTRY,
        &PYENV_CATALOG_ENTRY,
        &PYRUN_CATALOG_ENTRY,
        &PYRUNFILE_CATALOG_ENTRY,
        &TERMINATE_CATALOG_ENTRY,
    ] {
        assert!(
            entry
                .contract
                .effects
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "{} effects must be sorted and unique",
            entry.identity.name
        );
        assert!(
            entry
                .contract
                .capabilities
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "{} capabilities must be sorted and unique",
            entry.identity.name
        );
    }
}

fn request(arguments: Vec<ValueFact>, count: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    }
}

#[test]
fn pyargs_is_a_portable_local_bundle_not_a_python_service_call() {
    let result = infer(&request(Vec::new(), 1), &PYARGS_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert!(result.capabilities.0.is_empty());
    assert_eq!(
        PYARGS_CATALOG_ENTRY.placement.portability,
        BuiltinPortability::NativeAndWasm
    );
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("expected bundle object")
    };
    assert!(object
        .runtime_class
        .as_ref()
        .is_some_and(|class| class.is(runmat_types::standard::PYTHON_ARGUMENTS)));
    assert!(object.properties.contains_key("Names"));
    assert!(object.properties.contains_key("Values"));
}

#[test]
fn pyargs_rejects_known_bad_pair_while_preserving_integer_value_facts() {
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Int64,
        domain: NumericDomain::Real,
    }));
    let result = infer(
        &request(vec![numeric.clone(), numeric.clone()], 1),
        &PYARGS_CATALOG_ENTRY,
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-PYTHON-TEXT"));
    let odd = infer(
        &request(vec![ValueFact::scalar(ValueKindFact::String)], 1),
        &PYARGS_CATALOG_ENTRY,
    );
    assert!(odd
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-PYTHON-PAIRS"));
}

#[test]
fn foreign_calls_keep_foreign_capability_and_unknown_return_values() {
    let source = ValueFact::scalar(ValueKindFact::String);
    let text = ValueFact::scalar(ValueKindFact::String);
    let outputs = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(text.clone()),
            elements: vec![text.clone(), text],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Opaque,
    );
    let result = infer(&request(vec![source, outputs], 2), &PYRUN_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs.len(), 2);
    assert!(result
        .outputs
        .iter()
        .all(|fact| fact.kind == ValueKindFact::Unknown));
    assert!(result
        .capabilities
        .0
        .contains(&CapabilityRequirement::ForeignRuntime));
}

#[test]
fn pyenv_and_terminate_retain_environment_class_and_reject_known_invalid_receiver() {
    let status = infer(&request(Vec::new(), 1), &PYENV_CATALOG_ENTRY);
    assert!(status.effects.0.contains(&EffectKind::FilesystemRead));
    assert!(status.effects.0.contains(&EffectKind::EnvironmentRead));
    assert!(!status.effects.0.contains(&EffectKind::EnvironmentWrite));
    let ValueKindFact::Object(object) = &status.outputs[0].kind else {
        panic!("expected environment")
    };
    assert!(object
        .runtime_class
        .as_ref()
        .is_some_and(|class| class.is(PYTHON_ENVIRONMENT_CLASS)));
    let invalid = infer(
        &request(vec![ValueFact::scalar(ValueKindFact::String)], 1),
        &TERMINATE_CATALOG_ENTRY,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-PYTHON-ENVIRONMENT"));
}

#[test]
fn script_effects_and_known_tail_arity_are_not_erased_by_foreign_dynamic_result() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let file = infer(
        &request(vec![text.clone(), text.clone(), text.clone()], 1),
        &PYRUNFILE_CATALOG_ENTRY,
    );
    assert!(file
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-PYTHON-PAIRS"));
    assert!(file.effects.0.contains(&EffectKind::FilesystemRead));
    assert!(file.effects.0.contains(&EffectKind::Unknown));
    assert!(file
        .capabilities
        .0
        .contains(&CapabilityRequirement::ForeignRuntime));
    assert!(file
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
}
