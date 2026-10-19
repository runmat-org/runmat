use super::*;
use runmat_types::{
    DynamicReason, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, ValueFact, ValueKindFact,
};

fn call(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(vec![]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

fn numeric() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

#[test]
fn constructors_infer_concrete_host_handle_fixtures() {
    let path = inference::path_fixture(
        &call(vec![ValueFact::scalar(ValueKindFact::String)], 1),
        &PATH_FIXTURE_CATALOG_ENTRY,
    );
    let temp = inference::temporary_folder_fixture(
        &call(vec![], 1),
        &TEMPORARY_FOLDER_FIXTURE_CATALOG_ENTRY,
    );
    for (inferred, class, property) in [
        (path, "matlab.unittest.fixtures.PathFixture", "Path"),
        (
            temp,
            "matlab.unittest.fixtures.TemporaryFolderFixture",
            "Folder",
        ),
    ] {
        assert!(
            inferred.diagnostics.is_empty(),
            "{:#?}",
            inferred.diagnostics
        );
        assert_eq!(inferred.outputs[0].shape, ShapeFact::Scalar);
        assert_eq!(
            inferred.outputs[0].residency,
            runmat_types::ResidencyFact::Host
        );
        let ValueKindFact::Object(object) = &inferred.outputs[0].kind else {
            panic!("expected fixture object")
        };
        assert_eq!(
            object
                .runtime_class
                .as_ref()
                .map(|class| class.display_name()),
            Some(class)
        );
        assert_eq!(object.handle_semantics, Some(true));
        assert!(object.properties.contains_key(property));
    }
    assert!(inference::temporary_folder_fixture(
        &call(vec![], 0),
        &TEMPORARY_FOLDER_FIXTURE_CATALOG_ENTRY
    )
    .outputs
    .is_empty());
}

#[test]
fn invalid_constructor_forms_and_known_device_text_are_diagnosed() {
    assert!(
        !inference::path_fixture(&call(vec![], 1), &PATH_FIXTURE_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    assert!(
        !inference::path_fixture(&call(vec![numeric()], 1), &PATH_FIXTURE_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
    assert!(!inference::temporary_folder_fixture(
        &call(vec![unknown()], 1),
        &TEMPORARY_FOLDER_FIXTURE_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
    assert!(
        inference::path_fixture(&call(vec![unknown()], 1), &PATH_FIXTURE_CATALOG_ENTRY)
            .diagnostics
            .is_empty()
    );
}

#[test]
fn apply_preserves_fixture_identity_and_teardown_returns_empty_double() {
    let fixture = inference::path_fixture(
        &call(vec![ValueFact::scalar(ValueKindFact::String)], 1),
        &PATH_FIXTURE_CATALOG_ENTRY,
    )
    .outputs
    .remove(0);
    let applied = inference::apply_fixture(
        &call(vec![unknown(), fixture.clone()], 1),
        &APPLY_FIXTURE_CATALOG_ENTRY,
    );
    assert!(applied.diagnostics.is_empty(), "{:#?}", applied.diagnostics);
    assert_eq!(applied.outputs[0].kind, fixture.kind);
    assert!(applied
        .effects
        .0
        .contains(&runmat_types::EffectKind::FilesystemWrite));
    let teardown = inference::add_teardown(
        &call(vec![unknown(), unknown(), numeric()], 1),
        &ADD_TEARDOWN_CATALOG_ENTRY,
    );
    assert!(
        teardown.diagnostics.is_empty(),
        "{:#?}",
        teardown.diagnostics
    );
    assert_eq!(
        teardown.outputs[0].shape,
        ShapeFact::from(vec![Some(0), Some(0)])
    );
    assert!(teardown
        .effects
        .0
        .contains(&runmat_types::EffectKind::Unknown));
}

#[test]
fn invalid_method_forms_are_diagnosed_without_guessing_unknown_classes() {
    for inferred in [
        inference::apply_fixture(
            &call(vec![numeric(), unknown()], 1),
            &APPLY_FIXTURE_CATALOG_ENTRY,
        ),
        inference::apply_fixture(
            &call(vec![unknown(), numeric()], 1),
            &APPLY_FIXTURE_CATALOG_ENTRY,
        ),
        inference::add_teardown(
            &call(vec![unknown(), numeric()], 1),
            &ADD_TEARDOWN_CATALOG_ENTRY,
        ),
        inference::add_teardown(&call(vec![unknown()], 1), &ADD_TEARDOWN_CATALOG_ENTRY),
    ] {
        assert!(!inferred.diagnostics.is_empty());
    }
    assert!(inference::apply_fixture(
        &call(vec![unknown(), unknown()], 1),
        &APPLY_FIXTURE_CATALOG_ENTRY
    )
    .diagnostics
    .is_empty());
}
