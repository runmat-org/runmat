use super::*;
use runmat_types::*;

fn infer(first: &str, count: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::String); 3],
            literals: LiteralContext::new(vec![
                LiteralValue::String(first.into()),
                LiteralValue::Unknown,
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        },
        &WINQUERYREG_CATALOG_ENTRY,
    )
}

#[test]
fn name_form_has_cell_column_but_payload_type_remains_external() {
    let names = infer("name", 1);
    assert!(names.diagnostics.is_empty());
    assert!(matches!(names.outputs[0].kind, ValueKindFact::Cell(_)));
    assert_eq!(names.outputs[0].shape, ShapeFact::from(vec![None, Some(1)]));
    assert_eq!(
        infer("HKEY_CURRENT_USER", 1).outputs[0].kind,
        ValueKindFact::Unknown
    );
    // The special leading selector is exact, matching runtime, not a lowercase rewrite.
    assert_eq!(infer("NAME", 1).outputs[0].kind, ValueKindFact::Unknown);
    assert!(!infer("name", 2).diagnostics.is_empty());
}

#[test]
fn default_query_and_named_selector_are_distinct_typed_controls() {
    assert_eq!(
        WinqueryregSelection::value(None),
        WinqueryregSelection::DefaultValue
    );
    assert_eq!(
        WinqueryregSelection::value(Some(String::new())),
        WinqueryregSelection::DefaultValue
    );
    assert_eq!(
        WinqueryregSelection::value(Some(" spaced value ".into())),
        WinqueryregSelection::NamedValue(" spaced value ".into())
    );
}

#[test]
fn registry_contract_never_claims_device_or_complete_resource_execution() {
    assert_eq!(
        WINQUERYREG_CATALOG_ENTRY.contract.maturity,
        BuiltinContractMaturity::Incomplete
    );
    assert_eq!(
        WINQUERYREG_CATALOG_ENTRY.placement.accelerator,
        BuiltinAcceleratorPolicy::Forbidden
    );
    assert_eq!(
        WINQUERYREG_CATALOG_ENTRY.placement.preparation,
        BuiltinOperandPreparation::RuntimeOwned
    );
    assert!(WINQUERYREG_DESCRIPTOR
        .errors
        .iter()
        .any(|error| error.identifier == Some("RunMat:Placement:InsufficientResourceEvidence")));
}
