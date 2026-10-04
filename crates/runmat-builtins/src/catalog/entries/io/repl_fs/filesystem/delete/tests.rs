use super::*;
use runmat_types::{
    CallRequest, CapabilityRequirement, EffectKind, LiteralContext, LiteralValue, OutputSelection,
    RequestedOutputCount, ValueFact, ValueKindFact,
};

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(0)),
    }
}

#[test]
fn declaration_satisfies_catalog_invariants() {
    let errors = crate::validate_builtin_catalog(&[&DELETE_CATALOG_ENTRY]);
    assert!(errors.is_empty(), "{errors:#?}");
}

#[test]
fn file_route_uses_only_filesystem_capability() {
    let result = inference::infer(
        &request(
            vec![ValueFact::scalar(ValueKindFact::String)],
            vec![LiteralValue::String("scratch.txt".into())],
        ),
        &DELETE_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.capabilities.0.len(), 1);
    assert!(result
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    assert!(!result.effects.0.contains(&EffectKind::HostCallback));
}

#[test]
fn cell_route_retains_both_file_and_handle_possibilities() {
    use runmat_types::CellFact;
    let unknown = ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue);
    let cell = ValueFact::scalar(ValueKindFact::Cell(CellFact {
        element: Box::new(unknown),
        elements: vec![],
        elements_complete: false,
    }));
    let result = inference::infer(
        &request(vec![cell], vec![LiteralValue::Unknown]),
        &DELETE_CATALOG_ENTRY,
    );
    assert!(result.diagnostics.is_empty());
    assert!(result
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    assert!(result
        .capabilities
        .0
        .contains(&CapabilityRequirement::HostRuntime));
}

#[test]
fn exact_integer_option_and_invalid_fraction_are_distinguished() {
    let args = vec![
        ValueFact::scalar(ValueKindFact::String),
        ValueFact::scalar(ValueKindFact::String),
        ValueFact::scalar(ValueKindFact::Numeric(runmat_types::NumericFact {
            class: runmat_types::NumericClass::Int64,
            domain: runmat_types::NumericDomain::Real,
        })),
    ];
    let valid = inference::infer(
        &request(
            args.clone(),
            vec![
                LiteralValue::String("link".into()),
                LiteralValue::String("ResolveSymbolicLinks".into()),
                LiteralValue::Integer {
                    text: "1".into(),
                    class: runmat_types::NumericClass::Int64,
                },
            ],
        ),
        &DELETE_CATALOG_ENTRY,
    );
    assert!(valid.diagnostics.is_empty());
    let invalid = inference::infer(
        &request(
            args,
            vec![
                LiteralValue::String("link".into()),
                LiteralValue::String("ResolveSymbolicLinks".into()),
                LiteralValue::Number(0.5),
            ],
        ),
        &DELETE_CATALOG_ENTRY,
    );
    assert!(!invalid.diagnostics.is_empty());
}

#[test]
fn empty_file_and_extra_output_reject() {
    let mut call = request(
        vec![ValueFact::scalar(ValueKindFact::String)],
        vec![LiteralValue::String(String::new())],
    );
    call.outputs = OutputSelection::new(RequestedOutputCount::Exactly(1));
    let result = inference::infer(&call, &DELETE_CATALOG_ENTRY);
    assert!(result.diagnostics.len() >= 2);
}
