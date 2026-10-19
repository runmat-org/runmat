use super::fixtures::{builtin, cell, infer, numeric};
use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, OutputSelection, RequestedOutputCount,
    ValueFact, ValueKindFact,
};

#[test]
fn dynamic_duplicate_and_padded_options_match_runtime_collection_policy() {
    let base = vec![builtin("abs"), cell(numeric(NumericClass::Double), &[2, 2])];
    for (names, controls, expected_cell) in [
        (vec![LiteralValue::Unknown], vec![false], None),
        (
            vec![LiteralValue::String(" UniformOutput ".into())],
            vec![false],
            Some(true),
        ),
        (
            vec![
                LiteralValue::String("UniformOutput".into()),
                LiteralValue::String("UniformOutput".into()),
            ],
            vec![false, true],
            Some(true),
        ),
    ] {
        let mut arguments = base.clone();
        let mut literals = vec![LiteralValue::Unknown, LiteralValue::Unknown];
        for (name, control) in names.into_iter().zip(controls) {
            arguments.extend([
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Logical),
            ]);
            literals.extend([name, LiteralValue::Bool(control)]);
        }
        let result = infer(arguments, LiteralContext::new(literals));
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        match expected_cell {
            Some(true) => assert!(matches!(result.outputs[0].kind, ValueKindFact::Cell(_))),
            None => assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown),
            _ => unreachable!(),
        }
    }
}

#[test]
fn nonuniform_collection_preserves_callback_residency() {
    let mut device = numeric(NumericClass::Single);
    device.residency = runmat_types::ResidencyFact::Device {
        provider: Some("test".into()),
    };
    let mut callback = builtin("unused");
    let ValueKindFact::Callable(fact) = &mut callback.kind else {
        unreachable!()
    };
    fact.identity = None;
    fact.outputs = vec![device.clone()];
    fact.outputs_complete = true;
    let result = infer(
        vec![
            callback,
            cell(numeric(NumericClass::Double), &[1, 2]),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("UniformOutput".into()),
            LiteralValue::Bool(false),
        ]),
    );
    let ValueKindFact::Cell(output) = &result.outputs[0].kind else {
        panic!("cell")
    };
    assert_eq!(output.element.residency, device.residency);
}

#[test]
fn shorthand_multioutput_request_is_diagnosed() {
    let result = infer_catalog_call(
        builtin_catalog_entry_by_name("cellfun").unwrap(),
        &CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::String),
                cell(numeric(NumericClass::Double), &[0, 0]),
            ],
            literals: LiteralContext::new(vec![LiteralValue::String("isempty".into())]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        },
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-CELLFUN-OUTPUT-COUNT"));
}

#[test]
fn error_handler_capabilities_and_result_classes_are_not_lost() {
    let mut handler = builtin("unused");
    let ValueKindFact::Callable(fact) = &mut handler.kind else {
        unreachable!()
    };
    fact.identity = None;
    fact.capabilities
        .0
        .insert(runmat_types::CapabilityRequirement::Network);
    fact.outputs = vec![numeric(NumericClass::Double)];
    fact.outputs_complete = true;
    let result = infer(
        vec![
            builtin("abs"),
            cell(numeric(NumericClass::UInt64), &[1, 2]),
            ValueFact::scalar(ValueKindFact::String),
            handler,
        ],
        LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("ErrorHandler".into()),
            LiteralValue::Unknown,
        ]),
    );
    assert!(result
        .capabilities
        .0
        .contains(&runmat_types::CapabilityRequirement::Network));
    assert_ne!(
        result.outputs[0].numeric().map(|n| n.class),
        Some(NumericClass::UInt64)
    );
}
