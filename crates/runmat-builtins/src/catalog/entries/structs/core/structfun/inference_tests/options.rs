use super::fixtures::{callable, infer, numeric, structure, structure_with_completeness};
use runmat_types::{
    DimensionFact, LiteralContext, LiteralValue, NumericClass, RequestedOutputCount, ShapeFact,
    ValueFact, ValueKindFact,
};

#[test]
fn nonuniform_results_recursively_materialize_callback_payloads() {
    let mut device = numeric(NumericClass::Single);
    device.storage = runmat_types::StorageFact::Sparse;
    device.residency = runmat_types::ResidencyFact::Device {
        provider: Some("test".into()),
    };
    let nested = structure(&[("resident", device)]);
    let mut callback = callable("unused");
    let ValueKindFact::Callable(fact) = &mut callback.kind else {
        unreachable!()
    };
    fact.identity = None;
    fact.outputs = vec![nested];
    fact.outputs_complete = true;
    let result = infer(
        vec![
            callback,
            structure(&[("a", numeric(NumericClass::Double))]),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("UniformOutput".into()),
            LiteralValue::Bool(false),
        ]),
        RequestedOutputCount::One,
    );
    let ValueKindFact::Struct(output) = &result.outputs[0].kind else {
        panic!("struct")
    };
    let ValueKindFact::Struct(nested) = &output.fields["a"].kind else {
        panic!("nested struct")
    };
    assert_eq!(
        nested.fields["resident"].residency,
        runmat_types::ResidencyFact::Host
    );
    assert_eq!(
        nested.fields["resident"].storage,
        runmat_types::StorageFact::Sparse
    );
    assert_eq!(
        nested.fields["resident"].numeric().unwrap().class,
        NumericClass::Single
    );
}

#[test]
fn error_handler_capabilities_survive_an_unknown_primary_callback() {
    let mut handler = callable("unused");
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
            ValueFact::unknown(runmat_types::DynamicReason::UnresolvedCallable),
            structure(&[("a", numeric(NumericClass::UInt64))]),
            ValueFact::scalar(ValueKindFact::String),
            handler,
        ],
        LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("ErrorHandler".into()),
            LiteralValue::Unknown,
        ]),
        RequestedOutputCount::One,
    );
    assert!(result
        .capabilities
        .0
        .contains(&runmat_types::CapabilityRequirement::Network));
}

#[test]
fn incomplete_structure_and_runtime_option_names_remain_conservative() {
    let incomplete = infer(
        vec![
            callable("abs"),
            structure_with_completeness(&[("known", numeric(NumericClass::Int32))], false),
        ],
        LiteralContext::default(),
        RequestedOutputCount::One,
    );
    assert!(
        incomplete.diagnostics.is_empty(),
        "{:?}",
        incomplete.diagnostics
    );
    assert_eq!(
        incomplete.outputs[0].shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Unknown, DimensionFact::Known(1)]
        }
    );

    let runtime_option = infer(
        vec![
            callable("abs"),
            structure(&[("known", numeric(NumericClass::Int32))]),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Bool(false),
        ]),
        RequestedOutputCount::One,
    );
    assert!(
        runtime_option.diagnostics.is_empty(),
        "{:?}",
        runtime_option.diagnostics
    );
    assert!(matches!(
        runtime_option.outputs[0].kind,
        ValueKindFact::Unknown
    ));
}
