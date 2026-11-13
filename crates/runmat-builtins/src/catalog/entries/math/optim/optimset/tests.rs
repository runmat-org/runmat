use super::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount, StructFact,
    ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn call(args: Vec<ValueFact>, literals: Vec<LiteralValue>) -> runmat_types::CallInference {
    inference::infer(
        &CallRequest {
            arguments: args,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &OPTIMSET_CATALOG_ENTRY,
    )
}

#[test]
fn default_builder_proves_legacy_field_names() {
    let result = call(vec![], vec![]);
    let ValueKindFact::Struct(output) = &result.outputs[0].kind else {
        panic!("struct expected")
    };
    assert!(output.fields_complete);
    assert_eq!(output.fields.len(), crate::FminsearchOptionName::ALL.len());
    assert!(crate::FminsearchOptionName::ALL
        .into_iter()
        .all(|field| output.fields.contains_key(field.name())));
    assert_eq!(
        result.outputs[0].residency,
        runmat_types::ResidencyFact::Host
    );
}

#[test]
fn aliases_and_unknown_custom_fields_preserve_value_facts() {
    let input = ValueFact::scalar(ValueKindFact::String);
    let value = ValueFact::scalar(ValueKindFact::Logical);
    let result = call(
        vec![input.clone(), value.clone(), input, value],
        vec![
            LiteralValue::String("GradObj".into()),
            LiteralValue::Unknown,
            LiteralValue::String("FutureOption".into()),
            LiteralValue::Unknown,
        ],
    );
    let ValueKindFact::Struct(output) = &result.outputs[0].kind else {
        panic!("struct expected")
    };
    assert!(output.fields_complete);
    assert!(output.fields.contains_key("SpecifyObjectiveGradient"));
    assert!(output.fields.contains_key("FutureOption"));

    // Placement origin is absent from static facts. Both retain and gather
    // preserve a native numeric payload's class and shape, not its residency.
    let mut payload = ValueFact::proven(
        ValueKindFact::Numeric(runmat_types::NumericFact {
            class: runmat_types::NumericClass::UInt64,
            domain: runmat_types::NumericDomain::Real,
        }),
        runmat_types::ShapeFact::from(vec![Some(2), Some(3)]),
        runmat_types::StorageFact::Dense,
    );
    payload.residency = runmat_types::ResidencyFact::Device { provider: None };
    let result = call(
        vec![ValueFact::scalar(ValueKindFact::String), payload.clone()],
        vec![
            LiteralValue::String("FutureOption".into()),
            LiteralValue::Unknown,
        ],
    );
    let ValueKindFact::Struct(output) = &result.outputs[0].kind else {
        panic!("struct expected")
    };
    let stored = &output.fields["FutureOption"];
    assert_eq!(stored.kind, payload.kind);
    assert_eq!(stored.shape, payload.shape);
    assert_eq!(stored.storage, payload.storage);
    assert_eq!(stored.residency, runmat_types::ResidencyFact::Unknown);
}

#[test]
fn dynamic_name_may_overwrite_source_field_so_does_not_prove_its_value() {
    let mut fields = BTreeMap::new();
    fields.insert("MaxIter".into(), ValueFact::scalar(ValueKindFact::Logical));
    let source = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)));
    let result = call(
        vec![
            source,
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![LiteralValue::Unknown; 3],
    );
    let ValueKindFact::Struct(output) = &result.outputs[0].kind else {
        panic!("struct expected")
    };
    assert!(!output.fields_complete);
    assert!(!output.fields.contains_key("MaxIter"));

    let result = call(
        vec![
            ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("Display".into()),
            LiteralValue::Unknown,
        ],
    );
    assert!(
        result.diagnostics.is_empty(),
        "unknown existing struct is valid"
    );
    let ValueKindFact::Struct(output) = &result.outputs[0].kind else {
        panic!("struct expected")
    };
    assert!(!output.fields_complete);
    assert_eq!(output.fields["Display"].kind, ValueKindFact::Logical);
}

#[test]
fn bad_name_and_odd_pairs_have_diagnostics() {
    let bad = call(
        vec![
            ValueFact::scalar(ValueKindFact::Logical),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![LiteralValue::Unknown; 2],
    );
    assert!(bad
        .diagnostics
        .iter()
        .any(|item| item.code == "RM.OPTIMSET.INVALID_INPUT"));
    let odd = call(
        vec![ValueFact::scalar(ValueKindFact::String)],
        vec![LiteralValue::String("TolX".into())],
    );
    assert!(odd
        .diagnostics
        .iter()
        .any(|item| item.code == "RM.OPTIMSET.INVALID_ARGUMENT"));
}
