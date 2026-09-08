use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::{
    BuiltinId, CallRequest, CallableFact, CallableIdentity, CapabilitySet, DimensionFact,
    LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn numeric(class: NumericClass) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class,
        domain: NumericDomain::Real,
    }))
}
fn callable(name: &str) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: Some(CallableIdentity::Builtin(BuiltinId(name.into()))),
        capabilities: CapabilitySet::default(),
        parameters: Vec::new(),
        parameters_complete: false,
        outputs: Vec::new(),
        outputs_complete: false,
        variadic_inputs: true,
        variadic_outputs: true,
        captures: Vec::new(),
        captures_complete: true,
    }))
}
fn structure(fields: &[(&str, ValueFact)]) -> ValueFact {
    structure_with_completeness(fields, true)
}
fn structure_with_completeness(fields: &[(&str, ValueFact)], fields_complete: bool) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Struct(StructFact {
        fields: fields
            .iter()
            .map(|(name, value)| ((*name).into(), value.clone()))
            .collect::<BTreeMap<_, _>>(),
        fields_complete,
    }))
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
fn infer(
    arguments: Vec<ValueFact>,
    literals: LiteralContext,
    count: RequestedOutputCount,
) -> runmat_types::CallInference {
    infer_catalog_call(
        builtin_catalog_entry_by_name("structfun").unwrap(),
        &CallRequest {
            arguments,
            literals,
            outputs: OutputSelection::new(count),
        },
    )
}

#[test]
fn uniform_output_joins_field_results_into_a_host_column() {
    let result = infer(
        vec![
            callable("abs"),
            structure(&[
                ("a", numeric(NumericClass::Int32)),
                ("b", numeric(NumericClass::Int32)),
            ]),
        ],
        LiteralContext::default(),
        RequestedOutputCount::One,
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].numeric().map(|fact| fact.class),
        Some(NumericClass::Int32)
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(2), DimensionFact::Known(1)]
        }
    );
}

#[test]
fn nonuniform_output_preserves_field_facts() {
    let result = infer(
        vec![
            callable("abs"),
            structure(&[("a", numeric(NumericClass::UInt64))]),
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
        panic!("expected structure")
    };
    assert_eq!(
        output.fields.get("a").map(|fact| &fact.kind),
        Some(&ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }))
    );
}

#[test]
fn empty_struct_and_invalid_inputs_are_explicit() {
    let empty = infer(
        vec![callable("length"), structure(&[])],
        LiteralContext::default(),
        RequestedOutputCount::One,
    );
    assert_eq!(empty.outputs[0].shape.element_count(), Some(0));
    assert_eq!(
        empty.outputs[0].numeric().map(|fact| fact.class),
        Some(NumericClass::Double)
    );
    let invalid = infer(
        vec![numeric(NumericClass::Double), numeric(NumericClass::Double)],
        LiteralContext::default(),
        RequestedOutputCount::One,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-STRUCTFUN-CALLABLE"));
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-STRUCTFUN-STRUCT"));
}
