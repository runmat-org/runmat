use super::fixtures::{callable, infer, numeric, structure};
use runmat_types::{
    DimensionFact, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    RequestedOutputCount, ShapeFact, ValueFact, ValueKindFact,
};

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
