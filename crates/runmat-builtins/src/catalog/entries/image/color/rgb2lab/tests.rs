use super::*;
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn input(class: NumericClass, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn call(image: ValueFact, pairs: &[(&str, &str)]) -> CallInference {
    let mut arguments = vec![image];
    let mut literals = vec![LiteralValue::Unknown];
    for &(name, value) in pairs {
        arguments.extend([
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ]);
        literals.extend([
            LiteralValue::String(name.into()),
            LiteralValue::String(value.into()),
        ]);
    }
    infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
        &RGB2LAB_CATALOG_ENTRY,
    )
}

#[test]
fn truecolor_stack_and_colormap_keep_exact_shape_with_output_class() {
    for (class, shape, output) in [
        (
            NumericClass::Single,
            ShapeFact::from(vec![Some(2), Some(4), Some(3), Some(0)]),
            NumericClass::Single,
        ),
        (
            NumericClass::UInt16,
            ShapeFact::from(vec![Some(5), Some(3)]),
            NumericClass::Double,
        ),
        (
            NumericClass::Double,
            ShapeFact::from(vec![Some(1), Some(2), Some(3)]),
            NumericClass::Double,
        ),
    ] {
        let result = call(input(class, shape.clone()), &[]);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs[0].shape, shape);
        assert_eq!(result.outputs[0].numeric().unwrap().class, output);
    }
}

#[test]
fn color_space_and_white_point_literals_use_typed_decoders() {
    let image = input(
        NumericClass::Double,
        ShapeFact::from(vec![Some(2), Some(3)]),
    );
    let valid = call(
        image.clone(),
        &[("ColorSpace", "prophoto-rgb"), ("WhitePoint", "ICC")],
    );
    assert!(valid.diagnostics.is_empty(), "{:?}", valid.diagnostics);
    let invalid = call(image, &[("ColorSpace", "wide-rgb"), ("WhitePoint", "D99")]);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diag| diag.code == "RM-CATALOG-RGB2LAB-COLORSPACE"));
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diag| diag.code == "RM-CATALOG-RGB2LAB-WHITEPOINT"));
}

#[test]
fn unknown_and_invalid_input_facts_remain_distinct() {
    let unknown = call(ValueFact::unknown(DynamicReason::RuntimeValue), &[]);
    assert_eq!(unknown.outputs[0].shape, ShapeFact::Unknown);
    assert_eq!(unknown.outputs[0].kind, ValueKindFact::Unknown);
    let invalid = call(
        input(
            NumericClass::Int16,
            ShapeFact::from(vec![Some(0), Some(2), Some(4)]),
        ),
        &[],
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diag| diag.code == "RM-CATALOG-RGB2LAB-CLASS"));
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diag| diag.code == "RM-CATALOG-RGB2LAB-SHAPE"));
}
