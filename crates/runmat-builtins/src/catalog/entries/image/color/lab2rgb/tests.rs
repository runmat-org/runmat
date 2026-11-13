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

fn call(image: ValueFact, option: Option<(&str, LiteralValue)>) -> CallInference {
    let mut arguments = vec![image];
    let mut literals = vec![LiteralValue::Unknown];
    if let Some((name, value)) = option {
        arguments.extend([
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ]);
        literals.extend([LiteralValue::String(name.into()), value]);
    }
    infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
        &LAB2RGB_CATALOG_ENTRY,
    )
}

#[test]
fn default_preserves_floating_class_and_image_or_colormap_shape() {
    for (class, shape) in [
        (
            NumericClass::Single,
            ShapeFact::from(vec![Some(2), Some(3), Some(3)]),
        ),
        (
            NumericClass::Double,
            ShapeFact::from(vec![Some(4), Some(3)]),
        ),
    ] {
        let result = call(input(class, shape.clone()), None);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs[0].shape, shape);
        assert_eq!(result.outputs[0].numeric().unwrap().class, class);
    }
}

#[test]
fn typed_output_option_proves_native_integer_and_unknown_control_stays_dynamic() {
    let shape = ShapeFact::from(vec![Some(1), Some(2), Some(3)]);
    let result = call(
        input(NumericClass::Single, shape.clone()),
        Some(("OutputType", LiteralValue::String("uint16".into()))),
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].numeric().unwrap().class,
        NumericClass::UInt16
    );
    assert_eq!(result.outputs[0].shape, shape.clone());
    let unknown = call(
        input(NumericClass::Single, shape),
        Some(("OutputType", LiteralValue::Unknown)),
    );
    assert_eq!(unknown.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn known_invalid_class_shape_and_option_report_without_guessing() {
    let invalid = call(
        input(
            NumericClass::UInt8,
            ShapeFact::from(vec![Some(0), Some(2), Some(3)]),
        ),
        Some(("OutputType", LiteralValue::String("int32".into()))),
    );
    for code in [
        "RM-CATALOG-LAB2RGB-CLASS",
        "RM-CATALOG-LAB2RGB-SHAPE",
        "RM-CATALOG-LAB2RGB-OUTPUT-TYPE",
    ] {
        assert!(
            invalid.diagnostics.iter().any(|diag| diag.code == code),
            "missing {code}"
        );
    }
    let unknown = call(ValueFact::unknown(DynamicReason::RuntimeValue), None);
    assert_eq!(unknown.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(unknown.outputs[0].shape, ShapeFact::Unknown);
}
