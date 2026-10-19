use super::*;
use runmat_types::{DynamicReason, LiteralContext, OutputSelection, RequestedOutputCount};

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

fn call(
    entry: &BuiltinCatalogEntry,
    class: ImageOutputClass,
    image: ValueFact,
    option: Option<(ValueFact, LiteralValue)>,
    outputs: usize,
) -> CallInference {
    let mut arguments = vec![image];
    let mut literals = LiteralContext::default();
    if let Some((fact, literal)) = option {
        arguments.push(fact);
        literals.literal_args = vec![LiteralValue::Unknown, literal];
    }
    infer(
        &CallRequest {
            arguments,
            literals,
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        entry,
        class,
    )
}

fn indexed() -> (ValueFact, LiteralValue) {
    (
        ValueFact::scalar(ValueKindFact::String),
        LiteralValue::String(" InDeXeD ".into()),
    )
}

#[test]
fn indexed_character_control_accepts_flattened_row_column_and_nd_geometry() {
    for shape in [
        ShapeFact::from(vec![Some(1), Some(7)]),
        ShapeFact::from(vec![Some(7), Some(1)]),
        ShapeFact::from(vec![Some(1), Some(1), Some(7)]),
    ] {
        let result = call(
            &crate::IM2DOUBLE_CATALOG_ENTRY,
            ImageOutputClass::Double,
            input(NumericClass::UInt8, ShapeFact::Scalar),
            Some((
                ValueFact::proven(ValueKindFact::Character, shape, StorageFact::Dense),
                LiteralValue::Character("indexed".into()),
            )),
            1,
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    }
}

#[test]
fn each_conversion_keeps_nd_shape_and_proves_its_output_class() {
    let shape = ShapeFact::from(vec![Some(2), Some(3), Some(4), Some(0)]);
    for (entry, class, expected) in [
        (
            &crate::IM2DOUBLE_CATALOG_ENTRY,
            ImageOutputClass::Double,
            NumericClass::Double,
        ),
        (
            &crate::IM2UINT8_CATALOG_ENTRY,
            ImageOutputClass::UInt8,
            NumericClass::UInt8,
        ),
        (
            &crate::IM2UINT16_CATALOG_ENTRY,
            ImageOutputClass::UInt16,
            NumericClass::UInt16,
        ),
    ] {
        let result = call(
            entry,
            class,
            input(NumericClass::UInt8, shape.clone()),
            None,
            1,
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.outputs[0].shape, shape);
        assert_eq!(
            result.outputs[0].numeric().map(|fact| fact.class),
            Some(expected)
        );
    }
}

#[test]
fn indexed_mode_narrows_classes_without_guessing_unknown_control() {
    let image = input(NumericClass::Int16, ShapeFact::from(vec![Some(2), Some(2)]));
    let valid = call(
        &crate::IM2DOUBLE_CATALOG_ENTRY,
        ImageOutputClass::Double,
        image.clone(),
        None,
        1,
    );
    assert!(valid.diagnostics.is_empty());
    let invalid = call(
        &crate::IM2DOUBLE_CATALOG_ENTRY,
        ImageOutputClass::Double,
        image.clone(),
        Some(indexed()),
        1,
    );
    assert!(!invalid.diagnostics.is_empty());
    let dynamic_option = (
        ValueFact::unknown(DynamicReason::RuntimeValue),
        LiteralValue::Unknown,
    );
    let dynamic = call(
        &crate::IM2DOUBLE_CATALOG_ENTRY,
        ImageOutputClass::Double,
        image,
        Some(dynamic_option),
        1,
    );
    assert!(dynamic.diagnostics.is_empty());
}

#[test]
fn unknown_image_does_not_claim_a_shape_or_device_owner() {
    let result = call(
        &crate::IM2UINT8_CATALOG_ENTRY,
        ImageOutputClass::UInt8,
        ValueFact::unknown(DynamicReason::RuntimeValue),
        None,
        1,
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
    assert!(result.diagnostics.is_empty());
}

#[test]
fn known_invalid_option_class_complex_image_and_extra_output_diagnose() {
    let option = (
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        LiteralValue::Number(1.0),
    );
    let mut complex = input(NumericClass::Double, ShapeFact::Scalar);
    complex.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Complex,
    });
    let result = call(
        &crate::IM2UINT16_CATALOG_ENTRY,
        ImageOutputClass::UInt16,
        complex,
        Some(option),
        2,
    );
    assert!(result.diagnostics.len() >= 2, "{:?}", result.diagnostics);
}

#[test]
fn singleton_tensor_output_is_the_scalar_published_by_runtime() {
    let input = input(
        NumericClass::UInt16,
        ShapeFact::from(vec![Some(1), Some(1), Some(1)]),
    );
    let result = call(
        &crate::IM2DOUBLE_CATALOG_ENTRY,
        ImageOutputClass::Double,
        input,
        None,
        1,
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(result.outputs[0].storage, StorageFact::Scalar);
}

#[test]
fn known_index_bounds_match_existing_runtime_policy_without_rejecting_fractional_or_negative() {
    let cases = [
        (
            &crate::IM2UINT8_CATALOG_ENTRY,
            ImageOutputClass::UInt8,
            NumericClass::Double,
            256.0,
            false,
        ),
        (
            &crate::IM2UINT8_CATALOG_ENTRY,
            ImageOutputClass::UInt8,
            NumericClass::Double,
            256.5,
            true,
        ),
        (
            &crate::IM2UINT8_CATALOG_ENTRY,
            ImageOutputClass::UInt8,
            NumericClass::UInt16,
            255.0,
            false,
        ),
        (
            &crate::IM2UINT8_CATALOG_ENTRY,
            ImageOutputClass::UInt8,
            NumericClass::UInt16,
            256.0,
            true,
        ),
        (
            &crate::IM2UINT16_CATALOG_ENTRY,
            ImageOutputClass::UInt16,
            NumericClass::Double,
            65536.0,
            false,
        ),
        (
            &crate::IM2UINT16_CATALOG_ENTRY,
            ImageOutputClass::UInt16,
            NumericClass::Double,
            f64::INFINITY,
            true,
        ),
        (
            &crate::IM2UINT16_CATALOG_ENTRY,
            ImageOutputClass::UInt16,
            NumericClass::Double,
            -0.5,
            false,
        ),
        (
            &crate::IM2UINT8_CATALOG_ENTRY,
            ImageOutputClass::UInt8,
            NumericClass::Double,
            1.5,
            false,
        ),
    ];
    for (entry, output, class, number, invalid) in cases {
        let request = CallRequest {
            arguments: vec![input(class, ShapeFact::Scalar), indexed().0],
            literals: LiteralContext::new(vec![LiteralValue::Number(number), indexed().1]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let result = infer(&request, entry, output);
        assert_eq!(
            result
                .diagnostics
                .iter()
                .any(|item| item.code == "RM-CATALOG-IMAGE-CONVERSION-INDEX-BOUND"),
            invalid,
            "{class:?} {number:?}: {:?}",
            result.diagnostics,
        );
    }
}
