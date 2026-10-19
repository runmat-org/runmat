use super::*;
use runmat_types::{LiteralContext, OutputSelection, RequestedOutputCount};

fn call(class: NumericClass, shape: ShapeFact, conversion: RgbConversion) -> CallInference {
    let entry = match conversion {
        RgbConversion::Gray => &crate::RGB2GRAY_CATALOG_ENTRY,
        RgbConversion::Hsv => &crate::RGB2HSV_CATALOG_ENTRY,
    };
    infer(
        &CallRequest {
            arguments: vec![ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class,
                    domain: NumericDomain::Real,
                }),
                shape,
                StorageFact::Dense,
            )],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
        entry,
        conversion,
    )
}

#[test]
fn truecolor_shapes_and_classes_follow_runtime_publication() {
    let image = ShapeFact::from(vec![Some(2), Some(3), Some(3)]);
    let gray = call(NumericClass::UInt16, image.clone(), RgbConversion::Gray);
    assert_eq!(
        gray.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert_eq!(
        gray.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt16,
            domain: NumericDomain::Real,
        })
    );
    let hsv = call(NumericClass::UInt16, image, RgbConversion::Hsv);
    assert_eq!(
        hsv.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3), Some(3)])
    );
    assert_eq!(
        hsv.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
    let single = call(
        NumericClass::Single,
        ShapeFact::Ranked { rank: 3 },
        RgbConversion::Hsv,
    );
    assert_eq!(
        single.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })
    );
}

#[test]
fn colormap_requires_double_and_empty_or_wrong_channel_diagnoses() {
    let map = ShapeFact::from(vec![Some(4), Some(3)]);
    assert_eq!(
        call(NumericClass::Double, map.clone(), RgbConversion::Gray).outputs[0].shape,
        map
    );
    let bad_class = call(
        NumericClass::UInt8,
        ShapeFact::from(vec![Some(4), Some(3)]),
        RgbConversion::Hsv,
    );
    assert!(bad_class
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-RGB-CLASS"));
    for shape in [
        ShapeFact::from(vec![Some(0), Some(3)]),
        ShapeFact::from(vec![Some(2), Some(2), Some(4)]),
    ] {
        let invalid = call(NumericClass::Double, shape, RgbConversion::Gray);
        assert!(invalid
            .diagnostics
            .iter()
            .any(|item| item.code == "RM-CATALOG-RGB-SHAPE"));
    }
}

#[test]
fn singleton_truecolor_grayscale_is_scalar_and_unknown_shape_stays_unknown() {
    let singleton = call(
        NumericClass::UInt8,
        ShapeFact::from(vec![Some(1), Some(1), Some(3)]),
        RgbConversion::Gray,
    );
    assert_eq!(singleton.outputs[0].shape, ShapeFact::Scalar);
    assert_eq!(singleton.outputs[0].storage, StorageFact::Scalar);
    let single_pixel = call(
        NumericClass::Single,
        ShapeFact::from(vec![Some(1), Some(1), Some(3)]),
        RgbConversion::Gray,
    );
    assert_eq!(
        single_pixel.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
    let uncertain_single = call(
        NumericClass::Single,
        ShapeFact::Ranked { rank: 3 },
        RgbConversion::Gray,
    );
    assert_eq!(uncertain_single.outputs[0].kind, ValueKindFact::Unknown);
    let unknown = call(NumericClass::Double, ShapeFact::Unknown, RgbConversion::Hsv);
    assert_eq!(unknown.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn unknown_input_class_keeps_dynamic_output_kind_but_known_success_geometry() {
    let request = CallRequest {
        arguments: vec![ValueFact::unknown(
            runmat_types::DynamicReason::RuntimeValue,
        )],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inferred = infer(
        &request,
        &crate::RGB2GRAY_CATALOG_ENTRY,
        RgbConversion::Gray,
    );
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Unknown);
    assert_eq!(
        inferred.outputs[0].certainty,
        runmat_types::CertaintyFact::Dynamic(runmat_types::DynamicReason::RuntimeValue)
    );
}
