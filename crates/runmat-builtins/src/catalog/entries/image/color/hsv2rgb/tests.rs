use super::*;
use runmat_types::{
    CallRequest, DimensionFact, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

fn request(arguments: Vec<ValueFact>, count: usize) -> CallRequest {
    use runmat_types::{LiteralContext, OutputSelection, RequestedOutputCount};
    CallRequest {
        literals: LiteralContext::default(),
        arguments,
        outputs: OutputSelection::new(match count {
            0 => RequestedOutputCount::Zero,
            1 => RequestedOutputCount::One,
            count => RequestedOutputCount::Exactly(count),
        }),
    }
}

fn real(class: NumericClass, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

#[test]
fn known_single_image_and_colormap_facts_preserve_shape_and_class() {
    for shape in [
        ShapeFact::from(vec![Some(2), Some(4), Some(3)]),
        ShapeFact::from(vec![Some(4), Some(3)]),
    ] {
        let request = request(vec![real(NumericClass::Single, shape.clone())], 1);
        let inferred = infer(&request, &HSV2RGB_CATALOG_ENTRY);
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
        assert_eq!(inferred.outputs[0].shape, shape);
        assert_eq!(
            inferred.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real
            })
        );
    }
}

#[test]
fn logical_truecolor_is_double_and_logical_colormap_rejects() {
    let image = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(1), Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let inferred = infer(&request(vec![image], 1), &HSV2RGB_CATALOG_ENTRY);
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real
        })
    );
    let map = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let invalid = infer(&request(vec![map], 1), &HSV2RGB_CATALOG_ENTRY);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diag| diag.code == "RM-CATALOG-HSV2RGB-CLASS"));
}

#[test]
fn unknown_and_partial_geometry_do_not_invent_dimensions() {
    let unknown = infer(
        &request(
            vec![ValueFact::unknown(
                runmat_types::DynamicReason::RuntimeValue,
            )],
            1,
        ),
        &HSV2RGB_CATALOG_ENTRY,
    );
    assert_eq!(unknown.outputs[0].shape, ShapeFact::Unknown);
    let partial = ShapeFact::Shaped {
        dims: vec![
            DimensionFact::Unknown,
            DimensionFact::Known(5),
            DimensionFact::Known(3),
        ],
    };
    let inferred = infer(
        &request(vec![real(NumericClass::Double, partial.clone())], 1),
        &HSV2RGB_CATALOG_ENTRY,
    );
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(inferred.outputs[0].shape, partial);
}

#[test]
fn unknown_kind_does_not_raise_class_error_but_known_complex_does() {
    let shape = ShapeFact::from(vec![Some(2), Some(2), Some(3)]);
    let mut unknown = ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue);
    unknown.shape = shape.clone();
    let inferred = infer(&request(vec![unknown], 1), &HSV2RGB_CATALOG_ENTRY);
    assert!(!inferred
        .diagnostics
        .iter()
        .any(|diag| diag.code == "RM-CATALOG-HSV2RGB-CLASS"));
    let complex = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }),
        shape,
        StorageFact::Dense,
    );
    let invalid = infer(&request(vec![complex], 1), &HSV2RGB_CATALOG_ENTRY);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diag| diag.code == "RM-CATALOG-HSV2RGB-CLASS"));
}

#[test]
fn invalid_known_classes_and_shapes_report_diagnostics() {
    let invalid = infer(
        &request(
            vec![real(
                NumericClass::UInt8,
                ShapeFact::from(vec![Some(0), Some(2), Some(3)]),
            )],
            1,
        ),
        &HSV2RGB_CATALOG_ENTRY,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diag| diag.code == "RM-CATALOG-HSV2RGB-CLASS"));
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diag| diag.code == "RM-CATALOG-HSV2RGB-SHAPE"));
}
