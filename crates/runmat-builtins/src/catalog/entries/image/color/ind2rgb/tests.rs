use super::*;
use runmat_types::{
    CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact,
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

fn map() -> ValueFact {
    real(
        NumericClass::Double,
        ShapeFact::from(vec![Some(4), Some(3)]),
    )
}

#[test]
fn matrix_indices_append_rgb_channel_with_double_class() {
    for class in [
        NumericClass::Single,
        NumericClass::Double,
        NumericClass::UInt8,
        NumericClass::UInt16,
    ] {
        let result = infer(
            &request(
                vec![real(class, ShapeFact::from(vec![Some(2), Some(5)])), map()],
                1,
            ),
            &IND2RGB_CATALOG_ENTRY,
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(5), Some(3)])
        );
        assert_eq!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            })
        );
    }
}

#[test]
fn unknown_index_dimensions_do_not_fabricate_image_extent() {
    let result = infer(
        &request(
            vec![
                real(NumericClass::Double, ShapeFact::Ranked { rank: 2 }),
                map(),
            ],
            1,
        ),
        &IND2RGB_CATALOG_ENTRY,
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![None, None, Some(3)])
    );
    let unknown = infer(
        &request(
            vec![ValueFact::unknown(DynamicReason::RuntimeValue), map()],
            1,
        ),
        &IND2RGB_CATALOG_ENTRY,
    );
    assert_eq!(unknown.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn invalid_classes_and_colormap_geometry_are_diagnosed() {
    let result = infer(
        &request(
            vec![
                real(NumericClass::Int16, ShapeFact::from(vec![Some(1), Some(1)])),
                real(
                    NumericClass::Single,
                    ShapeFact::from(vec![Some(0), Some(2)]),
                ),
            ],
            1,
        ),
        &IND2RGB_CATALOG_ENTRY,
    );
    for code in [
        "RM-CATALOG-IND2RGB-INDEX-CLASS",
        "RM-CATALOG-IND2RGB-MAP-CLASS",
        "RM-CATALOG-IND2RGB-MAP-SHAPE",
    ] {
        assert!(
            result.diagnostics.iter().any(|diag| diag.code == code),
            "missing {code}"
        );
    }
}

#[test]
fn empty_index_image_remains_empty_rgb_array() {
    let result = infer(
        &request(
            vec![
                real(NumericClass::UInt8, ShapeFact::from(vec![Some(0), Some(2)])),
                map(),
            ],
            1,
        ),
        &IND2RGB_CATALOG_ENTRY,
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(0), Some(2), Some(3)])
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
}
