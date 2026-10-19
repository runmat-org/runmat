use super::common::*;
use super::*;
use runmat_types::{
    CallRequest, CellFact, DynamicReason, LiteralContext, LiteralValue, NumericClass,
    NumericDomain, NumericFact, OutputSelection as RequestedOutputs, RequestedOutputCount,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: RequestedOutputs::new(RequestedOutputCount::Exactly(1)),
    }
}

fn numeric_fact(shape: ShapeFact) -> ValueFact {
    numeric(NumericClass::Double, shape)
}

#[test]
fn encode_preserves_known_axes_without_guessing_class_count() {
    let args = vec![
        ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(3), Some(1)]),
            StorageFact::Dense,
        ),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        ValueFact::scalar(ValueKindFact::Character),
        ValueFact::proven(
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(1)]),
            StorageFact::Dense,
        ),
    ];
    let call = request(
        args,
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
            LiteralValue::Character("ClassNames".into()),
            LiteralValue::Unknown,
        ],
    );
    let result = encode(&call, &super::super::ONEHOTENCODE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(3), None])
    );
    assert_eq!(
        result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real
        })
    );
}

#[test]
fn invalid_known_dimension_does_not_publish_output_fact() {
    let call = request(
        vec![
            numeric_fact(ShapeFact::from(vec![Some(3), Some(1)])),
            numeric_fact(ShapeFact::Scalar),
        ],
        vec![LiteralValue::Unknown, LiteralValue::Number(0.0)],
    );
    let result = encode(&call, &super::super::ONEHOTENCODE_CATALOG_ENTRY);
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn decode_known_geometry_and_typed_output() {
    let args = vec![
        numeric_fact(ShapeFact::from(vec![Some(3), Some(2)])),
        ValueFact::unknown(DynamicReason::RuntimeValue),
        numeric_fact(ShapeFact::Scalar),
        ValueFact::scalar(ValueKindFact::Character),
    ];
    let call = request(
        args,
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Integer {
                text: "2".into(),
                class: NumericClass::UInt64,
            },
            LiteralValue::Character("uint64".into()),
        ],
    );
    let result = decode(&call, &super::super::ONEHOTDECODE_CATALOG_ENTRY);
    assert!(result.diagnostics.is_empty());
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    assert_eq!(
        result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real
        })
    );
}

#[test]
fn character_rows_and_cell_group_containers_do_not_reuse_outer_shape() {
    let char_input = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(3), Some(4)]),
        StorageFact::Dense,
    );
    let dim = numeric_fact(ShapeFact::Scalar);
    let call = request(
        vec![char_input, dim],
        vec![LiteralValue::Unknown, LiteralValue::Number(2.0)],
    );
    let encoded = encode(&call, &super::super::ONEHOTENCODE_CATALOG_ENTRY);
    assert_eq!(encoded.outputs[0].shape, ShapeFact::Unknown);

    let cell_input = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    let call = request(vec![cell_input], vec![LiteralValue::Unknown]);
    let dummy = dummyvar(&call, &super::super::DUMMYVAR_CATALOG_ENTRY);
    assert_eq!(dummy.outputs[0].shape, ShapeFact::from(vec![None, None]));
}

#[test]
fn invalid_known_output_type_remains_dynamic() {
    let call = request(
        vec![
            numeric_fact(ShapeFact::from(vec![Some(3), Some(1)])),
            numeric_fact(ShapeFact::Scalar),
            ValueFact::scalar(ValueKindFact::Character),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
            LiteralValue::Character("categorical".into()),
        ],
    );
    let result = encode(&call, &super::super::ONEHOTENCODE_CATALOG_ENTRY);
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn dummyvar_nd_numeric_and_logical_facts_use_flattened_row_count() {
    for (kind, dims, rows) in [
        (
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            vec![Some(2), Some(2), Some(2)],
            8,
        ),
        (ValueKindFact::Logical, vec![Some(1), Some(1), Some(3)], 3),
    ] {
        let input = ValueFact::proven(kind, ShapeFact::from(dims), StorageFact::Dense);
        let call = request(vec![input], vec![LiteralValue::Unknown]);
        let result = dummyvar(&call, &super::super::DUMMYVAR_CATALOG_ENTRY);
        assert!(result.diagnostics.is_empty());
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(rows), None])
        );
    }
}
