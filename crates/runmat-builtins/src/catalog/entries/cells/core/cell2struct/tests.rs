use super::*;
use crate::infer_catalog_call;
use runmat_types::{
    CallRequest, CellFact, DimensionFact, LiteralContext, LiteralValue, NumericClass,
    NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

fn request(shape: ShapeFact, dim: Option<f64>) -> CallRequest {
    let cell = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(
                runmat_types::DynamicReason::RuntimeValue,
            )),
            elements: Vec::new(),
            elements_complete: false,
        }),
        shape,
        StorageFact::Dense,
    );
    let mut arguments = vec![cell, ValueFact::scalar(ValueKindFact::String)];
    let mut literals = vec![LiteralValue::Unknown, LiteralValue::Unknown];
    if let Some(dim) = dim {
        arguments.push(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })));
        literals.push(LiteralValue::Number(dim));
    }
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn selected_dimension_becomes_singleton() {
    let request = request(
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(2), DimensionFact::Known(3)],
        },
        Some(1.0),
    );
    let inference = infer_catalog_call(&CELL2STRUCT_CATALOG_ENTRY, &request);
    assert_eq!(
        inference.outputs[0].shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(1), DimensionFact::Known(3)]
        }
    );
    assert!(matches!(inference.outputs[0].kind, ValueKindFact::Cell(_)));
}

#[test]
fn singleton_remaining_shape_returns_a_structure() {
    let inference = infer_catalog_call(
        &CELL2STRUCT_CATALOG_ENTRY,
        &request(
            ShapeFact::Shaped {
                dims: vec![DimensionFact::Known(2), DimensionFact::Known(1)],
            },
            None,
        ),
    );
    assert!(matches!(
        inference.outputs[0].kind,
        ValueKindFact::Struct(_)
    ));
}

#[test]
fn invalid_target_and_dimension_are_diagnostics() {
    let request = CallRequest {
        arguments: vec![
            ValueFact::scalar(ValueKindFact::Logical),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::Shaped {
                    dims: vec![DimensionFact::Known(1), DimensionFact::Known(2)],
                },
                StorageFact::Dense,
            ),
        ],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    assert_eq!(
        infer_catalog_call(&CELL2STRUCT_CATALOG_ENTRY, &request)
            .diagnostics
            .len(),
        2
    );
}

#[test]
fn dynamic_dimension_does_not_assume_the_default() {
    let mut request = request(
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(2), DimensionFact::Known(3)],
        },
        Some(1.0),
    );
    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::Unknown,
    ]);

    let inference = infer_catalog_call(&CELL2STRUCT_CATALOG_ENTRY, &request);
    assert_eq!(inference.outputs[0].shape, ShapeFact::Unknown);
}
