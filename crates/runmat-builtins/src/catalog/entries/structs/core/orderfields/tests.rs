use super::*;
use crate::infer_catalog_call;
use runmat_types::{
    CallRequest, DimensionFact, DynamicReason, LiteralContext, OutputSelection,
    RequestedOutputCount, ShapeFact, StorageFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn structure(names: &[&str]) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        names
            .iter()
            .map(|name| {
                (
                    (*name).into(),
                    ValueFact::unknown(DynamicReason::RuntimeValue),
                )
            })
            .collect::<BTreeMap<_, _>>(),
        true,
    )))
}

fn infer(arguments: Vec<ValueFact>, outputs: RequestedOutputCount) -> runmat_types::CallInference {
    infer_catalog_call(
        &ORDERFIELDS_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(outputs),
        },
    )
}

#[test]
fn preserves_structure_fact_and_types_permutation_column() {
    let input = structure(&["first", "second"]);
    let result = infer(vec![input.clone()], RequestedOutputCount::Exactly(2));
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].kind, input.kind);
    assert_eq!(
        result.outputs[1].shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(2), DimensionFact::Known(1)]
        }
    );
}

#[test]
fn typed_array_retains_shape_and_uniform_field_count() {
    let array = ValueFact::proven(
        structure(&["a", "b"]).kind,
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    let result = infer(vec![array.clone()], RequestedOutputCount::Exactly(2));
    assert_eq!(result.outputs[0].shape, array.shape);
    assert_eq!(
        result.outputs[1].shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(2), DimensionFact::Known(1)]
        }
    );
}

#[test]
fn incomplete_schema_keeps_permutation_length_dynamic() {
    let input = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
        BTreeMap::new(),
        false,
    )));
    let result = infer(vec![input], RequestedOutputCount::Exactly(2));
    assert_eq!(
        result.outputs[1].shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Unknown, DimensionFact::Known(1)]
        }
    );
}

#[test]
fn one_field_permutation_uses_the_scalar_carrier() {
    let result = infer(vec![structure(&["only"])], RequestedOutputCount::Exactly(2));
    assert_eq!(result.outputs[1].shape, ShapeFact::Scalar);
    assert_eq!(result.outputs[1].storage, StorageFact::Scalar);
}

#[test]
fn diagnoses_invalid_target_order_and_arity() {
    let result = infer(
        vec![
            ValueFact::scalar(ValueKindFact::Logical),
            ValueFact::scalar(ValueKindFact::Logical),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        RequestedOutputCount::One,
    );
    for code in [
        "RM-CATALOG-ORDERFIELDS-TARGET",
        "RM-CATALOG-ORDERFIELDS-ORDER",
        "RM-CATALOG-ORDERFIELDS-ARITY",
    ] {
        assert!(result.diagnostics.iter().any(|item| item.code == code));
    }
}
