use super::*;
use crate::infer_catalog_call;
use runmat_types::{
    CallRequest, DimensionFact, DynamicReason, LiteralContext, ObjectFact, OutputSelection,
    RequestedOutputCount, ShapeFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn infer(arguments: Vec<ValueFact>) -> runmat_types::CallInference {
    infer_catalog_call(
        &FIELDNAMES_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    )
}

fn structure(names: &[&str], complete: bool) -> ValueFact {
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
        complete,
    )))
}

#[test]
fn scalar_structure_produces_an_exact_column_of_character_rows() {
    let result = infer(vec![structure(&["first", "second"], true)]);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(2), DimensionFact::Known(1)]
        }
    );
    let ValueKindFact::Cell(cell) = &result.outputs[0].kind else {
        panic!("expected cell fact")
    };
    assert_eq!(cell.element.kind, ValueKindFact::Character);
    assert_eq!(
        cell.element.shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(1), DimensionFact::Unknown]
        }
    );
}

#[test]
fn typed_array_schema_and_incomplete_schema_are_typed() {
    let array = ValueFact::proven(
        structure(&["a", "shared"], true).kind,
        ShapeFact::from(vec![Some(1), Some(2)]),
        runmat_types::StorageFact::Dense,
    );
    assert_eq!(
        infer(vec![array]).outputs[0].shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(2), DimensionFact::Known(1)]
        }
    );
    assert_eq!(
        infer(vec![structure(&["known"], false)]).outputs[0].shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Unknown, DimensionFact::Known(1)]
        }
    );
}

#[test]
fn object_properties_and_invalid_inputs_are_typed() {
    let object = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: None,
        properties: BTreeMap::from([(
            "Value".into(),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        )]),
        properties_complete: true,
        handle_semantics: Some(false),
    }));
    assert_eq!(
        infer(vec![object]).outputs[0].shape,
        ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(1), DimensionFact::Known(1)]
        }
    );
    let invalid = infer(vec![ValueFact::scalar(ValueKindFact::Logical)]);
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-FIELDNAMES-TARGET"));
}
