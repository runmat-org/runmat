use super::*;
use runmat_types::{
    CallRequest, CellFact, LiteralContext, OutputSelection, RequestedOutputCount, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

fn infer(arguments: Vec<ValueFact>, outputs: usize) -> CallInference {
    crate::infer_catalog_call(
        &DLUPDATE_CATALOG_ENTRY,
        &CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
    )
}

#[test]
fn cell_tree_preserves_container_geometry_but_not_unknown_callback_leaf_type() {
    let leaf = ValueFact::scalar(ValueKindFact::Numeric(runmat_types::NumericFact {
        class: runmat_types::NumericClass::Double,
        domain: runmat_types::NumericDomain::Real,
    }));
    let tree = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(leaf.clone()),
            elements: vec![leaf.clone(), leaf],
            elements_complete: true,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    let callable = ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue);
    let result = infer(vec![callable, tree.clone(), tree], 2);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs.len(), 2);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
    let ValueKindFact::Cell(cell) = &result.outputs[0].kind else {
        panic!("expected cell tree")
    };
    assert!(matches!(cell.element.kind, ValueKindFact::Unknown));
    assert_eq!(cell.elements.len(), 2);
}

#[test]
fn known_incompatible_container_shape_and_noncallable_diagnose() {
    let cell = |shape: Vec<Option<usize>>| {
        ValueFact::proven(
            ValueKindFact::Cell(CellFact {
                element: Box::new(ValueFact::unknown(
                    runmat_types::DynamicReason::RuntimeValue,
                )),
                elements: Vec::new(),
                elements_complete: false,
            }),
            ShapeFact::from(shape),
            StorageFact::Dense,
        )
    };
    let result = infer(
        vec![
            ValueFact::scalar(ValueKindFact::String),
            cell(vec![Some(1), Some(2)]),
            cell(vec![Some(2), Some(1)]),
        ],
        1,
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|diag| diag.code == "RM-DL-DLUPDATE-CALLABLE"));
    assert!(result
        .diagnostics
        .iter()
        .any(|diag| diag.code == "RM-DL-DLUPDATE-TREE"));
    assert_eq!(
        DLUPDATE_DESCRIPTOR.errors[0].identifier,
        Some("RunMat:deepLearning:InvalidInput")
    );
}
