use super::*;
use runmat_types::*;

fn observations(rows: usize, columns: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}

fn infer_with(selection: Vec<(ValueFact, LiteralValue)>, count: usize) -> CallInference {
    let mut arguments = vec![observations(4, 2), observations(3, 2)];
    let mut literals = vec![LiteralValue::Unknown, LiteralValue::Unknown];
    for (fact, literal) in selection {
        arguments.push(fact);
        literals.push(literal);
    }
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
        },
        &PDIST2_CATALOG_ENTRY,
    )
}

#[test]
fn matrix_and_selected_shapes_follow_runtime_order() {
    let regular = infer_with(vec![], 1);
    assert!(regular.diagnostics.is_empty(), "{:?}", regular.diagnostics);
    assert_eq!(
        regular.outputs[0].shape,
        ShapeFact::from(vec![Some(4), Some(3)])
    );
    let selection = infer_with(
        vec![
            (
                ValueFact::scalar(ValueKindFact::String),
                LiteralValue::String("Smallest".to_owned()),
            ),
            (
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
                LiteralValue::Number(2.0),
            ),
        ],
        2,
    );
    assert!(
        selection.diagnostics.is_empty(),
        "{:?}",
        selection.diagnostics
    );
    assert_eq!(
        selection.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert_eq!(
        selection.outputs[1].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert_eq!(
        selection.outputs[1].numeric().unwrap().class,
        NumericClass::Double
    );
}

#[test]
fn second_output_requires_selection_and_unknown_name_stays_dynamic() {
    assert!(!infer_with(vec![], 2).diagnostics.is_empty());
    let dynamic = infer_with(
        vec![
            (
                ValueFact::scalar(ValueKindFact::String),
                LiteralValue::Unknown,
            ),
            (
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
                LiteralValue::Number(1.0),
            ),
        ],
        2,
    );
    assert_eq!(
        dynamic.outputs[0].shape,
        ShapeFact::from(vec![None, Some(3)])
    );
}
