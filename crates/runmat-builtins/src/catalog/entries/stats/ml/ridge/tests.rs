use super::*;
use runmat_types::{
    CallRequest, DimensionFact, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn numeric(rows: usize, cols: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(cols)]),
        StorageFact::Dense,
    )
}

fn request(y: ValueFact, x: ValueFact, k: ValueFact, scaled: Option<LiteralValue>) -> CallRequest {
    let mut arguments = vec![y, x, k];
    let mut literals = vec![LiteralValue::Unknown; 3];
    if let Some(scaled) = scaled {
        arguments.push(numeric(1, 1));
        literals.push(scaled);
    }
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
    }
}

#[test]
fn output_class_and_scaled_shape_follow_runtime() {
    for (scaled, expected_rows) in [
        (None, 3),
        (Some(LiteralValue::Number(0.0)), 4),
        (Some(LiteralValue::Bool(true)), 3),
    ] {
        let call = request(numeric(7, 1), numeric(7, 3), numeric(1, 2), scaled);
        let inferred = inference::infer(&call, &RIDGE_CATALOG_ENTRY);
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
        assert!(matches!(
            inferred.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })
        ));
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(expected_rows), Some(2)])
        );
    }
}

#[test]
fn unknown_flag_does_not_invent_output_row_count() {
    let call = request(
        numeric(7, 1),
        numeric(7, 3),
        numeric(1, 2),
        Some(LiteralValue::Unknown),
    );
    let inferred = inference::infer(&call, &RIDGE_CATALOG_ENTRY);
    let ShapeFact::Shaped { dims } = &inferred.outputs[0].shape else {
        panic!("ridge returns a matrix");
    };
    assert_eq!(dims[0], DimensionFact::Unknown);
    assert_eq!(dims[1], DimensionFact::Known(2));
}

#[test]
fn mismatched_rows_and_nonvector_parameters_have_diagnostics() {
    let call = request(numeric(6, 1), numeric(7, 3), numeric(2, 2), None);
    let inferred = inference::infer(&call, &RIDGE_CATALOG_ENTRY);
    assert!(inferred
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-RIDGE-ROWS"));
    assert!(inferred
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-CATALOG-RIDGE-VECTOR"));
}
