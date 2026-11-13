use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::{
    CallInference, CallRequest, CellFact, LiteralContext, LiteralValue, NumericClass,
    NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

fn numeric(class: NumericClass, shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn complete_cell(elements: Vec<ValueFact>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::unknown(
                runmat_types::DynamicReason::RuntimeValue,
            )),
            elements_complete: true,
            elements: elements.clone(),
        }),
        ShapeFact::from(vec![Some(1), Some(elements.len())]),
        StorageFact::Dense,
    )
}

fn call(
    name: &str,
    arguments: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
    outputs: RequestedOutputCount,
) -> CallInference {
    infer_catalog_call(
        builtin_catalog_entry_by_name(name).expect("catalog identity"),
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(outputs),
        },
    )
}

#[test]
fn find_preserves_orientation_and_exact_value_class() {
    let output = call(
        "find",
        vec![numeric(
            NumericClass::UInt64,
            ShapeFact::from(vec![Some(1), Some(5)]),
        )],
        vec![],
        RequestedOutputCount::One,
    );
    assert_eq!(
        output.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
    let multi = call(
        "find",
        vec![numeric(
            NumericClass::UInt64,
            ShapeFact::from(vec![Some(2), Some(3)]),
        )],
        vec![],
        RequestedOutputCount::Exactly(3),
    );
    assert_eq!(
        multi.outputs[2].numeric().map(|fact| fact.class),
        Some(NumericClass::UInt64)
    );
    assert_eq!(multi.outputs[2].shape, ShapeFact::from(vec![None, Some(1)]));
    assert_eq!(
        multi.outputs[0].numeric().map(|fact| fact.class),
        Some(NumericClass::Double)
    );
}

#[test]
fn find_options_validate_without_claiming_runtime_match_count() {
    let input = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(1), Some(4)]),
    );
    let invalid = call(
        "find",
        vec![
            input.clone(),
            numeric(NumericClass::Double, ShapeFact::Scalar),
        ],
        vec![LiteralValue::Unknown, LiteralValue::Number(0.0)],
        RequestedOutputCount::One,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-FIND-LIMIT"));
    let valid = call(
        "find",
        vec![
            input,
            numeric(NumericClass::Double, ShapeFact::Scalar),
            ValueFact::scalar(ValueKindFact::String),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
            LiteralValue::String("last".into()),
        ],
        RequestedOutputCount::One,
    );
    assert!(valid.diagnostics.is_empty());
    assert_eq!(valid.outputs[0].shape, ShapeFact::from(vec![Some(1), None]));
}

#[test]
fn find_empty_matrix_and_row_have_distinct_exact_shapes() {
    let matrix = call(
        "find",
        vec![numeric(
            NumericClass::Double,
            ShapeFact::from(vec![Some(0), Some(0)]),
        )],
        vec![],
        RequestedOutputCount::One,
    );
    assert_eq!(
        matrix.outputs[0].shape,
        ShapeFact::from(vec![Some(0), Some(0)])
    );
    let row = call(
        "find",
        vec![numeric(
            NumericClass::Double,
            ShapeFact::from(vec![Some(1), Some(0)]),
        )],
        vec![],
        RequestedOutputCount::One,
    );
    assert_eq!(
        row.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(0)])
    );
}

#[test]
fn find_scalar_contents_orientation_and_output_count_stay_sound() {
    let scalar = numeric(NumericClass::Double, ShapeFact::Scalar);
    for (literal, expected) in [
        (LiteralValue::Number(0.0), vec![Some(0), Some(0)]),
        (LiteralValue::Number(7.0), vec![Some(1), Some(1)]),
    ] {
        let inferred = call(
            "find",
            vec![scalar.clone()],
            vec![literal],
            RequestedOutputCount::One,
        );
        assert_eq!(inferred.outputs[0].shape, ShapeFact::from(expected));
    }
    let unknown_scalar = call("find", vec![scalar], vec![], RequestedOutputCount::One);
    assert_eq!(
        unknown_scalar.outputs[0].shape,
        ShapeFact::from(vec![None, None])
    );
    for input_shape in [
        ShapeFact::Unknown,
        ShapeFact::from(vec![Some(1), None]),
        ShapeFact::from(vec![None, Some(3)]),
    ] {
        let inferred = call(
            "find",
            vec![numeric(NumericClass::Double, input_shape)],
            vec![],
            RequestedOutputCount::CurrentFunctionNargout,
        );
        assert_eq!(inferred.outputs[0].shape, ShapeFact::from(vec![None, None]));
    }
}

#[test]
fn ind2sub_uses_literal_dimensions_not_size_vector_shape() {
    let size = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(1), Some(2)]),
    );
    let index = numeric(
        NumericClass::UInt64,
        ShapeFact::from(vec![Some(2), Some(3)]),
    );
    let unknown = call(
        "ind2sub",
        vec![size.clone(), index.clone()],
        vec![],
        RequestedOutputCount::One,
    );
    assert_eq!(
        unknown.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    let known = call(
        "ind2sub",
        vec![size, index],
        vec![LiteralValue::Vector(vec![
            LiteralValue::Number(2.0),
            LiteralValue::Number(3.0),
        ])],
        RequestedOutputCount::Exactly(2),
    );
    assert_eq!(known.outputs.len(), 2);
    assert_eq!(
        known.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert_eq!(
        known.outputs[1].numeric().map(|fact| fact.class),
        Some(NumericClass::Double)
    );
}

#[test]
fn sub2ind_preserves_broadcast_shape_and_reports_invalid_count() {
    let size = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(1), Some(2)]),
    );
    let rows = numeric(NumericClass::Int64, ShapeFact::from(vec![Some(3), Some(1)]));
    let cols = numeric(NumericClass::UInt64, ShapeFact::Scalar);
    let output = call(
        "sub2ind",
        vec![size.clone(), rows.clone(), cols],
        vec![LiteralValue::Vector(vec![
            LiteralValue::Number(3.0),
            LiteralValue::Number(4.0),
        ])],
        RequestedOutputCount::One,
    );
    assert_eq!(output.outputs[0].shape, rows.shape);
    assert_eq!(
        output.outputs[0].numeric().map(|fact| fact.class),
        Some(NumericClass::Double)
    );
    assert!(output.diagnostics.is_empty());
    let invalid = call(
        "sub2ind",
        vec![size, rows],
        vec![LiteralValue::Vector(vec![
            LiteralValue::Number(3.0),
            LiteralValue::Number(4.0),
        ])],
        RequestedOutputCount::One,
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-SUB2IND-COUNT"));
}

#[test]
fn size_literals_reject_definite_invalid_dimensions_without_erasing_unknowns() {
    let size = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(1), Some(2)]),
    );
    let index = numeric(NumericClass::Double, ShapeFact::Scalar);
    for bad in [
        LiteralValue::Number(-2.0),
        LiteralValue::Number(1.5),
        LiteralValue::Number(f64::INFINITY),
        LiteralValue::Number(usize::MAX as f64),
    ] {
        for name in ["ind2sub", "sub2ind"] {
            let inferred = call(
                name,
                vec![size.clone(), index.clone()],
                vec![LiteralValue::Vector(vec![
                    LiteralValue::Number(2.0),
                    bad.clone(),
                ])],
                RequestedOutputCount::One,
            );
            assert!(
                inferred
                    .diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.code.ends_with("-SIZE")),
                "{name}: {bad:?}"
            );
        }
    }
    let scalar_size = call(
        "ind2sub",
        vec![
            numeric(NumericClass::Double, ShapeFact::Scalar),
            index.clone(),
        ],
        vec![LiteralValue::Number(4.0)],
        RequestedOutputCount::One,
    );
    assert!(scalar_size
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-IND2SUB-SIZE"));
    let unknown = call(
        "ind2sub",
        vec![size, index],
        vec![LiteralValue::Vector(vec![
            LiteralValue::Unknown,
            LiteralValue::Number(2.0),
        ])],
        RequestedOutputCount::One,
    );
    assert!(!unknown
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-IND2SUB-SIZE"));
}

#[test]
fn sub2ind_partial_shapes_refine_without_false_incompatibility() {
    let size = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(1), Some(2)]),
    );
    let partial = numeric(NumericClass::Double, ShapeFact::from(vec![None, Some(1)]));
    let rows = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(3), Some(1)]),
    );
    let inferred = call(
        "sub2ind",
        vec![size.clone(), partial, rows],
        vec![],
        RequestedOutputCount::One,
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
    assert!(!inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-SUB2IND-SHAPE"));
    let incompatible = call(
        "sub2ind",
        vec![
            size.clone(),
            numeric(
                NumericClass::Double,
                ShapeFact::from(vec![Some(2), Some(1)]),
            ),
            numeric(
                NumericClass::Double,
                ShapeFact::from(vec![Some(3), Some(1)]),
            ),
        ],
        vec![],
        RequestedOutputCount::One,
    );
    assert!(incompatible
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-SUB2IND-SHAPE"));
    for (left, right, mismatch) in [
        (
            vec![Some(2), Some(3)],
            vec![Some(2), Some(3), Some(1)],
            true,
        ),
        (vec![Some(2), None], vec![Some(2), Some(3)], false),
        (vec![Some(2), None], vec![Some(2), Some(3), Some(1)], true),
    ] {
        let inferred = call(
            "sub2ind",
            vec![
                size.clone(),
                numeric(NumericClass::Double, ShapeFact::from(left)),
                numeric(NumericClass::Double, ShapeFact::from(right)),
            ],
            vec![],
            RequestedOutputCount::One,
        );
        assert_eq!(
            inferred
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "RM-CATALOG-SUB2IND-SHAPE"),
            mismatch
        );
    }
}

#[test]
fn complete_size_cells_reject_known_invalid_members_and_counts() {
    let scalar = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let invalid_members = [
        ValueFact::scalar(ValueKindFact::String),
        numeric(NumericClass::Double, ShapeFact::Scalar),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })),
        numeric(
            NumericClass::Double,
            ShapeFact::from(vec![Some(1), Some(2)]),
        ),
        ValueFact::scalar(ValueKindFact::Logical),
    ];
    for invalid in invalid_members {
        let size = complete_cell(vec![scalar.clone(), invalid]);
        for name in ["ind2sub", "sub2ind"] {
            let inferred = call(
                name,
                vec![size.clone(), scalar.clone(), scalar.clone()],
                vec![],
                RequestedOutputCount::One,
            );
            assert!(
                inferred
                    .diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.code.ends_with("SIZE-TYPE")),
                "{name}"
            );
        }
    }
    let one = complete_cell(vec![scalar.clone()]);
    let inferred = call(
        "ind2sub",
        vec![one, scalar.clone()],
        vec![],
        RequestedOutputCount::One,
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-IND2SUB-SIZE"));
    let two = complete_cell(vec![scalar.clone(), scalar.clone()]);
    let inferred = call(
        "sub2ind",
        vec![two, scalar.clone()],
        vec![],
        RequestedOutputCount::One,
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-SUB2IND-COUNT"));
    let valid = call(
        "ind2sub",
        vec![complete_cell(vec![scalar.clone(), scalar.clone()]), scalar],
        vec![],
        RequestedOutputCount::One,
    );
    assert!(!valid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code.ends_with("SIZE-TYPE")));
}

#[test]
fn scalarized_index_outputs_and_find_value_storage_are_sound() {
    let scalar = numeric(NumericClass::Double, ShapeFact::Scalar);
    let size = numeric(
        NumericClass::Double,
        ShapeFact::from(vec![Some(1), Some(2)]),
    );
    let ind = call(
        "ind2sub",
        vec![size.clone(), scalar.clone()],
        vec![],
        RequestedOutputCount::One,
    );
    assert_eq!(ind.outputs[0].storage, StorageFact::Scalar);
    let sub = call(
        "sub2ind",
        vec![size, scalar.clone(), scalar.clone()],
        vec![],
        RequestedOutputCount::One,
    );
    assert_eq!(sub.outputs[0].storage, StorageFact::Scalar);
    let find_one = call(
        "find",
        vec![scalar.clone()],
        vec![LiteralValue::Number(2.0)],
        RequestedOutputCount::One,
    );
    assert_eq!(find_one.outputs[0].storage, StorageFact::Scalar);
    let find_values = call(
        "find",
        vec![numeric(NumericClass::UInt64, ShapeFact::Scalar)],
        vec![],
        RequestedOutputCount::Exactly(3),
    );
    assert_eq!(find_values.outputs[2].storage, StorageFact::Unknown);
    let single_values = call(
        "find",
        vec![numeric(NumericClass::Single, ShapeFact::Scalar)],
        vec![],
        RequestedOutputCount::Exactly(3),
    );
    assert_eq!(single_values.outputs[2].storage, StorageFact::Dense);
    for (class, expected_storage) in [
        (NumericClass::Double, StorageFact::Unknown),
        (NumericClass::Single, StorageFact::Dense),
    ] {
        let input = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Complex,
            }),
            ShapeFact::Scalar,
            if class == NumericClass::Single {
                StorageFact::Dense
            } else {
                StorageFact::Scalar
            },
        );
        let inferred = call(
            "find",
            vec![input],
            vec![],
            RequestedOutputCount::Exactly(3),
        );
        assert_eq!(inferred.outputs[2].storage, expected_storage);
    }
}
