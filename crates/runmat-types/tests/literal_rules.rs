use runmat_types::{
    infer_literal, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    ShapeFact, ValueKindFact,
};

#[test]
fn scalar_literal_categories_preserve_exact_semantic_kind() {
    assert_eq!(
        infer_literal(&LiteralValue::Real {
            text: "1.234567890123456789".into(),
            class: NumericClass::Double,
        })
        .fact
        .kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(
        infer_literal(&LiteralValue::Integer {
            text: "18446744073709551615".into(),
            class: NumericClass::UInt64,
        })
        .fact
        .kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(
        infer_literal(&LiteralValue::Complex {
            real: "1.25".into(),
            imaginary: "-2.5".into(),
            class: NumericClass::Single,
        })
        .fact
        .kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        })
    );
    let characters = infer_literal(&LiteralValue::Character("λx".into())).fact;
    assert_eq!(characters.kind, ValueKindFact::Character);
    assert_eq!(characters.shape, ShapeFact::from(vec![Some(1), Some(2)]));
    assert_eq!(
        infer_literal(&LiteralValue::String("λx".into())).fact.kind,
        ValueKindFact::String
    );
    assert_eq!(
        infer_literal(&LiteralValue::Symbolic("x + y".into()))
            .fact
            .kind,
        ValueKindFact::Symbolic
    );
}

#[test]
fn vector_and_matrix_literals_preserve_shape_without_inventing_cells() {
    let vector = infer_literal(&LiteralValue::Vector(vec![
        LiteralValue::Number(1.0),
        LiteralValue::Unknown,
        LiteralValue::Number(3.0),
    ]));
    assert_eq!(vector.fact.kind, ValueKindFact::Unknown);
    assert_eq!(vector.fact.shape, ShapeFact::from(vec![Some(1), Some(3)]));

    let matrix = infer_literal(&LiteralValue::Matrix(vec![
        vec![LiteralValue::Number(1.0), LiteralValue::Number(2.0)],
        vec![LiteralValue::Number(3.0), LiteralValue::Number(4.0)],
    ]));
    assert_eq!(matrix.fact.shape, ShapeFact::from(vec![Some(2), Some(2)]));
    assert!(matrix.diagnostics.is_empty());

    let ragged = infer_literal(&LiteralValue::Matrix(vec![
        vec![LiteralValue::Number(1.0)],
        vec![LiteralValue::Number(2.0), LiteralValue::Number(3.0)],
    ]));
    assert_eq!(ragged.diagnostics[0].code, "RM-TYPE-LITERAL-MATRIX");
}

#[test]
fn literal_context_keeps_integer_dimensions_exact_and_options_distinct() {
    let context = LiteralContext::new(vec![
        LiteralValue::Integer {
            text: usize::MAX.to_string(),
            class: NumericClass::UInt64,
        },
        LiteralValue::Keyword("OmItNaN".into()),
    ]);
    assert_eq!(context.numeric_dims()[0], Some(usize::MAX));
    assert_eq!(context.literal_string_at(1).as_deref(), Some("omitnan"));
}
