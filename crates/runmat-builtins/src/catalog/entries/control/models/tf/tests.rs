use super::*;
use runmat_types::*;

fn numeric(rows: usize, columns: usize) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}

fn text() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    )
}

fn call(args: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallInference {
    crate::infer_catalog_call(
        &TF_CATALOG_ENTRY,
        &CallRequest {
            arguments: args,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
    )
}

#[test]
fn scalar_coefficients_produce_new_host_tf_with_known_row_shapes() {
    let result = call(vec![numeric(1, 1), numeric(1, 1)], vec![], 1);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let value = &result.outputs[0];
    assert_eq!(value.residency, ResidencyFact::Host);
    assert_eq!(value.alias, AliasFact::Unique);
    let ValueKindFact::Object(object) = &value.kind else {
        panic!("tf object expected")
    };
    assert_eq!(
        object.runtime_class,
        Some(standard::TRANSFER_FUNCTION.owned())
    );
    assert!(object.properties_complete);
    assert_eq!(
        object.properties["Numerator"].shape,
        ShapeFact::from(vec![Some(1), Some(1)])
    );
    assert_eq!(object.properties["Ts"].shape, ShapeFact::Scalar);
    assert!(result.effects.0.contains(&EffectKind::MaySuspend));
}

#[test]
fn variable_shorthand_has_known_coefficient_lengths() {
    let result = call(vec![text()], vec![LiteralValue::Character("s".into())], 1);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("tf object")
    };
    assert_eq!(
        object.properties["Numerator"].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
    assert_eq!(
        object.properties["Denominator"].shape,
        ShapeFact::from(vec![Some(1), Some(1)])
    );
}

#[test]
fn vector_length_is_dynamic_because_leading_zeros_are_trimmed() {
    let result = call(vec![numeric(1, 3), numeric(1, 3)], vec![], 1);
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("tf object")
    };
    assert_eq!(
        object.properties["Numerator"].shape,
        ShapeFact::from(vec![Some(1), None])
    );
    assert_eq!(
        object.properties["Numerator"].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real
        })
    );
}

#[test]
fn complex_coefficients_retain_known_complex_double_facts() {
    let complex = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    let result = call(vec![complex, numeric(1, 1)], vec![], 1);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("tf object")
    };
    assert_eq!(
        object.properties["Numerator"].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        })
    );
}

#[test]
fn invalid_shapes_options_and_sample_time_are_diagnosed() {
    assert!(call(vec![numeric(2, 2), numeric(1, 2)], vec![], 1)
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-TF-COEFFICIENTS"));
    assert!(
        call(vec![text()], vec![LiteralValue::Character("bad".into())], 1)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-TF-VARIABLE")
    );
    let args = vec![numeric(1, 1), numeric(1, 1), text(), numeric(1, 1)];
    let prefix = vec![LiteralValue::Unknown; 2];
    assert!(call(
        args.clone(),
        prefix
            .clone()
            .into_iter()
            .chain([
                LiteralValue::Character("bad".into()),
                LiteralValue::Number(0.1)
            ])
            .collect(),
        1
    )
    .diagnostics
    .iter()
    .any(|d| d.code == "RM-CATALOG-TF-OPTION"));
    assert!(call(
        args,
        prefix
            .into_iter()
            .chain([
                LiteralValue::Character("Ts".into()),
                LiteralValue::Number(-2.0)
            ])
            .collect(),
        1
    )
    .diagnostics
    .iter()
    .any(|d| d.code == "RM-CATALOG-TF-SAMPLE-TIME"));
}

#[test]
fn known_sparse_coefficients_and_multiline_shorthand_are_rejected() {
    let mut sparse = numeric(1, 2);
    sparse.storage = StorageFact::Sparse;
    assert!(call(vec![sparse, numeric(1, 1)], vec![], 1)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-TF-COEFFICIENTS"));
    let multiline = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(2), Some(2)]),
        StorageFact::Dense,
    );
    assert!(call(vec![multiline], vec![], 1)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-TF-ARITY"));
}

#[test]
fn logical_sample_time_matches_runtime_scalar_admission() {
    let logical = ValueFact::scalar(ValueKindFact::Logical);
    let result = call(vec![numeric(1, 1), numeric(1, 1), logical], vec![], 1);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
}

#[test]
fn output_arity_is_fixed() {
    let args = vec![numeric(1, 1), numeric(1, 1)];
    assert!(call(args.clone(), vec![], 0).outputs.is_empty());
    assert!(!call(args, vec![], 2).diagnostics.is_empty());
}
