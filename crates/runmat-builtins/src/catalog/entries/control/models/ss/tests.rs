use super::*;
use runmat_types::*;

#[test]
fn precision_boundary_errors_are_canonical_and_distinct_from_extensions() {
    let errors = SS_DESCRIPTOR.errors;
    for expected in [
        SS_ERROR_INVALID_MATRIX.identifier,
        SS_ERROR_INVALID_SAMPLE_TIME.identifier,
    ] {
        assert!(errors.iter().any(|error| error.identifier == expected));
    }
    assert_eq!(
        SS_ERROR_INVALID_MATRIX.identifier,
        Some("RunMat:ss:InvalidMatrix")
    );
    assert_eq!(
        SS_ERROR_INVALID_SAMPLE_TIME.identifier,
        Some("RunMat:ss:InvalidSampleTime")
    );
    assert_ne!(
        SS_ERROR_INVALID_MATRIX.identifier,
        SS_INTEGER_MATRIX_EXTENSION.error_identifier
    );
}

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

fn call(args: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallInference {
    crate::infer_catalog_call(
        &SS_CATALOG_ENTRY,
        &CallRequest {
            arguments: args,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
    )
}

#[test]
fn known_matrix_dimensions_flow_to_complete_host_object() {
    let result = call(
        vec![numeric(2, 2), numeric(2, 1), numeric(1, 2), numeric(1, 1)],
        vec![],
        1,
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let value = &result.outputs[0];
    assert_eq!(value.residency, ResidencyFact::Host);
    assert_eq!(value.alias, AliasFact::Unique);
    let ValueKindFact::Object(object) = &value.kind else {
        panic!("ss object expected")
    };
    assert_eq!(object.runtime_class, Some(standard::STATE_SPACE.owned()));
    assert!(object.properties_complete);
    assert_eq!(
        object.properties["A"].shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );
    assert_eq!(
        object.properties["B"].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
    assert_eq!(
        object.properties["InputDelay"].shape,
        ShapeFact::from(vec![Some(1), Some(1)])
    );
    assert_eq!(
        object.properties["StateName"].shape,
        ShapeFact::from(vec![Some(2), Some(1)])
    );
    assert!(result.effects.0.contains(&EffectKind::MaySuspend));
}

#[test]
fn unknown_shapes_do_not_invent_matrix_dimensions() {
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let result = call(
        vec![unknown.clone(), unknown.clone(), unknown.clone(), unknown],
        vec![],
        1,
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let ValueKindFact::Object(object) = &result.outputs[0].kind else {
        panic!("ss object")
    };
    assert_eq!(
        object.properties["A"].shape,
        ShapeFact::from(vec![None, None])
    );
    assert_eq!(
        object.properties["StateName"].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
}

#[test]
fn invalid_dimension_option_and_sample_time_are_diagnosed() {
    let base = vec![numeric(2, 2), numeric(1, 1), numeric(1, 2), numeric(1, 1)];
    assert!(call(base.clone(), vec![], 1)
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-SS-DIMENSIONS"));
    let mut valid = base;
    valid[1] = numeric(2, 1);
    valid.push(ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(4)]),
        StorageFact::Dense,
    ));
    valid.push(numeric(1, 1));
    assert!(call(
        valid.clone(),
        vec![LiteralValue::Unknown; 4]
            .into_iter()
            .chain([
                LiteralValue::Character("oops".into()),
                LiteralValue::Number(0.1)
            ])
            .collect(),
        1
    )
    .diagnostics
    .iter()
    .any(|d| d.code == "RM-CATALOG-SS-OPTION"));
    assert!(call(
        valid,
        vec![LiteralValue::Unknown; 4]
            .into_iter()
            .chain([
                LiteralValue::Character("Ts".into()),
                LiteralValue::Number(-1.0)
            ])
            .collect(),
        1
    )
    .diagnostics
    .iter()
    .any(|d| d.code == "RM-CATALOG-SS-SAMPLE-TIME"));
}

#[test]
fn known_sparse_matrix_and_multiline_option_name_are_rejected() {
    let mut sparse = numeric(1, 1);
    sparse.storage = StorageFact::Sparse;
    let mut args = vec![sparse, numeric(1, 1), numeric(1, 1), numeric(1, 1)];
    assert!(call(args.clone(), vec![], 1)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-SS-MATRIX"));
    args[0] = numeric(1, 1);
    args.extend([
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(2), Some(2)]),
            StorageFact::Dense,
        ),
        numeric(1, 1),
    ]);
    assert!(call(args, vec![], 1)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-SS-OPTION"));
}

#[test]
fn output_arity_is_fixed() {
    let args = vec![numeric(1, 1); 4];
    assert!(call(args.clone(), vec![], 0).outputs.is_empty());
    assert!(!call(args, vec![], 2).diagnostics.is_empty());
}
