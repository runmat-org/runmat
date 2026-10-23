use super::*;
use crate::*;
use runmat_types::*;

fn numeric(class: NumericClass, shape: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}
fn call(
    entry: &BuiltinCatalogEntry,
    arguments: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
    outputs: usize,
) -> CallInference {
    crate::infer_catalog_call(
        entry,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
    )
}
#[test]
fn complete_family_has_canonical_bindings_asserted_examples_and_bounded_outputs() {
    let counts = [1, 2, 2, 1, 1, 3, 3, 3, 2, 3, 2, 2];
    let arities = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2];
    let example_counts = [7, 7, 8, 8, 1, 9, 6, 6, 8, 4, 7, 6];
    let mut entries = Vec::new();
    super::extend_entries(&mut entries);
    assert_eq!(entries.len(), counts.len());
    for (((entry, count), arity), examples) in
        entries.iter().zip(counts).zip(arities).zip(example_counts)
    {
        assert_eq!(entry.bindings, &REQUIRED_DEFAULT_BINDING);
        assert_eq!(entry.documentation.examples.len(), examples);
        assert!(entry.documentation.examples.iter().all(|example| matches!(
            example.verification,
            BuiltinExampleVerification::Assertions { .. }
        )));
        let arguments = vec![numeric(NumericClass::Single, &[3, 2]); arity];
        let inferred = call(entry, arguments.clone(), vec![], count);
        assert!(
            inferred.diagnostics.is_empty(),
            "{}: {:?}",
            entry.identity.name,
            inferred.diagnostics
        );
        assert_eq!(inferred.outputs.len(), count);
        assert!(!call(entry, arguments, vec![], count + 1)
            .diagnostics
            .is_empty());
        assert!(!call(entry, vec![], vec![], 1).diagnostics.is_empty());
    }
}
#[test]
fn classes_shapes_and_host_predicates_are_distinct() {
    let mut input = numeric(NumericClass::UInt64, &[2, 3]);
    input.residency = ResidencyFact::Device {
        provider: Some("fixture".into()),
    };
    let sorted = call(&SORT_CATALOG_ENTRY, vec![input.clone()], vec![], 2);
    assert_eq!(sorted.outputs[0].kind, input.kind);
    assert_eq!(sorted.outputs[0].shape, input.shape);
    assert_eq!(
        sorted.outputs[1].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(sorted.outputs[1].residency, input.residency);
    for entry in [&ISSORTED_CATALOG_ENTRY, &ISSORTEDROWS_CATALOG_ENTRY] {
        let result = call(entry, vec![input.clone()], vec![], 1);
        assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
        assert_eq!(result.outputs[0].shape, ShapeFact::Scalar);
        assert_eq!(result.outputs[0].residency, ResidencyFact::Host);
    }
    for entry in [
        &UNION_CATALOG_ENTRY,
        &INTERSECT_CATALOG_ENTRY,
        &SETDIFF_CATALOG_ENTRY,
        &SETXOR_CATALOG_ENTRY,
    ] {
        let result = call(
            entry,
            vec![
                numeric(NumericClass::Single, &[1, 3]),
                numeric(NumericClass::Double, &[1, 2]),
            ],
            vec![],
            1,
        );
        assert_eq!(
            result.outputs[0].numeric().unwrap().class,
            NumericClass::Single
        );
    }
}
#[test]
fn known_controls_diagnose_invalid_values_and_preserve_data_dependent_extents() {
    let data = numeric(NumericClass::Double, &[3, 2]);
    let control = numeric(NumericClass::Double, &[1, 1]);
    let bad_dimension = call(
        &SORT_CATALOG_ENTRY,
        vec![data.clone(), control.clone()],
        vec![LiteralValue::Unknown, LiteralValue::Number(0.0)],
        1,
    );
    assert!(!bad_dimension.diagnostics.is_empty());
    let bad_tolerance = call(
        &ISMEMBERTOL_CATALOG_ENTRY,
        vec![data.clone(), data.clone(), control],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Number(0.0),
        ],
        1,
    );
    assert!(!bad_tolerance.diagnostics.is_empty());
    let text = ValueFact::scalar(ValueKindFact::Character);
    let rows = call(
        &UNIQUE_CATALOG_ENTRY,
        vec![data, text],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Character("rows".into()),
        ],
        3,
    );
    assert!(rows.diagnostics.is_empty());
    assert_eq!(rows.outputs[0].shape, ShapeFact::from(vec![None, Some(2)]));
    assert_eq!(
        rows.outputs[2].shape,
        ShapeFact::from(vec![Some(3), Some(1)])
    );
}

#[test]
fn set_orientations_and_setdiff_left_class_are_typed_per_declaration() {
    let a = numeric(NumericClass::Double, &[1, 3]);
    let b = numeric(NumericClass::Single, &[3, 1]);
    for entry in [&UNION_CATALOG_ENTRY, &INTERSECT_CATALOG_ENTRY] {
        let inferred = call(entry, vec![a.clone(), b.clone()], vec![], 1);
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![None, Some(1)])
        );
        assert_eq!(
            inferred.outputs[0].numeric().unwrap().class,
            NumericClass::Single
        );
        let inferred = call(
            entry,
            vec![a.clone(), numeric(NumericClass::Single, &[1, 3])],
            vec![],
            1,
        );
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(1), None])
        );
    }
    let inferred = call(&SETDIFF_CATALOG_ENTRY, vec![a, b], vec![], 2);
    assert_eq!(
        inferred.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
    assert_eq!(
        inferred.outputs[1].shape,
        ShapeFact::from(vec![None, Some(1)])
    );
    let logical = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let inferred = call(
        &SETDIFF_CATALOG_ENTRY,
        vec![logical, numeric(NumericClass::Double, &[1, 1])],
        vec![],
        1,
    );
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Logical);
    let inferred = call(
        &SETDIFF_CATALOG_ENTRY,
        vec![
            numeric(NumericClass::Double, &[1, 3]),
            numeric(NumericClass::UInt64, &[1, 2]),
        ],
        vec![],
        2,
    );
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(
        inferred.outputs[0].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
    assert_eq!(
        inferred.outputs[1].numeric().unwrap().class,
        NumericClass::Double
    );
}
