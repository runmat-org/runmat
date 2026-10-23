//! Inference oracles moved from the retired Runtime-only resolver adapters.
use crate::*;
use runmat_types::{
    CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    OutputSelection, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn double(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}

fn infer(
    entry: &BuiltinCatalogEntry,
    arguments: Vec<ValueFact>,
    literals: Vec<LiteralValue>,
) -> ValueFact {
    let mut literals = literals;
    literals.resize(arguments.len(), LiteralValue::Unknown);
    let result = infer_catalog_call(
        entry,
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::default(),
        },
    );
    assert!(
        result.diagnostics.is_empty(),
        "{}: {:?}",
        entry.identity.name,
        result.diagnostics
    );
    assert_eq!(result.outputs.len(), 1);
    result.outputs.into_iter().next().unwrap()
}

#[test]
fn constructor_defaults_unknown_dimensions_and_literal_vectors() {
    for entry in [
        &ONES_CATALOG_ENTRY,
        &RAND_CATALOG_ENTRY,
        &RANDN_CATALOG_ENTRY,
        &EYE_CATALOG_ENTRY,
    ] {
        let scalar = infer(entry, vec![], vec![]);
        assert!(scalar.shape.is_proven_equivalent(&ShapeFact::Scalar));
        assert_eq!(scalar.numeric().unwrap().class, NumericClass::Double);
        let unknown = infer(entry, vec![double(ShapeFact::Scalar)], vec![]);
        assert_eq!(unknown.shape.known_dims(), Some(vec![None, None]));
    }
    let vector = double(ShapeFact::from(vec![Some(1), Some(4)]));
    assert_eq!(
        infer(&ONES_CATALOG_ENTRY, vec![vector], vec![])
            .shape
            .known_dims(),
        Some(vec![None; 4])
    );
    let vector = double(ShapeFact::from(vec![Some(1), Some(2)]));
    assert_eq!(
        infer(
            &ONES_CATALOG_ENTRY,
            vec![vector],
            vec![LiteralValue::Vector(vec![
                LiteralValue::Number(2.0),
                LiteralValue::Number(3.0),
            ])]
        )
        .shape
        .known_dims(),
        Some(vec![Some(2), Some(3)])
    );
    let bound = double(ShapeFact::Scalar);
    assert!(infer(&RANDI_CATALOG_ENTRY, vec![bound.clone()], vec![])
        .shape
        .is_proven_equivalent(&ShapeFact::Scalar));
    assert_eq!(
        infer(
            &RANDI_CATALOG_ENTRY,
            vec![bound.clone(), bound.clone(), bound],
            vec![]
        )
        .shape
        .known_dims(),
        Some(vec![None, None])
    );
}

#[test]
fn sequences_keep_row_geometry_and_known_counts() {
    let scalar = double(ShapeFact::Scalar);
    assert_eq!(
        infer(&COLON_CATALOG_ENTRY, vec![scalar.clone(); 2], vec![])
            .shape
            .known_dims(),
        Some(vec![Some(1), None])
    );
    assert_eq!(
        infer(
            &COLON_CATALOG_ENTRY,
            vec![scalar.clone(); 3],
            vec![
                LiteralValue::Number(-2.0),
                LiteralValue::Number(0.02),
                LiteralValue::Number(2.0)
            ]
        )
        .shape
        .known_dims(),
        Some(vec![Some(1), Some(201)])
    );
    for (entry, count) in [
        (&LINSPACE_CATALOG_ENTRY, 100),
        (&LOGSPACE_CATALOG_ENTRY, 50),
    ] {
        // Unlike the old adapter, the selected contract knows the default count.
        assert_eq!(
            infer(entry, vec![scalar.clone(); 2], vec![])
                .shape
                .known_dims(),
            Some(vec![Some(1), Some(count)])
        );
        assert_eq!(
            infer(entry, vec![scalar.clone(); 3], vec![])
                .shape
                .known_dims(),
            Some(vec![Some(1), None])
        );
    }
    assert_eq!(
        infer(&RANDPERM_CATALOG_ENTRY, vec![scalar], vec![])
            .shape
            .known_dims(),
        Some(vec![Some(1), None])
    );
}

#[test]
fn meshgrid_uses_axis_values_geometry_and_normalizes_trailing_singletons() {
    for count in [2, 3] {
        let output = infer(
            &MESHGRID_CATALOG_ENTRY,
            vec![double(ShapeFact::Scalar); count],
            vec![],
        );
        assert!(output
            .shape
            .is_proven_equivalent(&ShapeFact::from(vec![Some(1); count])));
    }
    let output = infer(
        &MESHGRID_CATALOG_ENTRY,
        vec![
            double(ShapeFact::from(vec![Some(1), Some(201)])),
            double(ShapeFact::from(vec![Some(1), Some(101)])),
        ],
        vec![],
    );
    assert_eq!(output.shape.known_dims(), Some(vec![Some(101), Some(201)]));
}

#[test]
fn createarray_uses_public_fillvalue_form_not_retired_kernel_harness_signature() {
    let numeric = double(ShapeFact::Scalar);
    let output = infer(
        &CREATEARRAY_CATALOG_ENTRY,
        vec![numeric.clone()],
        vec![LiteralValue::Number(1.0)],
    );
    assert!(output.shape.is_proven_equivalent(&ShapeFact::Scalar));
    assert_eq!(output.numeric().unwrap().class, NumericClass::Double);
    let output = infer(
        &CREATEARRAY_CATALOG_ENTRY,
        vec![
            numeric.clone(),
            numeric,
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::Logical),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::String("logical".into()),
            LiteralValue::String("FillValue".into()),
            LiteralValue::Unknown,
        ],
    );
    assert_eq!(output.kind, ValueKindFact::Logical);
    assert_eq!(output.shape.known_dims(), Some(vec![None, None]));
}
