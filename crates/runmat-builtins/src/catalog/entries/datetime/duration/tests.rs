use super::*;
use runmat_types::{
    CallRequest, ClassId, DimensionFact, DynamicReason, LiteralContext, LiteralValue, NumericClass,
    NumericDomain, NumericFact, ObjectFact, OutputSelection, RequestedOutputCount, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn call(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(vec![]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}
fn number(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Dense,
    )
}
fn logical(shape: ShapeFact) -> ValueFact {
    ValueFact::proven(ValueKindFact::Logical, shape, StorageFact::Dense)
}
fn shape(rows: usize, cols: usize) -> ShapeFact {
    ShapeFact::Shaped {
        dims: vec![DimensionFact::Known(rows), DimensionFact::Known(cols)],
    }
}

#[test]
fn duration_inventory_is_exact_and_all_examples_have_assertions() {
    assert_eq!(ENTRIES.len(), 18);
    for entry in ENTRIES {
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert!(
            !entry.documentation.examples.is_empty(),
            "{}",
            entry.identity.name
        );
        assert!(
            entry.documentation.examples.iter().all(|example| matches!(
                example.verification,
                BuiltinExampleVerification::Assertions { .. }
            )),
            "{}",
            entry.identity.name
        );
        assert_eq!(entry.bindings.len(), 1, "{}", entry.identity.name);
    }
}

#[test]
fn constructor_and_units_prove_shapes_without_guessing_unknown_class() {
    let built =
        crate::infer_catalog_call(&DURATION_CATALOG_ENTRY, &call(vec![number(shape(4, 3))], 1));
    assert!(built.diagnostics.is_empty());
    assert_eq!(built.outputs[0].shape, shape(4, 1));
    assert_eq!(facts::duration_class(&built.outputs[0]), Some(true));

    let unit = crate::infer_catalog_call(
        &HOURS_CATALOG_ENTRY,
        &call(vec![built.outputs[0].clone()], 1),
    );
    assert_eq!(unit.outputs[0].shape, shape(4, 1));
    assert!(matches!(
        unit.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            ..
        })
    ));

    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let dynamic = crate::infer_catalog_call(&DAYS_CATALOG_ENTRY, &call(vec![unknown], 1));
    assert!(matches!(dynamic.outputs[0].kind, ValueKindFact::Unknown));
    assert!(!dynamic.dynamic_outputs);
}

#[test]
fn constructor_format_pair_does_not_corrupt_component_inference() {
    let mut request = call(
        vec![
            number(shape(1, 2)),
            number(ShapeFact::Scalar),
            number(ShapeFact::Scalar),
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::scalar(ValueKindFact::String),
        ],
        1,
    );
    request.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::Unknown,
        LiteralValue::String("Format".into()),
        LiteralValue::String("hh:mm".into()),
    ]);
    let inferred = crate::infer_catalog_call(&DURATION_CATALOG_ENTRY, &request);
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(inferred.outputs[0].shape, shape(1, 2));
    let two_outputs =
        crate::infer_catalog_call(&DURATION_CATALOG_ENTRY, &call(vec![number(shape(2, 3))], 2));
    assert!(!two_outputs.diagnostics.is_empty());
}

#[test]
fn logical_constructor_inputs_match_runtime_admission() {
    for inputs in [
        vec![logical(shape(2, 3))],
        vec![
            logical(shape(1, 2)),
            number(ShapeFact::Scalar),
            number(ShapeFact::Scalar),
        ],
        vec![
            logical(ShapeFact::Scalar),
            number(ShapeFact::Scalar),
            number(ShapeFact::Scalar),
        ],
    ] {
        let inferred = crate::infer_catalog_call(&DURATION_CATALOG_ENTRY, &call(inputs, 1));
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
        assert_eq!(facts::duration_class(&inferred.outputs[0]), Some(true));
    }
}

#[test]
fn constructor_known_shape_conflict_is_diagnosed_without_guessing_unknown_extents() {
    let conflict = crate::infer_catalog_call(
        &DURATION_CATALOG_ENTRY,
        &call(
            vec![
                number(shape(2, 1)),
                number(shape(3, 1)),
                number(ShapeFact::Scalar),
            ],
            1,
        ),
    );
    assert!(conflict
        .diagnostics
        .iter()
        .any(|diagnostic| { diagnostic.code == "RM-DURATION-COMPONENT-SHAPE" }));

    let uncertain = ShapeFact::Shaped {
        dims: vec![DimensionFact::Unknown, DimensionFact::Known(1)],
    };
    let dynamic = crate::infer_catalog_call(
        &DURATION_CATALOG_ENTRY,
        &call(
            vec![
                number(shape(2, 1)),
                number(uncertain),
                number(ShapeFact::Scalar),
            ],
            1,
        ),
    );
    assert!(!dynamic
        .diagnostics
        .iter()
        .any(|diagnostic| { diagnostic.code == "RM-DURATION-COMPONENT-SHAPE" }));
    assert_eq!(dynamic.outputs[0].shape, ShapeFact::Unknown);
}

#[test]
fn known_other_object_class_is_not_inferred_as_duration() {
    let other = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: Some(ClassId(7)),
            runtime_class: None,
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    assert_eq!(facts::duration_class(&other), Some(false));
    assert!(
        !crate::infer_catalog_call(&SECONDS_CATALOG_ENTRY, &call(vec![other], 1))
            .diagnostics
            .is_empty()
    );
}

#[test]
fn comparison_result_is_logical_and_broadcasted() {
    let left = facts::duration(shape(2, 1));
    let right = facts::duration(shape(1, 3));
    let inferred =
        crate::infer_catalog_call(&DURATION_EQ_CATALOG_ENTRY, &call(vec![left, right], 1));
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(inferred.outputs[0].shape, shape(2, 3));
    assert!(matches!(inferred.outputs[0].kind, ValueKindFact::Logical));
}

#[test]
fn comparison_method_rejects_two_known_numeric_operands() {
    let inferred = crate::infer_catalog_call(
        &DURATION_EQ_CATALOG_ENTRY,
        &call(vec![number(shape(1, 1)), number(shape(1, 1))], 1),
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-DURATION-OP-RECEIVER"));
}
