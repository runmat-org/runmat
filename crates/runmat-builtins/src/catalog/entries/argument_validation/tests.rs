use crate::*;
use runmat_types::*;

#[test]
fn text_shape_inference_retains_unknown_trailing_dimensions() {
    for (trailing, rejected) in [(Some(1), false), (Some(3), true), (None, false)] {
        let value = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), Some(6), trailing]),
            StorageFact::Dense,
        );
        for entry in [
            &MUST_BE_TEXT_CATALOG_ENTRY,
            &MUST_BE_TEXT_SCALAR_CATALOG_ENTRY,
            &MUST_BE_VALID_VARIABLE_NAME_CATALOG_ENTRY,
        ] {
            let result = infer_catalog_call(entry, &request(vec![value.clone()], 0));
            assert_eq!(
                !result.diagnostics.is_empty(),
                rejected,
                "{} {trailing:?}",
                entry.identity.name
            );
        }
    }
}

fn request(arguments: Vec<ValueFact>, count: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    }
}

#[test]
fn unknown_inputs_preserve_zero_output_effects_and_exact_arity() {
    for entry in [
        &MUST_BE_A_CATALOG_ENTRY,
        &MUST_BE_COLUMN_CATALOG_ENTRY,
        &MUST_BE_FLOAT_CATALOG_ENTRY,
        &MUST_BE_NONEMPTY_CATALOG_ENTRY,
        &MUST_BE_NONSPARSE_CATALOG_ENTRY,
        &MUST_BE_NUMERIC_CATALOG_ENTRY,
        &MUST_BE_NUMERIC_OR_LOGICAL_CATALOG_ENTRY,
        &MUST_BE_SCALAR_OR_EMPTY_CATALOG_ENTRY,
        &MUST_BE_SPARSE_CATALOG_ENTRY,
        &MUST_BE_TEXT_CATALOG_ENTRY,
        &MUST_BE_TEXT_SCALAR_CATALOG_ENTRY,
        &MUST_BE_NONZERO_LENGTH_TEXT_CATALOG_ENTRY,
        &MUST_BE_UNDERLYING_TYPE_CATALOG_ENTRY,
        &MUST_BE_VALID_VARIABLE_NAME_CATALOG_ENTRY,
        &MUST_BE_VECTOR_CATALOG_ENTRY,
        &VALIDATE_FUNCTION_SIGNATURES_JSON_CATALOG_ENTRY,
    ] {
        let n = entry.descriptor.signatures[0].inputs.len();
        let args = vec![ValueFact::unknown(DynamicReason::RuntimeValue); n];
        let result = infer_catalog_call(entry, &request(args.clone(), 0));
        assert_eq!(entry.placement.residency, BuiltinResidencyPolicy::Host);
        assert!(result.outputs.is_empty(), "{}", entry.identity.name);
        assert!(
            result.diagnostics.is_empty(),
            "{} {:?}",
            entry.identity.name,
            result.diagnostics
        );
        assert!(result.effects.0.contains(&EffectKind::MayThrow));
        assert!(!infer_catalog_call(entry, &request(args, 1))
            .diagnostics
            .is_empty());
        assert!(!infer_catalog_call(entry, &request(vec![], 0))
            .diagnostics
            .is_empty());
    }
}

#[test]
fn scalar_empty_and_vector_facts_do_not_ignore_unknown_or_trailing_dimensions() {
    let numeric = |shape| {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }),
            shape,
            StorageFact::Dense,
        )
    };
    for (shape, rejected) in [
        (ShapeFact::from(vec![Some(1), Some(1), Some(2)]), true),
        (ShapeFact::from(vec![Some(2), Some(3), Some(0)]), false),
        (ShapeFact::from(vec![Some(1), Some(1), None]), false),
    ] {
        let result = infer_catalog_call(
            &MUST_BE_SCALAR_OR_EMPTY_CATALOG_ENTRY,
            &request(vec![numeric(shape)], 0),
        );
        assert_eq!(!result.diagnostics.is_empty(), rejected);
    }
    let mut unknown_flag = request(
        vec![
            numeric(ShapeFact::from(vec![Some(0), Some(2), Some(3)])),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ],
        0,
    );
    assert!(
        infer_catalog_call(&MUST_BE_VECTOR_CATALOG_ENTRY, &unknown_flag)
            .diagnostics
            .is_empty()
    );
    unknown_flag.literals = LiteralContext::new(vec![
        LiteralValue::Unknown,
        LiteralValue::String("bad".into()),
    ]);
    assert!(
        !infer_catalog_call(&MUST_BE_VECTOR_CATALOG_ENTRY, &unknown_flag)
            .diagnostics
            .is_empty()
    );
}

#[test]
fn namedargs_cell_extent_uses_complete_field_facts_not_sorted_field_order() {
    let fields = std::collections::BTreeMap::from([
        ("z".into(), ValueFact::scalar(ValueKindFact::Logical)),
        ("a".into(), ValueFact::scalar(ValueKindFact::String)),
    ]);
    let value = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)));
    let result = infer_catalog_call(&NAMEDARGS2CELL_CATALOG_ENTRY, &request(vec![value], 1));
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(4)])
    );
    let ValueKindFact::Cell(cell) = &result.outputs[0].kind else {
        panic!("cell output")
    };
    assert!(!cell.elements_complete);
    assert!(cell.elements.is_empty());
}

#[test]
fn integer_and_logical_class_facts_remain_distinct() {
    let logical = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::Scalar,
        StorageFact::Sparse,
    );
    assert!(!infer_catalog_call(
        &MUST_BE_NUMERIC_CATALOG_ENTRY,
        &request(vec![logical.clone()], 0)
    )
    .diagnostics
    .is_empty());
    assert!(infer_catalog_call(
        &MUST_BE_NUMERIC_OR_LOGICAL_CATALOG_ENTRY,
        &request(vec![logical], 0)
    )
    .diagnostics
    .is_empty());
    let integer = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    assert!(
        !infer_catalog_call(&MUST_BE_FLOAT_CATALOG_ENTRY, &request(vec![integer], 0))
            .diagnostics
            .is_empty()
    );
}
