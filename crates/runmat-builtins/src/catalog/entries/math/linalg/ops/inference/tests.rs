use super::*;

fn matrix(rows: usize, columns: usize, class: NumericClass) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(rows), Some(columns)]),
        StorageFact::Dense,
    )
}

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext {
            literal_args: literals,
        },
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn page_product_projects_scalar_and_literal_transpose_without_inventing_dynamic_shape() {
    let x = matrix(2, 3, NumericClass::Single);
    let y = matrix(2, 4, NumericClass::Single);
    let four = request(
        vec![
            x.clone(),
            ValueFact::scalar(ValueKindFact::Character),
            y.clone(),
            ValueFact::scalar(ValueKindFact::Character),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Character("transpose".into()),
            LiteralValue::Unknown,
            LiteralValue::Character("none".into()),
        ],
    );
    let result = infer(
        &four,
        &super::super::pagemtimes::PAGEMTIMES_CATALOG_ENTRY,
        Operation::Pagemtimes,
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(3), Some(4)])
    );
    assert!(
        matches!(&result.outputs[0].kind, ValueKindFact::Numeric(n) if n.class == NumericClass::Single)
    );

    let mut dynamic = four;
    dynamic.literals.literal_args[1] = LiteralValue::Unknown;
    assert_eq!(
        infer(
            &dynamic,
            &super::super::pagemtimes::PAGEMTIMES_CATALOG_ENTRY,
            Operation::Pagemtimes
        )
        .outputs[0]
            .shape,
        ShapeFact::Unknown
    );

    let scalar = request(vec![x, matrix(1, 1, NumericClass::Double)], vec![]);
    let result = infer(
        &scalar,
        &super::super::pagemtimes::PAGEMTIMES_CATALOG_ENTRY,
        Operation::Pagemtimes,
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert!(
        matches!(&result.outputs[0].kind, ValueKindFact::Numeric(n) if n.class == NumericClass::Double)
    );
}

#[test]
fn reduction_and_structure_shapes_follow_runtime_float_boundaries() {
    let operands = request(
        vec![
            matrix(2, 3, NumericClass::Single),
            matrix(2, 3, NumericClass::Single),
        ],
        vec![],
    );
    let dot = infer(
        &operands,
        &super::super::dot::DOT_CATALOG_ENTRY,
        Operation::Dot,
    );
    assert_eq!(
        dot.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(3)])
    );
    assert!(
        matches!(&dot.outputs[0].kind, ValueKindFact::Numeric(n) if n.class == NumericClass::Single)
    );

    let cross = infer(
        &operands,
        &super::super::cross::CROSS_CATALOG_ENTRY,
        Operation::Cross,
    );
    assert_eq!(
        cross.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
}

#[test]
fn six_operation_entries_have_distinct_catalog_contracts_and_original_example_counts() {
    for (entry, examples) in [
        (&super::super::cross::CROSS_CATALOG_ENTRY, 4),
        (&super::super::dot::DOT_CATALOG_ENTRY, 5),
        (&super::super::pagemtimes::PAGEMTIMES_CATALOG_ENTRY, 5),
        (&super::super::pagetranspose::PAGETRANSPOSE_CATALOG_ENTRY, 4),
        (&super::super::trace::TRACE_CATALOG_ENTRY, 6),
        (&super::super::transpose::TRANSPOSE_CATALOG_ENTRY, 6),
    ] {
        assert_eq!(
            entry.documentation.authority,
            crate::BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(
            entry.documentation.examples.len(),
            examples,
            "{}",
            entry.identity.name
        );
        assert!(
            !entry.descriptor.signatures.is_empty(),
            "{}",
            entry.identity.name
        );
    }
}
