use super::infer;
use crate::{
    builtin_catalog_entry_by_name, LogicalBinaryOperator, LogicalElementwiseRule,
    LogicalUnaryOperator,
};
use runmat_types::{
    CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, ObjectFact,
    OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

fn tabular(class: runmat_types::ClassIdentity) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class),
            properties: Default::default(),
            properties_complete: false,
            handle_semantics: None,
        }),
        ShapeFact::from(vec![Some(2), Some(2)]),
        StorageFact::Opaque,
    )
}

fn numeric(shape: Vec<Option<usize>>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape),
        StorageFact::Dense,
    )
}

#[test]
fn binary_operators_broadcast_to_logical() {
    for (name, operator) in [
        ("and", LogicalBinaryOperator::And),
        ("or", LogicalBinaryOperator::Or),
        ("xor", LogicalBinaryOperator::Xor),
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("logical operator entry");
        let inferred = infer(
            LogicalElementwiseRule::Binary(operator),
            &request(vec![
                numeric(vec![Some(3), Some(1)]),
                numeric(vec![Some(1), Some(4)]),
            ]),
            entry,
        );
        assert!(inferred.diagnostics.is_empty(), "{name}");
        assert_eq!(inferred.outputs[0].kind, ValueKindFact::Logical);
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(3), Some(4)])
        );
    }
}

#[test]
fn unary_operator_preserves_shape() {
    let entry = builtin_catalog_entry_by_name("not").expect("not entry");
    let inferred = infer(
        LogicalElementwiseRule::Unary(LogicalUnaryOperator::Not),
        &request(vec![numeric(vec![Some(2), Some(5)])]),
        entry,
    );
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(inferred.outputs[0].kind, ValueKindFact::Logical);
    assert_eq!(
        inferred.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(5)])
    );
}

#[test]
fn tabular_operators_preserve_identity_and_reject_mixed_containers() {
    let entry = builtin_catalog_entry_by_name("and").expect("and entry");
    let table = tabular(runmat_types::standard::TABLE.owned());
    let inferred = infer(
        LogicalElementwiseRule::Binary(LogicalBinaryOperator::And),
        &request(vec![table.clone(), numeric(vec![Some(1), Some(1)])]),
        entry,
    );
    assert!(inferred.diagnostics.is_empty());
    assert_eq!(inferred.outputs[0].kind, table.kind);

    let mixed = infer(
        LogicalElementwiseRule::Binary(LogicalBinaryOperator::And),
        &request(vec![
            table,
            tabular(runmat_types::standard::TIMETABLE.owned()),
        ]),
        entry,
    );
    assert!(mixed
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOGICAL-TABULAR-CLASS"));
}
