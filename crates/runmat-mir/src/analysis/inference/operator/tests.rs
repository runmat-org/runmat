use super::*;
use runmat_types::{
    standard, ClassIdentity, NumericClass, NumericDomain, NumericFact, ObjectFact, ShapeFact,
    StorageFact,
};
use std::collections::BTreeMap;

fn number() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

#[test]
fn primitive_operators_use_their_canonical_call_contract() {
    let text = ValueFact::scalar(ValueKindFact::String);
    let chars = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(2), Some(4)]),
        StorageFact::Dense,
    );
    for (left, right) in [(text.clone(), chars.clone()), (chars, text)] {
        let inferred = infer_binary_operator(
            OperatorKind::Add,
            left,
            right,
            [LiteralValue::Unknown, LiteralValue::Unknown],
        );
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
        assert_eq!(inferred.fact.kind, ValueKindFact::String);
        assert_eq!(inferred.fact.shape, ShapeFact::from(vec![Some(2), Some(1)]));
        assert_eq!(inferred.fact.residency, runmat_types::ResidencyFact::Host);
    }
    let inferred = infer_binary_operator(
        OperatorKind::Add,
        number(),
        number(),
        [LiteralValue::Unknown, LiteralValue::Unknown],
    );
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(inferred.fact.kind, number().kind);
}

fn object(class: ClassIdentity, properties: BTreeMap<String, ValueFact>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class),
            properties,
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn tf() -> ValueFact {
    let mut properties = BTreeMap::new();
    properties.insert("Numerator".into(), number());
    properties.insert("Denominator".into(), number());
    object(standard::TRANSFER_FUNCTION.owned(), properties)
}

fn output_class(inferred: &FactInference) -> Option<&ClassIdentity> {
    match &inferred.fact.kind {
        ValueKindFact::Object(object) => object.runtime_class.as_ref(),
        _ => None,
    }
}

#[test]
fn nested_tf_arithmetic_uses_typed_rhs_and_lhs_method_contracts() {
    let scaled = infer_binary_operator(
        OperatorKind::MatrixMultiply,
        number(),
        tf(),
        [LiteralValue::Number(2.0), LiteralValue::Unknown],
    );
    assert!(scaled.diagnostics.is_empty(), "{:?}", scaled.diagnostics);
    assert_eq!(
        output_class(&scaled),
        Some(&standard::TRANSFER_FUNCTION.owned())
    );

    let shifted = infer_binary_operator(
        OperatorKind::Add,
        scaled.fact,
        number(),
        [LiteralValue::Unknown, LiteralValue::Number(3.0)],
    );
    assert!(shifted.diagnostics.is_empty(), "{:?}", shifted.diagnostics);
    assert_eq!(
        output_class(&shifted),
        Some(&standard::TRANSFER_FUNCTION.owned())
    );

    let divided = infer_binary_operator(
        OperatorKind::Mrdivide,
        number(),
        shifted.fact,
        [LiteralValue::Number(1.0), LiteralValue::Unknown],
    );
    assert!(divided.diagnostics.is_empty(), "{:?}", divided.diagnostics);
    assert_eq!(
        output_class(&divided),
        Some(&standard::TRANSFER_FUNCTION.owned())
    );
}

#[test]
fn tf_invalid_inputs_still_use_the_method_contract() {
    let inferred = infer_binary_operator(
        OperatorKind::MatrixPower,
        tf(),
        number(),
        [LiteralValue::Unknown, LiteralValue::Number(0.5)],
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-TF-METHOD-EXPONENT"));
    assert!(!inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-BINARY-OPERAND"));

    let structure = ValueFact::scalar(ValueKindFact::Struct(runmat_types::StructFact::scalar(
        BTreeMap::new(),
        true,
    )));
    let invalid_operand = infer_binary_operator(
        OperatorKind::Add,
        tf(),
        structure,
        [LiteralValue::Unknown, LiteralValue::Unknown],
    );
    assert!(invalid_operand
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-TF-METHOD-MODEL"));
    assert!(!invalid_operand
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-BINARY-OPERAND"));
}

#[test]
fn adjacent_cataloged_object_families_share_operator_resolution() {
    let duration = object(standard::DURATION.owned(), BTreeMap::new());
    let duration_sum = infer_binary_operator(
        OperatorKind::Add,
        duration.clone(),
        duration,
        [LiteralValue::Unknown, LiteralValue::Unknown],
    );
    assert!(duration_sum.diagnostics.is_empty());
    assert_eq!(
        output_class(&duration_sum),
        Some(&standard::DURATION.owned())
    );

    let mut dl_properties = BTreeMap::new();
    dl_properties.insert("Data".into(), number());
    let dlarray = object(standard::DL_ARRAY.owned(), dl_properties);
    let dl_sum = infer_binary_operator(
        OperatorKind::Add,
        dlarray,
        number(),
        [LiteralValue::Unknown, LiteralValue::Number(1.0)],
    );
    assert!(dl_sum.diagnostics.is_empty(), "{:?}", dl_sum.diagnostics);
    assert_eq!(output_class(&dl_sum), Some(&standard::DL_ARRAY.owned()));

    let decomposition = object(ClassIdentity::from("decomposition"), BTreeMap::new());
    let scaled = infer_binary_operator(
        OperatorKind::MatrixMultiply,
        decomposition,
        number(),
        [LiteralValue::Unknown, LiteralValue::Number(2.0)],
    );
    assert!(scaled.diagnostics.is_empty(), "{:?}", scaled.diagnostics);
    assert_eq!(
        output_class(&scaled),
        Some(&ClassIdentity::from("decomposition"))
    );

    let negated = infer_unary_operator(
        OperatorKind::UnaryMinus,
        object(ClassIdentity::from("decomposition"), BTreeMap::new()),
        LiteralValue::Unknown,
    );
    assert!(negated.diagnostics.is_empty(), "{:?}", negated.diagnostics);
    assert_eq!(
        output_class(&negated),
        Some(&ClassIdentity::from("decomposition"))
    );
}

#[test]
fn uncataloged_object_operator_stays_dynamic_without_weakening_numeric_rules() {
    let unknown_object = object(ClassIdentity::from("example.Widget"), BTreeMap::new());
    let object_result = infer_binary_operator(
        OperatorKind::Add,
        unknown_object,
        number(),
        [LiteralValue::Unknown, LiteralValue::Unknown],
    );
    assert!(object_result.diagnostics.is_empty());
    assert!(matches!(
        object_result.fact.certainty,
        runmat_types::CertaintyFact::Dynamic(DynamicReason::DynamicDispatch)
    ));

    let structure = ValueFact::scalar(ValueKindFact::Struct(runmat_types::StructFact::scalar(
        BTreeMap::new(),
        true,
    )));
    let primitive = infer_binary_operator(
        OperatorKind::Add,
        structure.clone(),
        number(),
        [LiteralValue::Unknown, LiteralValue::Unknown],
    );
    let canonical = runmat_builtins::infer_catalog_call(
        &runmat_builtins::PLUS_CATALOG_ENTRY,
        &CallRequest {
            arguments: vec![structure, number()],
            literals: LiteralContext::new(vec![LiteralValue::Unknown, LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
    );
    assert_eq!(primitive.diagnostics, canonical.diagnostics);
    assert_eq!(canonical.outputs.len(), 1);
    assert_eq!(primitive.fact, canonical.outputs[0]);
    assert!(primitive.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "RM-CATALOG-BINARY-ARITHMETIC-INPUT"
            && diagnostic.severity == runmat_types::InferenceSeverity::Error
            && diagnostic.argument == Some(0)
    }));
}
