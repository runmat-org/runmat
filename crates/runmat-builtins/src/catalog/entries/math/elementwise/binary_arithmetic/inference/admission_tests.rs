use super::test_support::{numeric, request};
use crate::{
    infer_catalog_call, LDIVIDE_CATALOG_ENTRY, MINUS_CATALOG_ENTRY, PLUS_CATALOG_ENTRY,
    POWER_CATALOG_ENTRY, RDIVIDE_CATALOG_ENTRY, TIMES_CATALOG_ENTRY,
};
use runmat_types::{
    CertaintyFact, DynamicReason, LiteralValue, NumericClass, NumericDomain, NumericFact,
    ValueFact, ValueKindFact,
};

#[test]
fn arithmetic_certainty_preserves_semantic_proof_without_claiming_placement() {
    let operand = numeric(
        NumericClass::Double,
        NumericDomain::Real,
        &[Some(2), Some(3)],
    );
    for entry in [
        &PLUS_CATALOG_ENTRY,
        &MINUS_CATALOG_ENTRY,
        &TIMES_CATALOG_ENTRY,
        &RDIVIDE_CATALOG_ENTRY,
        &LDIVIDE_CATALOG_ENTRY,
    ] {
        let result = infer_catalog_call(entry, &request(vec![operand.clone(), operand.clone()]));
        assert!(result.diagnostics.is_empty(), "{}", entry.identity.name);
        assert_eq!(result.outputs[0].certainty, CertaintyFact::Proven);
        assert_eq!(
            result.outputs[0].residency,
            runmat_types::ResidencyFact::Unknown
        );
        let mut dynamic = operand.clone();
        dynamic.certainty = CertaintyFact::Dynamic(DynamicReason::ExternalData);
        for arguments in [
            vec![dynamic.clone(), operand.clone()],
            vec![operand.clone(), dynamic],
        ] {
            let result = infer_catalog_call(entry, &request(arguments));
            assert_eq!(
                result.outputs[0].certainty,
                CertaintyFact::Dynamic(DynamicReason::ExternalData)
            );
        }
        let mismatched = numeric(
            NumericClass::Double,
            NumericDomain::Real,
            &[Some(4), Some(3)],
        );
        let result = infer_catalog_call(entry, &request(vec![operand.clone(), mismatched]));
        assert!(!result.diagnostics.is_empty());
        assert_ne!(result.outputs[0].certainty, CertaintyFact::Proven);
    }
    // Real base/exponent classes alone do not prove a real power result.
    let result = infer_catalog_call(
        &POWER_CATALOG_ENTRY,
        &request(vec![operand.clone(), operand]),
    );
    assert_ne!(result.outputs[0].certainty, CertaintyFact::Proven);
}

#[test]
fn integer_inference_rejects_non_numeric_and_complex_pairs() {
    let integer = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    for entry in [
        &PLUS_CATALOG_ENTRY,
        &MINUS_CATALOG_ENTRY,
        &TIMES_CATALOG_ENTRY,
        &RDIVIDE_CATALOG_ENTRY,
        &LDIVIDE_CATALOG_ENTRY,
        &POWER_CATALOG_ENTRY,
    ] {
        let name = entry.identity.name;
        for other in [
            ValueFact::scalar(ValueKindFact::Logical),
            ValueFact::scalar(ValueKindFact::Character),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Complex,
            })),
        ] {
            for pair in [
                vec![integer.clone(), other.clone()],
                vec![other, integer.clone()],
            ] {
                assert!(
                    infer_catalog_call(entry, &request(pair))
                        .diagnostics
                        .iter()
                        .any(|d| d.code == "RM-CATALOG-BINARY-ARITHMETIC-INPUT"),
                    "{name}"
                );
            }
        }
        let result = infer_catalog_call(entry, &request(vec![integer.clone(), integer.clone()]));
        assert!(result.diagnostics.is_empty(), "{name}");
        assert_eq!(result.outputs[0].kind, integer.kind, "{name}");
        let unknown_shape = numeric(NumericClass::Double, NumericDomain::Real, &[None, None]);
        let result = infer_catalog_call(entry, &request(vec![integer.clone(), unknown_shape]));
        assert!(
            result.diagnostics.is_empty(),
            "{name}: unknown shape may be scalar"
        );
    }
}

#[test]
fn dynamic_like_keyword_is_not_a_proven_argument_error() {
    for entry in [
        &PLUS_CATALOG_ENTRY,
        &MINUS_CATALOG_ENTRY,
        &TIMES_CATALOG_ENTRY,
        &RDIVIDE_CATALOG_ENTRY,
        &LDIVIDE_CATALOG_ENTRY,
        &POWER_CATALOG_ENTRY,
    ] {
        let name = entry.identity.name;
        let operand = numeric(
            NumericClass::Double,
            NumericDomain::Real,
            &[Some(2), Some(3)],
        );
        for option in [
            ValueFact::scalar(ValueKindFact::String),
            ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
        ] {
            let call = request(vec![
                operand.clone(),
                operand.clone(),
                option,
                operand.clone(),
            ]);
            let inferred = infer_catalog_call(entry, &call);
            assert!(
                inferred.diagnostics.is_empty(),
                "{name}: {:?}",
                inferred.diagnostics
            );
        }
        let mut call = request(vec![
            operand.clone(),
            operand.clone(),
            ValueFact::scalar(ValueKindFact::String),
            operand.clone(),
        ]);
        call.literals.literal_args = vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Keyword("not-like".into()),
            LiteralValue::Unknown,
        ];
        assert!(
            infer_catalog_call(entry, &call)
                .diagnostics
                .iter()
                .any(|d| d.code == "RM-CATALOG-BINARY-ARITHMETIC-OPTION"),
            "{name}"
        );
        call.arguments[2] = operand;
        call.literals = Default::default();
        assert!(
            infer_catalog_call(entry, &call)
                .diagnostics
                .iter()
                .any(|d| d.code == "RM-CATALOG-BINARY-ARITHMETIC-OPTION"),
            "{name}"
        );
    }
}
