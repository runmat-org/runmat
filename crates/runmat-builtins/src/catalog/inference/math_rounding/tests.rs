use crate::{builtin_catalog_entry_by_name, infer_catalog_call};
use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::new(literals),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn rounding_singletons_match_numeric_and_character_carriers() {
    for name in ["ceil", "floor", "fix", "round"] {
        for (kind, expected) in [
            (
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Single,
                    domain: NumericDomain::Real,
                }),
                StorageFact::Unknown,
            ),
            (
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                StorageFact::Unknown,
            ),
            (
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Complex,
                }),
                StorageFact::Dense,
            ),
            (ValueKindFact::Character, StorageFact::Dense),
        ] {
            let input = ValueFact::proven(kind, ShapeFact::Scalar, StorageFact::Dense);
            let result = infer_catalog_call(
                builtin_catalog_entry_by_name(name).unwrap(),
                &request(vec![input], vec![]),
            );
            assert!(result.diagnostics.is_empty(), "{name}");
            assert_eq!(result.outputs[0].storage, expected, "{name}");
        }
    }
}

#[test]
fn automatic_rounding_does_not_promise_host_placement() {
    for name in ["ceil", "fix", "floor", "round"] {
        for storage in [
            StorageFact::Dense,
            StorageFact::Unknown,
            StorageFact::Scalar,
        ] {
            let input = ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::Scalar,
                storage,
            );
            let result = infer_catalog_call(
                builtin_catalog_entry_by_name(name).unwrap(),
                &request(vec![input.clone()], vec![]),
            );
            assert_eq!(
                result.outputs[0].residency,
                if storage == StorageFact::Scalar {
                    runmat_types::ResidencyFact::Host
                } else {
                    runmat_types::ResidencyFact::Unknown
                }
            );
            if name == "round" {
                let result = infer_catalog_call(
                    builtin_catalog_entry_by_name(name).unwrap(),
                    &request(
                        vec![
                            input.clone(),
                            ValueFact::proven(
                                input.kind.clone(),
                                ShapeFact::Scalar,
                                StorageFact::Scalar,
                            ),
                        ],
                        vec![LiteralValue::Number(23.75), LiteralValue::Number(20.0)],
                    ),
                );
                assert_eq!(
                    result.outputs[0].residency,
                    if storage == StorageFact::Scalar {
                        runmat_types::ResidencyFact::Host
                    } else {
                        runmat_types::ResidencyFact::Unknown
                    }
                );
            }
        }
    }
}

#[test]
fn round_unknown_digits_and_padded_mode_do_not_invent_errors() {
    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
    let result = infer_catalog_call(
        builtin_catalog_entry_by_name("round").unwrap(),
        &request(vec![input.clone(), unknown], vec![]),
    );
    assert!(result.diagnostics.is_empty());
    let text = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    let result = infer_catalog_call(
        builtin_catalog_entry_by_name("round").unwrap(),
        &request(
            vec![input.clone(), input, text],
            vec![
                LiteralValue::Number(1.25),
                LiteralValue::Number(2.0),
                LiteralValue::String(" significant ".into()),
            ],
        ),
    );
    assert!(result.diagnostics.is_empty());
}
