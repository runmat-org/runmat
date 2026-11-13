use super::*;
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        },
        &JSONENCODE_CATALOG_ENTRY,
    )
}

#[test]
fn supported_option_spellings_and_host_character_output() {
    let value = ValueFact::unknown(DynamicReason::RuntimeValue);
    let name = ValueFact::scalar(ValueKindFact::String);
    let logical = ValueFact::scalar(ValueKindFact::Logical);
    let result = request(
        vec![value, name, logical],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("pReTtYpRiNt".into()),
        ],
    );
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.outputs[0].kind, ValueKindFact::Character);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
}

#[test]
fn malformed_pairs_and_every_known_invalid_name_are_diagnosed() {
    let value = ValueFact::unknown(DynamicReason::RuntimeValue);
    let name = ValueFact::scalar(ValueKindFact::String);
    let flag = ValueFact::scalar(ValueKindFact::Logical);
    let odd = request(
        vec![value.clone(), name.clone(), flag.clone(), name.clone()],
        vec![],
    );
    assert!(!odd.diagnostics.is_empty());
    let unknown = request(
        vec![value, name.clone(), flag.clone(), name, flag],
        vec![
            LiteralValue::Unknown,
            LiteralValue::String("PrettyPrint".into()),
            LiteralValue::Unknown,
            LiteralValue::String("Wrong".into()),
        ],
    );
    assert!(!unknown.diagnostics.is_empty());
}

#[test]
fn invalid_option_value_rejects_but_numeric_scalar_remains_admitted() {
    let value = ValueFact::unknown(DynamicReason::RuntimeValue);
    let name = ValueFact::scalar(ValueKindFact::String);
    let bad = request(
        vec![
            value.clone(),
            name.clone(),
            ValueFact::proven(
                ValueKindFact::Logical,
                ShapeFact::from(vec![Some(2), Some(2)]),
                StorageFact::Dense,
            ),
        ],
        vec![],
    );
    assert!(!bad.diagnostics.is_empty());
    let numeric = request(
        vec![
            value,
            name,
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            })),
        ],
        vec![],
    );
    assert!(numeric.diagnostics.is_empty());
}
