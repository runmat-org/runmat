use super::*;
use runmat_types::*;

fn request(pp: ValueFact, query: ValueFact) -> CallRequest {
    CallRequest {
        arguments: vec![pp, query],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn unknown_polynomial_components_do_not_falsely_prove_query_shape() {
    let pp = ValueFact::unknown(DynamicReason::RuntimeValue);
    let query = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(2)]),
        StorageFact::Dense,
    );
    let inferred = inference::infer(&request(pp, query), &PPVAL_CATALOG_ENTRY);
    assert_eq!(inferred.outputs[0].shape, ShapeFact::Unknown);
    assert_eq!(
        inferred.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
}

#[test]
fn invalid_known_first_argument_is_diagnosed() {
    let inferred = inference::infer(
        &request(
            ValueFact::scalar(ValueKindFact::Logical),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        ),
        &PPVAL_CATALOG_ENTRY,
    );
    assert!(inferred
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-PPVAL-PP"));
}
