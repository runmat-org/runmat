use super::*;

#[test]
fn unknown_scalar_class_does_not_invent_scalar_storage() {
    let mut req = request(vec![Some(1), Some(1)], LiteralValue::Number(1.));
    req.arguments[0].kind = ValueKindFact::Unknown;
    let result = infer(&req, &super::super::MAXK_CATALOG_ENTRY);
    assert_eq!(result.outputs[0].storage, StorageFact::Unknown);
    assert_eq!(result.outputs[1].storage, StorageFact::Scalar);
}
fn input(shape: Vec<Option<usize>>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(shape),
        StorageFact::Dense,
    )
}
fn request(shape: Vec<Option<usize>>, k: LiteralValue) -> CallRequest {
    CallRequest {
        arguments: vec![
            input(shape),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            })),
        ],
        literals: LiteralContext::new(vec![LiteralValue::Unknown, k]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    }
}
#[test]
fn exact_k_clamps_and_indices_are_double_for_both_identities() {
    for entry in [
        &super::super::MAXK_CATALOG_ENTRY,
        &super::super::MINK_CATALOG_ENTRY,
    ] {
        let result = infer(
            &request(
                vec![Some(3), Some(2)],
                LiteralValue::Integer {
                    text: "18446744073709551615".into(),
                    class: NumericClass::UInt64,
                },
            ),
            entry,
        );
        if usize::BITS == 64 {
            assert_eq!(
                result.outputs[0].shape,
                ShapeFact::from(vec![Some(3), Some(2)])
            );
            assert_eq!(
                result.outputs[0].numeric().unwrap().class,
                NumericClass::UInt64
            );
            assert_eq!(
                result.outputs[1].numeric().unwrap().class,
                NumericClass::Double
            );
        }
    }
}
#[test]
fn unknown_k_or_axis_never_claim_input_shape_is_output_shape() {
    let entry = &super::super::MAXK_CATALOG_ENTRY;
    let result = infer(
        &request(vec![Some(5), Some(2)], LiteralValue::Unknown),
        entry,
    );
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![None, Some(2)])
    );
    let result = infer(
        &request(vec![None, Some(5)], LiteralValue::Number(2.0)),
        entry,
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::Unknown);
    let mut req = request(vec![Some(1), None], LiteralValue::Number(2.0));
    req.arguments
        .push(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })));
    req.literals.literal_args.push(LiteralValue::Number(2.0));
    assert_eq!(
        infer(&req, entry).outputs[0].shape,
        ShapeFact::from(vec![Some(1), None])
    );
}
#[test]
fn invalid_inputs_options_and_output_counts_are_diagnosed() {
    let entry = &super::super::MINK_CATALOG_ENTRY;
    let mut req = request(vec![Some(2), Some(3)], LiteralValue::Number(-1.0));
    assert!(!infer(&req, entry).diagnostics.is_empty());
    req.literals.literal_args[1] = LiteralValue::Number(0.0);
    req.outputs = OutputSelection::new(RequestedOutputCount::Exactly(3));
    assert!(!infer(&req, entry).diagnostics.is_empty());
    req.outputs = OutputSelection::new(RequestedOutputCount::Exactly(2));
    req.arguments.extend([
        ValueFact::scalar(ValueKindFact::String),
        ValueFact::scalar(ValueKindFact::String),
    ]);
    req.literals.literal_args.extend([
        LiteralValue::String("ComparisonMethod".into()),
        LiteralValue::String("bogus".into()),
    ]);
    assert!(!infer(&req, entry).diagnostics.is_empty());
}
#[test]
fn zero_k_empty_and_resident_facts_are_sound() {
    let entry = &super::super::MAXK_CATALOG_ENTRY;
    let mut req = request(vec![Some(2), Some(0)], LiteralValue::Number(0.0));
    assert_eq!(
        infer(&req, entry).outputs[0].shape,
        ShapeFact::from(vec![Some(0), Some(0)])
    );
    req.arguments[0].residency = ResidencyFact::Device { provider: None };
    let result = infer(&req, entry);
    assert_eq!(result.outputs[0].residency, ResidencyFact::Unknown);
    assert_eq!(result.outputs[1].storage, StorageFact::Unknown);
}
