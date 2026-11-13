use super::*;
use runmat_types::*;

fn numeric(dims: &[usize]) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(dims.iter().copied().map(Some).collect::<Vec<_>>()),
        StorageFact::Dense,
    )
}

fn infer(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>, outputs: usize) -> CallInference {
    inference::infer(
        &CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        },
        &TTEST2_CATALOG_ENTRY,
    )
}

#[test]
fn dimensional_outputs_are_host_logical_and_double_with_struct_fields() {
    let inputs = vec![
        numeric(&[3, 2]),
        numeric(&[4, 2]),
        ValueFact::scalar(ValueKindFact::Character),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
    ];
    let result = infer(
        inputs,
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Character("Dim".into()),
            LiteralValue::Number(1.0),
        ],
        4,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Logical);
    assert_eq!(
        result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(2)])
    );
    assert_eq!(
        result.outputs[1].numeric().unwrap().class,
        NumericClass::Double
    );
    assert_eq!(
        result.outputs[2].shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );
    assert!(matches!(result.outputs[3].kind, ValueKindFact::Struct(_)));
    assert!(result
        .outputs
        .iter()
        .all(|output| output.residency == ResidencyFact::Host));
    for dimension in [
        1025.0,
        f64::from_bits(2f64.powi(usize::BITS as i32).to_bits() - 1).floor(),
    ] {
        let result = infer(
            vec![
                numeric(&[3, 2]),
                numeric(&[4, 2]),
                ValueFact::scalar(ValueKindFact::Character),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            ],
            vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::Character("Dim".into()),
                LiteralValue::Number(dimension),
            ],
            4,
        );
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[1].shape, ShapeFact::Unknown);
        assert_eq!(result.outputs[2].shape, ShapeFact::Unknown);
    }
}

#[test]
fn malformed_control_or_nonreal_sample_does_not_prove_outputs() {
    let unsupported = ValueFact::proven(
        ValueKindFact::String,
        ShapeFact::from(vec![Some(3), Some(1)]),
        StorageFact::Dense,
    );
    let result = infer(vec![unsupported, numeric(&[3, 1])], vec![], 1);
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);

    let result = infer(
        vec![
            numeric(&[3, 1]),
            numeric(&[4, 1]),
            ValueFact::scalar(ValueKindFact::Character),
            ValueFact::scalar(ValueKindFact::Character),
        ],
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Character("Tail".into()),
            LiteralValue::Character("sideways".into()),
        ],
        1,
    );
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn known_wrong_control_carriers_do_not_publish_proven_facts() {
    let data = vec![numeric(&[3, 1]), numeric(&[4, 1])];
    let mut alpha_args = data.clone();
    alpha_args.extend([
        ValueFact::scalar(ValueKindFact::Character),
        ValueFact::scalar(ValueKindFact::Character),
    ]);
    let alpha = infer(
        alpha_args,
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Character("Alpha".into()),
            LiteralValue::Character("bad".into()),
        ],
        1,
    );
    assert!(!alpha.diagnostics.is_empty());
    assert_eq!(alpha.outputs[0].kind, ValueKindFact::Unknown);

    let mut tail_args = data;
    tail_args.extend([
        ValueFact::scalar(ValueKindFact::Character),
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
    ]);
    let tail = infer(
        tail_args,
        vec![
            LiteralValue::Unknown,
            LiteralValue::Unknown,
            LiteralValue::Character("Tail".into()),
            LiteralValue::Unknown,
        ],
        1,
    );
    assert!(!tail.diagnostics.is_empty());
    assert_eq!(tail.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn incompatible_known_non_test_axes_are_diagnosed() {
    let result = infer(vec![numeric(&[3, 2]), numeric(&[4, 3])], vec![], 1);
    assert!(!result.diagnostics.is_empty());
    assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn source_option_tokens_decode_to_canonical_semantics() {
    assert_eq!(
        Ttest2OptionName::from_public("vArTyPe"),
        Some(Ttest2OptionName::VarType)
    );
    assert_eq!(Ttest2Tail::from_public("larger"), Some(Ttest2Tail::Right));
    assert_eq!(
        Ttest2Variance::from_public("unequal"),
        Some(Ttest2Variance::Unequal)
    );
}
