use super::*;
use runmat_accelerate_api::RealPowerOperand;
use runmat_value::Tensor;

pub(super) fn power_receipt(evidence: FusionAdmissionEvidence) -> FusionPowerArithmeticReceipt {
    let Some(FusionArithmeticEvidence::RealPower(receipt)) = evidence.arithmetic else {
        panic!("selected POWER arithmetic receipt is absent")
    };
    receipt
}

fn computed_base_with(base: Value, exponent: Value) -> Option<FusionAdmissionEvidence> {
    let facts = [
        crate::value_fact::value_fact(&base),
        crate::value_fact::value_fact(&exponent),
    ];
    let operands = [None, Some(&exponent)];
    let mut work = FusionAdmissionWork::None;
    futures::executor::block_on(admit(
        FusionAdmissionContext::new(&[0, 1], &operands, &facts, &[None, None], &mut work)
            .expect("aligned power operands"),
    ))
    .unwrap()
}

pub(super) fn computed_base(exponent: Value) -> Option<FusionAdmissionEvidence> {
    computed_base_with(
        Value::Tensor(Tensor::from_f32(vec![-3.0, f32::NAN], vec![1, 2]).unwrap()),
        exponent,
    )
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_finite_quote_uses_original_frame_and_certified_double_controls_only() {
    super::preparation::assert_selected_finite_quote();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rejected_paired_host_domain_retains_completed_scan_without_evidence() {
    let base = Value::Num(-1.0);
    let exponent = Value::Num(0.5);
    let facts = [
        crate::value_fact::value_fact(&base),
        crate::value_fact::value_fact(&exponent),
    ];
    let operands = [Some(&base), Some(&exponent)];
    let mut work = FusionAdmissionWork::None;
    let context = FusionAdmissionContext::new(&[0, 1], &operands, &facts, &[None, None], &mut work)
        .expect("aligned POWER source");
    assert!(admit_static(&context).unwrap().is_none());
    assert!(futures::executor::block_on(admit(context))
        .unwrap()
        .is_none());
    assert_eq!(work, FusionAdmissionWork::Host { elements: 1 });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn represented_integral_scalar_proves_computed_real_base_without_sampling_it() {
    for exponent in [-3.0, -0.0, 0.0, 2.0] {
        let admitted = computed_base(Value::Tensor(
            Tensor::from_f32(vec![exponent as f32], vec![1, 1]).unwrap(),
        ))
        .expect("matching Single integer power is real");
        assert_eq!(
            admitted.numerical_domain,
            NumericalDomainEvidence::RealPower(PowerResultDomain::Real)
        );
        assert_eq!(
            admitted.output_kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            })
        );
    }
    let mixed = computed_base(Value::Num(2.0)).expect("Double literal retains mixed arithmetic");
    let receipt = power_receipt(mixed);
    assert_eq!(
        receipt.mode,
        RealPowerArithmeticMode::MixedDoubleThenSingle {
            single_operand: RealPowerOperand::Base
        }
    );
    assert_eq!(
        receipt.exponent_scalar,
        Some(FusionScalarControl::DoubleBits(2.0f64.to_bits()))
    );
    assert_eq!((receipt.base_id, receipt.exponent_id), (0, 1));
    let near_rounding_boundary =
        Value::Tensor(Tensor::from_f32(vec![1.000_000_1, 0.999_999_94], vec![1, 2]).unwrap());
    assert!(computed_base_with(near_rounding_boundary, Value::Num(2.0)).is_some());
    let wide = computed_base(Value::Num(16_777_217.0)).expect("original Double control is exact");
    assert_eq!(
        power_receipt(wide).exponent_scalar,
        Some(FusionScalarControl::DoubleBits(16_777_217.0f64.to_bits()))
    );
    assert!(computed_base(Value::Num(0.5)).is_none());
    // This Double control rounds to an integral F32, but the original
    // nonintegral source exponent cannot prove a real result for every base.
    assert!(computed_base(Value::Num(16_777_216.5)).is_none());
    assert!(computed_base(Value::Num(1.3)).is_none());
    assert!(computed_base(Value::Num(f64::NAN)).is_none());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_computed_base_retains_single_exponent_class() {
    let base = Value::Tensor(Tensor::new(vec![-3.0, f64::NAN], vec![1, 2]).unwrap());
    let single_exponent = Value::Tensor(Tensor::from_f32(vec![-2.0], vec![1, 1]).unwrap());
    let mixed = computed_base_with(base.clone(), single_exponent).expect("mixed computation");
    assert_eq!(
        mixed.output_kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real
        })
    );
    assert_eq!(
        power_receipt(mixed).mode,
        RealPowerArithmeticMode::MixedDoubleThenSingle {
            single_operand: RealPowerOperand::Exponent
        }
    );
    let double_exponent = Value::Tensor(Tensor::new(vec![-2.0], vec![1, 1]).unwrap());
    let admitted = computed_base_with(base, double_exponent).expect("matched double domain");
    assert_eq!(
        admitted.output_kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nonfloating_or_unknown_computed_base_has_no_universal_proof() {
    let integer_base = Value::Tensor(
        Tensor::from_numeric_storage(runmat_value::NumericStorage::I32(vec![-3, 4]), vec![1, 2])
            .unwrap(),
    );
    assert!(computed_base_with(integer_base, Value::Num(2.0)).is_none());
    let exponent = Value::Tensor(Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap());
    let facts = [
        runmat_types::ValueFact::unknown(runmat_types::DynamicReason::Unspecified),
        crate::value_fact::value_fact(&exponent),
    ];
    let operands = [None, Some(&exponent)];
    let mut work = FusionAdmissionWork::None;
    let rejected = futures::executor::block_on(admit(
        FusionAdmissionContext::new(&[0, 1], &operands, &facts, &[None, None], &mut work)
            .expect("aligned power operands"),
    ))
    .unwrap();
    assert!(rejected.is_none());
    assert_eq!(work, FusionAdmissionWork::None);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn prior_nonnegative_proof_admits_fractional_single_exponent_only_for_its_operand() {
    let base = Value::Tensor(Tensor::from_f32(vec![0.0, f32::NAN], vec![1, 2]).unwrap());
    let exponent = Value::Tensor(Tensor::from_f32(vec![1.5], vec![1, 1]).unwrap());
    let facts = [
        crate::value_fact::value_fact(&base),
        crate::value_fact::value_fact(&exponent),
    ];
    let operands = [None, Some(&exponent)];
    let mut work = FusionAdmissionWork::None;
    let proof = FusionOperandDomainEvidence {
        value_id: 9,
        numerical_domain: NumericalDomainEvidence::NonnegativeOrNaN,
        sign: None,
    };
    let admitted = futures::executor::block_on(admit(
        FusionAdmissionContext::new(&[9, 10], &operands, &facts, &[Some(proof), None], &mut work)
            .expect("matching selected base evidence"),
    ))
    .unwrap()
    .expect("exact computed base proof");
    assert_eq!(
        admitted.numerical_domain,
        NumericalDomainEvidence::RealPower(PowerResultDomain::Real)
    );
    let rejected = futures::executor::block_on(admit(
        FusionAdmissionContext::new(&[10, 9], &operands, &facts, &[None, Some(proof)], &mut work)
            .expect("evidence belongs to the exponent, not base"),
    ))
    .unwrap();
    assert!(rejected.is_none());
    let wrong_base_id = FusionOperandDomainEvidence {
        value_id: 11,
        ..proof
    };
    assert!(FusionAdmissionContext::new(
        &[9, 10],
        &operands,
        &facts,
        &[Some(wrong_base_id), None],
        &mut work,
    )
    .is_none());
}
