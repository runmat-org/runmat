use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn double_round_and_remainder_keep_ieee_word_requirement() {
    let inputs = [
        Value::Tensor(runmat_value::Tensor::new(vec![2.5, -1.5], vec![1, 2]).unwrap()),
        Value::Tensor(runmat_value::Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
    ];
    let cases = [
        plan(vec![cast("round", 0, 2)], 2),
        plan(
            vec![FusionOp::Builtin {
                origin: Default::default(),
                name: "rem".into(),
                inputs: vec![0, 1],
                output: Some(2),
            }],
            2,
        ),
    ];
    for (name, selected_plan) in ["round", "rem"].into_iter().zip(cases) {
        let program = resolve(&selected_plan, &inputs).unwrap();
        let selected = selected_plan
            .prepare_typed_shader(&[2], &program, ScalarType::F64)
            .unwrap();
        assert_eq!(
            selected.requirements(),
            runmat_accelerate_api::ProviderShaderRequirements::F64IeeeWords
        );
        assert!(selected.source().contains("runmat_f64_low"));
        crate::wgsl::validation_tests::validate_generated(
            name,
            selected.source(),
            64,
            8,
            crate::wgsl::validation_tests::RequiredCaps::Float64Int64,
        );
    }
}

#[test]
fn power_uses_exact_paired_evidence_and_the_shared_exceptional_value_helper() {
    let power_plan = plan(
        vec![FusionOp::Primitive {
            op: PrimitiveOp::ElemPow,
            inputs: vec![0, 1],
            output: Some(2),
        }],
        2,
    );
    for inputs in [
        [Value::Num(-2.0), Value::Num(3.0)],
        [Value::Num(f64::NAN), Value::Num(0.0)],
        [Value::Num(-0.0), Value::Num(-3.0)],
    ] {
        let program = resolve(&power_plan, &inputs).unwrap();
        let shader = power_plan
            .generate_typed_wgsl(&[2], &program, ScalarType::F64)
            .unwrap();
        let selected = power_plan
            .prepare_typed_shader(&[2], &program, ScalarType::F64)
            .unwrap();
        assert_eq!(selected.source(), shader);
        assert_eq!(
            selected.requirements(),
            runmat_accelerate_api::ProviderShaderRequirements::F64IeeeWords
        );
        crate::wgsl::validation_tests::validate_generated(
            "selected-double-power",
            selected.source(),
            64,
            8,
            crate::wgsl::validation_tests::RequiredCaps::Float64Int64,
        );
        assert!(shader.contains("fn rm_power_f64("), "{shader}");
        assert!(shader.contains("= rm_power_f64("), "{shader}");
        assert_eq!(
            program.evidence.get(&2),
            Some(&runmat_accelerate_api::NumericalDomainEvidence::RealPower(
                runmat_accelerate_api::PowerResultDomain::Real
            ))
        );
    }
    let computed = plan(
        vec![
            cast("single", 0, 2),
            FusionOp::Primitive {
                op: PrimitiveOp::ElemPow,
                inputs: vec![2, 1],
                output: Some(3),
            },
        ],
        3,
    );
    let program = resolve(&computed, &[Value::Num(-2.0), Value::Num(3.0)]).unwrap();
    let receipt = program.power_arithmetic(3).expect("mixed square receipt");
    assert_eq!(
        receipt.mode,
        runmat_accelerate_api::RealPowerArithmeticMode::MixedDoubleThenSingle {
            single_operand: runmat_accelerate_api::RealPowerOperand::Base
        }
    );
    assert!(computed
        .generate_typed_wgsl(&[3], &program, ScalarType::F32)
        .is_err());
    let changed_control = resolve(&computed, &[Value::Num(-2.0), Value::Num(4.0)]).unwrap();
    assert_ne!(
        program.power_arithmetic(3),
        changed_control.power_arithmetic(3)
    );
    assert_ne!(
        program.cache_identity().unwrap(),
        changed_control.cache_identity().unwrap()
    );
    let mut changed_source = computed.clone();
    if let FusionOp::Primitive { inputs, .. } = &mut changed_source.operations[1] {
        inputs[0] = 0;
    }
    assert!(program.require_source_plan(&changed_source).is_err());

    let single_gamma =
        Value::Tensor(runmat_value::Tensor::from_f32(vec![1.8], vec![1, 1]).unwrap());
    let single_base = Value::Tensor(runmat_value::Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap());
    let native = plan(
        vec![FusionOp::Primitive {
            op: PrimitiveOp::ElemPow,
            inputs: vec![0, 1],
            output: Some(2),
        }],
        2,
    );
    let native_program = resolve(&native, &[single_base, single_gamma]).unwrap();
    assert_eq!(
        native_program.power_arithmetic(2).unwrap().mode,
        runmat_accelerate_api::RealPowerArithmeticMode::NativeSingle
    );
    assert_ne!(
        program.cache_identity().unwrap(),
        native_program.cache_identity().unwrap()
    );
    let double_base = Value::Num(2.0);
    let single_exponent =
        Value::Tensor(runmat_value::Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap());
    let inverse_mixed = resolve(&native, &[double_base, single_exponent]).unwrap();
    assert_eq!(
        inverse_mixed.power_arithmetic(2).unwrap().mode,
        runmat_accelerate_api::RealPowerArithmeticMode::MixedDoubleThenSingle {
            single_operand: runmat_accelerate_api::RealPowerOperand::Exponent
        }
    );
    assert_eq!(
        inverse_mixed.floating_class(2).unwrap(),
        NumericClass::Single
    );
    assert!(native
        .generate_typed_wgsl(&[2], &inverse_mixed, ScalarType::F32)
        .is_err());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn same_plan_power_receipts_distinguish_signed_zero_bits_and_original_class() {
    let source = plan(
        vec![FusionOp::Primitive {
            op: PrimitiveOp::ElemPow,
            inputs: vec![0, 1],
            output: Some(2),
        }],
        2,
    );
    let base = || Value::Tensor(runmat_value::Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap());
    let positive = resolve(&source, &[base(), Value::Num(0.0)]).unwrap();
    let negative = resolve(&source, &[base(), Value::Num(-0.0)]).unwrap();
    assert_ne!(positive.power_arithmetic(2), negative.power_arithmetic(2));
    assert_ne!(
        positive.cache_identity().unwrap(),
        negative.cache_identity().unwrap()
    );

    let double = resolve(&source, &[base(), Value::Num(2.0)]).unwrap();
    let single = resolve(
        &source,
        &[
            base(),
            Value::Tensor(runmat_value::Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap()),
        ],
    )
    .unwrap();
    assert_eq!(
        double.power_arithmetic(2).unwrap().exponent_class,
        NumericClass::Double
    );
    assert_eq!(
        single.power_arithmetic(2).unwrap().exponent_class,
        NumericClass::Single
    );
    assert_ne!(
        double.cache_identity().unwrap(),
        single.cache_identity().unwrap()
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn static_power_resolution_keeps_computed_producer_identity_and_matches_async_receipt() {
    let source = plan(
        vec![
            cast("single", 0, 2),
            FusionOp::Primitive {
                op: PrimitiveOp::ElemPow,
                inputs: vec![2, 1],
                output: Some(3),
            },
        ],
        3,
    );
    let inputs = [Value::Num(-2.0), Value::Num(3.0)];
    let borrowed: Vec<_> = inputs.iter().collect();
    let bindings = semantic_bindings::capture(&source).unwrap();
    let static_program = ResolvedProgram::resolve_static_captured(
        &source,
        &borrowed,
        ProviderPrecision::F32,
        &bindings,
    )
    .unwrap()
    .expect("integral exponent proves a computed base without a probe");
    let async_program = futures::executor::block_on(ResolvedProgram::resolve(
        &source,
        &inputs,
        ProviderPrecision::F32,
        None,
    ))
    .unwrap();
    assert_eq!(
        static_program.power_arithmetic(3),
        async_program.power_arithmetic(3)
    );
    assert_eq!(
        static_program.cache_identity().unwrap(),
        async_program.cache_identity().unwrap()
    );
    let mut rewired = source.clone();
    if let FusionOp::Primitive { inputs, .. } = &mut rewired.operations[1] {
        inputs[0] = 0;
    }
    assert!(static_program.require_source_plan(&rewired).is_err());
}
