use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn certified_power_fusion_embeds_original_double_controls_before_operand_casts() {
    let source = plan(
        vec![
            FusionOp::Primitive {
                op: PrimitiveOp::ElemPow,
                inputs: vec![0, 1],
                output: Some(2),
            },
            FusionOp::Primitive {
                op: PrimitiveOp::UPlus,
                inputs: vec![2],
                output: Some(3),
            },
        ],
        3,
    );
    let base = Value::Tensor(runmat_value::Tensor::from_f32(vec![-0.0, -2.0], vec![1, 2]).unwrap());
    let mut identities = Vec::new();
    for exponent in [0.0_f64, -0.0, 1.0, 2.0] {
        for policy in [PrecisionPolicy::Exact, PrecisionPolicy::AllowDoubleToSingle] {
            let program = resolve_policy(
                &source,
                &[base.clone(), Value::Num(exponent)],
                ProviderPrecision::F32,
                policy,
            )
            .unwrap();
            let receipt = program.power_arithmetic(2).unwrap();
            assert_eq!(
                receipt.exponent_scalar,
                Some(runmat_accelerate_api::PowerScalarControl::DoubleBits(
                    exponent.to_bits()
                ))
            );
            assert_eq!(
                receipt.mode,
                runmat_accelerate_api::RealPowerArithmeticMode::MixedDoubleThenSingle {
                    single_operand: runmat_accelerate_api::RealPowerOperand::Base
                }
            );
            let generated = source
                .prepare_typed_shader(&[3], &program, ScalarType::F32)
                .unwrap();
            let (shader, requirements, data_inputs) = generated.into_parts();
            assert_eq!(data_inputs, [0]);
            assert_eq!(
                requirements,
                runmat_accelerate_api::ProviderShaderRequirements::Basic
            );
            assert!(shader.contains(": u32 = rm_exact_power_"));
            assert!(shader.contains("struct Tensor { data: array<u32>"));
            assert!(!shader.contains("var<storage, read> input1"));
            assert!(!shader.contains("fn rm_power_f32("));
            // Capability validation below checks actual types. The source
            // identity comment can contain arbitrary hexadecimal substrings.
            crate::wgsl::validation_tests::validate_generated(
                "certified-power-fusion",
                &shader,
                64,
                8,
                crate::wgsl::validation_tests::RequiredCaps::None,
            );
            // Captured metadata controls share constant-leaf normalization,
            // but must not instantiate a Double data value in an F32 shader.
            // Two results force the complete, non-checked fallback as well.
            let mut captured = source.clone();
            captured.inputs = vec![0];
            captured.const_values.insert(1, Value::Num(exponent));
            let captured_program = resolve_policy(
                &captured,
                std::slice::from_ref(&base),
                ProviderPrecision::F32,
                policy,
            )
            .unwrap();
            let generated = captured
                .prepare_typed_shader(&[2, 3], &captured_program, ScalarType::F32)
                .unwrap();
            assert_eq!(
                generated.requirements(),
                runmat_accelerate_api::ProviderShaderRequirements::Basic
            );
            assert!(!generated.source().contains("let control1"));
            crate::wgsl::validation_tests::validate_generated(
                "captured-power-control",
                generated.source(),
                64,
                8,
                crate::wgsl::validation_tests::RequiredCaps::None,
            );
            if policy == PrecisionPolicy::Exact {
                identities.push(program.cache_identity().unwrap());
            }
        }
    }
    assert_ne!(
        identities[0], identities[1],
        "signed Double control bits remain semantic identity"
    );
    let f64_program = resolve_policy(
        &source,
        &[base.clone(), Value::Num(2.0)],
        ProviderPrecision::F64,
        PrecisionPolicy::Exact,
    )
    .unwrap();
    assert!(
        source
            .prepare_typed_shader(&[3], &f64_program, ScalarType::F64)
            .is_err(),
        "legacy F64 data staging is not certified binary32 word transport"
    );
    for exponent in [3.0_f64, f64::from_bits(2.0_f64.to_bits() + 1), 0.5] {
        let inputs = [base.clone(), Value::Num(exponent)];
        if let Ok(program) = resolve_policy(
            &source,
            &inputs,
            ProviderPrecision::F32,
            PrecisionPolicy::Exact,
        ) {
            assert!(source
                .prepare_typed_shader(&[3], &program, ScalarType::F32)
                .is_err());
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn certified_exponent_fanout_cannot_be_omitted_from_physical_data_inputs() {
    let source = plan(
        vec![
            FusionOp::Primitive {
                op: PrimitiveOp::ElemPow,
                inputs: vec![0, 1],
                output: Some(2),
            },
            FusionOp::Primitive {
                op: PrimitiveOp::Add,
                inputs: vec![2, 1],
                output: Some(3),
            },
        ],
        3,
    );
    let base = Value::Tensor(runmat_value::Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let program = resolve_policy(
        &source,
        &[base, Value::Num(2.0)],
        ProviderPrecision::F64,
        PrecisionPolicy::Exact,
    )
    .unwrap();
    assert!(program.embedded_scalar_controls(&source, &[3]).is_empty());
    assert!(program
        .embedded_scalar_controls(&source, &[1, 2])
        .is_empty());
    assert!(
        source
            .prepare_typed_shader_with_inputs(&[3], &program, ScalarType::F32, &[0])
            .is_err(),
        "fanout data operand cannot be omitted by a variant projection"
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn single_fallback_shaders_preserve_primary_physical_projection_and_output_binding() {
    let source = plan(
        vec![FusionOp::Primitive {
            op: PrimitiveOp::ElemPow,
            inputs: vec![0, 1],
            output: Some(2),
        }],
        2,
    );
    let base = Value::Tensor(runmat_value::Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let program = resolve_policy(
        &source,
        &[base, Value::Num(2.0)],
        ProviderPrecision::F32,
        PrecisionPolicy::Exact,
    )
    .unwrap();
    // Public Double control output currently forces runtime F32 admission to
    // decline. Still verify the shader projection invariant independently so
    // extending that ABI cannot make fallback rebind prepared handles.
    for outputs in [&[0, 2][..], &[1, 2][..]] {
        let (_, _, projection) = source
            .prepare_typed_shader(outputs, &program, ScalarType::F32)
            .unwrap()
            .into_parts();
        assert_eq!(
            projection,
            if outputs[0] == 0 { vec![0] } else { vec![0, 1] }
        );
        for &output in outputs {
            let single = source
                .prepare_typed_shader_with_inputs(&[output], &program, ScalarType::F32, &projection)
                .unwrap();
            assert!(single.source().contains(&format!(
                "@binding({}) var<storage, read_write> output0",
                projection.len()
            )));
            assert!(single.source().contains(&format!(
                "@binding({}) var<uniform> params",
                projection.len() + 1
            )));
            let (shader, _, inputs) = single.into_parts();
            assert_eq!(inputs, projection);
            crate::wgsl::validation_tests::validate_generated(
                "stable-projection-single-fallback",
                &shader,
                64,
                8,
                crate::wgsl::validation_tests::RequiredCaps::None,
            );
        }
    }
}

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

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_scalar_cast_and_arithmetic_control_admit_produced_power_exponent() {
    let mut source = plan(
        vec![
            cast("single", 1, 2),
            FusionOp::Primitive {
                op: PrimitiveOp::Add,
                inputs: vec![2, 3],
                output: Some(4),
            },
            FusionOp::Primitive {
                op: PrimitiveOp::ElemPow,
                inputs: vec![0, 4],
                output: Some(5),
            },
        ],
        5,
    );
    source.const_values.insert(
        3,
        Value::Tensor(runmat_value::Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap()),
    );
    let base = Value::Tensor(runmat_value::Tensor::from_f32(vec![-2.0, 3.0], vec![1, 2]).unwrap());
    let inputs = [base, Value::Num(2.0)];
    let program = resolve_policy(
        &source,
        &inputs,
        ProviderPrecision::F32,
        PrecisionPolicy::Exact,
    )
    .expect("selected pure scalar producers retain exact Single exponent");
    assert_eq!(
        program.power_arithmetic(5).unwrap().exponent_scalar,
        Some(runmat_accelerate_api::PowerScalarControl::SingleBits(
            2.0_f32.to_bits()
        ))
    );
    let borrowed: Vec<_> = inputs.iter().collect();
    let bindings = semantic_bindings::capture(&source).unwrap();
    let static_program = ResolvedProgram::resolve_static_captured(
        &source,
        &borrowed,
        ProviderPrecision::F32,
        &bindings,
    )
    .unwrap()
    .expect("same selected scalar control is available to static admission");
    assert_eq!(
        static_program.power_arithmetic(5),
        program.power_arithmetic(5)
    );
    let mut named = source.clone();
    named.operations[1] = FusionOp::Builtin {
        origin: Default::default(),
        name: "plus".into(),
        inputs: vec![2, 3],
        output: Some(4),
    };
    let named_program = resolve_policy(
        &named,
        &inputs,
        ProviderPrecision::F32,
        PrecisionPolicy::Exact,
    )
    .expect("captured named arithmetic uses its selected represented recipe");
    assert_eq!(
        named_program.power_arithmetic(5),
        program.power_arithmetic(5)
    );
    assert_eq!(
        static_program.cache_identity().unwrap(),
        program.cache_identity().unwrap()
    );
    let mut changed = inputs.clone();
    changed[1] = Value::Num(3.0);
    let changed_program = resolve_policy(
        &source,
        &changed,
        ProviderPrecision::F32,
        PrecisionPolicy::Exact,
    )
    .unwrap();
    assert_ne!(
        program.power_arithmetic(5).unwrap().exponent_scalar,
        changed_program.power_arithmetic(5).unwrap().exponent_scalar
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn produced_scalar_evidence_records_exact_preparation_without_provider_work() {
    use crate::placement::{begin_trace, report, PlacementEventKind};

    let mut source = plan(
        vec![
            cast("single", 1, 2),
            FusionOp::Primitive {
                op: PrimitiveOp::Add,
                inputs: vec![2, 3],
                output: Some(4),
            },
        ],
        4,
    );
    source.const_values.insert(
        3,
        Value::Tensor(runmat_value::Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap()),
    );
    let inputs = [
        Value::Tensor(runmat_value::Tensor::from_f32(vec![-2.0, 3.0], vec![1, 2]).unwrap()),
        Value::Num(2.0),
    ];
    let trace = begin_trace("fusion.scalar_evidence_preparation", None);
    let correlation = trace.correlation();
    let program = futures::executor::block_on(ResolvedProgram::resolve(
        &source,
        &inputs,
        ProviderPrecision::F32,
        Some(correlation),
    ))
    .expect("selected scalar control is admitted");
    assert_eq!(
        program.fact(4).unwrap().numeric().unwrap().class,
        NumericClass::Single
    );
    let mut nonmonotonic = source.clone();
    nonmonotonic.operations[0] = cast("single", 1, 9);
    nonmonotonic.operations[1] = FusionOp::Primitive {
        op: PrimitiveOp::Add,
        inputs: vec![9, 3],
        output: Some(4),
    };
    let reordered = futures::executor::block_on(ResolvedProgram::resolve(
        &nonmonotonic,
        &inputs,
        ProviderPrecision::F32,
        None,
    ))
    .expect("indexed scalar slots follow producer identity, not operation order");
    assert_eq!(reordered.fact(4).unwrap(), program.fact(4).unwrap());
    trace.complete(None, None);
    let first_trace_report = report();
    let recorded = first_trace_report
        .traces
        .iter()
        .find(|trace| trace.correlation == correlation)
        .unwrap();
    let events = recorded
        .events
        .iter()
        .filter(|event| event.reason.as_deref() == Some("fusion_scalar_evidence_wall"))
        .collect::<Vec<_>>();
    assert_eq!(events.len(), 1);
    let event = events[0];
    assert_eq!(event.kind, PlacementEventKind::Prepare);
    assert!(event.duration_ns.is_some());
    assert_eq!(event.bytes, None);
    for (key, expected) in [("completed", 1), ("attempted", 2), ("produced", 2)] {
        assert!(event
            .attributes
            .iter()
            .any(|attribute| attribute.key == key && attribute.value == expected));
    }

    source.operations.push(FusionOp::Primitive {
        op: PrimitiveOp::ElemPow,
        inputs: vec![0, 4],
        output: Some(5),
    });
    source.output = Some(5);
    let rejected = begin_trace("fusion.scalar_evidence_then_domain_rejection", None);
    let rejected_correlation = rejected.correlation();
    let rejected_inputs = [inputs[0].clone(), Value::Num(0.5)];
    assert!(futures::executor::block_on(ResolvedProgram::resolve(
        &source,
        &rejected_inputs,
        ProviderPrecision::F32,
        Some(rejected_correlation),
    ))
    .is_err());
    rejected.complete(None, None);
    let rejected_trace_report = report();
    let rejected_trace = rejected_trace_report
        .traces
        .iter()
        .find(|trace| trace.correlation == rejected_correlation)
        .unwrap();
    let event = rejected_trace
        .events
        .iter()
        .find(|event| event.reason.as_deref() == Some("fusion_scalar_evidence_wall"))
        .unwrap();
    assert!(event.duration_ns.is_some());
    for (key, expected) in [("completed", 0), ("attempted", 2), ("produced", 2)] {
        assert!(event
            .attributes
            .iter()
            .any(|attribute| attribute.key == key && attribute.value == expected));
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn vector_producer_does_not_gain_a_fake_value_when_scalar_control_is_computed() {
    let source = plan(
        vec![
            cast("tanh", 0, 2),
            cast("single", 1, 3),
            FusionOp::Primitive {
                op: PrimitiveOp::ElemPow,
                inputs: vec![2, 3],
                output: Some(4),
            },
        ],
        4,
    );
    let signal = Value::Tensor(
        runmat_value::Tensor::from_f32(vec![0.125, 0.25, 0.5, 0.75], vec![4, 1]).unwrap(),
    );
    let program = resolve_policy(
        &source,
        &[signal, Value::Num(2.0)],
        ProviderPrecision::F32,
        PrecisionPolicy::Exact,
    )
    .expect("an exact computed exponent proves any real vector base");
    let receipt = program.power_arithmetic(4).unwrap();
    assert_eq!(receipt.base_id, 2);
    assert_eq!(receipt.exponent_id, 3);
    assert_eq!(
        receipt.exponent_scalar,
        Some(runmat_accelerate_api::PowerScalarControl::SingleBits(
            2.0_f32.to_bits()
        ))
    );
    assert_eq!(
        program.evidence.get(&4),
        Some(&runmat_accelerate_api::NumericalDomainEvidence::RealPower(
            runmat_accelerate_api::PowerResultDomain::Real
        ))
    );
    let non_scalar_control = plan(
        vec![
            cast("single", 1, 3),
            FusionOp::Primitive {
                op: PrimitiveOp::ElemPow,
                inputs: vec![0, 3],
                output: Some(4),
            },
        ],
        4,
    );
    let vector = Value::Tensor(runmat_value::Tensor::new(vec![2.0, 4.0], vec![1, 2]).unwrap());
    let base = Value::Tensor(runmat_value::Tensor::from_f32(vec![-2.0, 3.0], vec![1, 2]).unwrap());
    assert!(matches!(
        resolve_policy(
            &non_scalar_control,
            &[base, vector],
            ProviderPrecision::F32,
            PrecisionPolicy::Exact,
        ),
        Err(FusionResolveFailure::Structural(_))
    ));
}
