use super::*;
use crate::fusion::{FusionGroup, FusionKernelSpec, FusionKind};
use crate::graph::{InstrSpan, PrimitiveOp, ShapeInfo};
use crate::precision::PrecisionPolicy;
use runmat_accelerate_api::ProviderPrecision;
use runmat_value::Value;

#[path = "typed_producer_primitive_tests.rs"]
mod primitive_tests;
#[path = "typed_producer_unary_tests.rs"]
mod unary_tests;

fn plan(operations: Vec<FusionOp>, output: ValueId) -> FusionGroupPlan {
    FusionGroupPlan {
        index: 0,
        group: FusionGroup {
            id: 0,
            kind: FusionKind::ElementwiseChain,
            nodes: vec![],
            shape: ShapeInfo::Scalar,
            span: InstrSpan { start: 0, end: 4 },
            pattern: None,
            stack_layout: None,
        },
        operations,
        inputs: vec![0, 1],
        stack_pattern: vec![0, 1],
        constants: HashMap::new(),
        const_values: HashMap::new(),
        materialized_stores: vec![],
        output: Some(output),
        kernel: FusionKernelSpec {
            kind: FusionKind::ElementwiseChain,
            supported: true,
        },
        reduction_data: None,
        reduction_dim: None,
        reduction_flavor: None,
        reduction_axes: None,
        pattern: None,
    }
}

fn add(left: ValueId, right: ValueId, output: ValueId) -> FusionOp {
    FusionOp::Primitive {
        op: PrimitiveOp::Add,
        inputs: vec![left, right],
        output: Some(output),
    }
}
fn leaves() -> HashMap<ValueId, String> {
    HashMap::from([(0, "input0.data[i0]".into()), (1, "input1.data[i1]".into())])
}
fn resolve(
    plan: &FusionGroupPlan,
    inputs: &[Value],
    precision: ProviderPrecision,
) -> ResolvedProgram {
    futures::executor::block_on(ResolvedProgram::resolve_with_policy(
        plan,
        inputs,
        precision,
        PrecisionPolicy::Exact,
    ))
    .unwrap()
}

#[test]
fn captured_scalar_leaves_seed_exact_words_for_single_and_multi_output_fallbacks() {
    use crate::wgsl::validation_tests::{validate_generated, RequiredCaps};
    use runmat_accelerate_api::ProviderShaderRequirements;

    for single in [false, true] {
        let cases = if single {
            vec![0.1_f32.to_bits() as u64, 0x8000_0000, 1, 0x7fc0_0042]
        } else {
            vec![
                0.1_f64.to_bits(),
                0x8000_0000_0000_0000,
                1,
                0x7ff8_0000_0000_0042,
            ]
        };
        for bits in cases {
            let mut source = plan(vec![add(0, 1, 2), add(2, 1, 3)], 3);
            source.inputs = vec![0];
            let (input, value, scalar, precision, expected, requirements, caps) = if single {
                (
                    Value::Tensor(
                        runmat_value::Tensor::from_f32(vec![0.25, 0.5], vec![1, 2]).unwrap(),
                    ),
                    Value::Tensor(
                        runmat_value::Tensor::from_f32(
                            vec![f32::from_bits(bits as u32)],
                            vec![1, 1],
                        )
                        .unwrap(),
                    ),
                    ScalarType::F32,
                    ProviderPrecision::F32,
                    format!("bitcast<f32>(0x{bits:08x}u)"),
                    ProviderShaderRequirements::Basic,
                    RequiredCaps::None,
                )
            } else {
                (
                    Value::Tensor(runmat_value::Tensor::new(vec![0.25, 0.5], vec![1, 2]).unwrap()),
                    Value::Num(f64::from_bits(bits)),
                    ScalarType::F64,
                    ProviderPrecision::F64,
                    format!(
                        "runmat_f64_from_words(0x{:08x}u, 0x{:08x}u)",
                        bits as u32,
                        (bits >> 32) as u32
                    ),
                    ProviderShaderRequirements::F64IeeeWords,
                    RequiredCaps::Float64Int64,
                )
            };
            source.const_values.insert(1, value);
            // An unused aggregate is not an additional scalar leaf to render.
            source.const_values.insert(
                99,
                Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            );
            let program = resolve(&source, std::slice::from_ref(&input), precision);
            for outputs in [&[3][..], &[2, 3][..], &[1, 3][..]] {
                let bound = source
                    .measured_typed_shader_preparation_bytes(outputs, &program, scalar, 1)
                    .unwrap();
                let shader = source
                    .prepare_typed_shader_with_inputs_bounded(
                        outputs,
                        &program,
                        scalar,
                        &[0],
                        Some(bound.body_bytes),
                    )
                    .unwrap();
                assert!(
                    bound.retained_bytes >= shader.source().len() as u64,
                    "retained source exceeded pre-admitted geometry"
                );
                assert!(shader.source().contains(&expected), "{}", shader.source());
                assert_eq!(shader.source().matches("let control1 =").count(), 1);
                assert!(!shader.source().contains("control99"));
                assert_eq!(shader.requirements(), requirements);
                validate_generated("captured-scalar-leaves", shader.source(), 64, 8, caps);
            }
            // Same-ID captured values must never replace actual invocation
            // operands, including when their literal class differs.
            source.inputs.push(1);
            let program = resolve(&source, &[input.clone(), input], precision);
            let shader = source.prepare_typed_shader(&[3], &program, scalar).unwrap();
            assert!(!shader.source().contains("control1"));
            assert_eq!(shader.requirements(), ProviderShaderRequirements::Basic);
        }
    }
}

#[test]
fn captured_double_scalar_narrowing_requires_the_admitted_precision_policy() {
    let mut source = plan(vec![add(0, 1, 2)], 2);
    source.inputs = vec![0];
    source.const_values.insert(1, Value::Num(0.1));
    let input = Value::Tensor(runmat_value::Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap());
    for policy in [PrecisionPolicy::Exact, PrecisionPolicy::AllowDoubleToSingle] {
        let program = futures::executor::block_on(ResolvedProgram::resolve_with_policy(
            &source,
            std::slice::from_ref(&input),
            ProviderPrecision::F32,
            policy,
        ))
        .unwrap();
        let shader = source.prepare_typed_shader(&[2], &program, ScalarType::F32);
        if policy == PrecisionPolicy::Exact {
            assert!(shader.is_err());
        } else {
            let shader = shader.unwrap();
            assert!(shader.source().contains(&format!(
                "bitcast<f32>(0x{:08x}u)",
                (0.1_f64 as f32).to_bits()
            )));
            assert!(!shader.source().contains("runmat_f64_from_words"));
            assert_eq!(
                shader.requirements(),
                runmat_accelerate_api::ProviderShaderRequirements::Basic
            );
            crate::wgsl::validation_tests::validate_generated(
                "opted-captured-double",
                shader.source(),
                64,
                8,
                crate::wgsl::validation_tests::RequiredCaps::None,
            );
        }
    }
}

#[test]
fn root_slice_keeps_shared_dependencies_once_and_original_temporary_indices() {
    let plan = plan(
        vec![add(0, 1, 2), add(0, 0, 3), add(2, 2, 4), add(2, 4, 5)],
        5,
    );
    let program = resolve(
        &plan,
        &[Value::Num(1.0), Value::Num(2.0)],
        ProviderPrecision::F64,
    );
    let sliced =
        TypedProducer::emit(&plan, &program, ScalarType::F64, leaves(), Some(&[4, 5])).unwrap();
    assert_eq!(sliced.body.matches("let typed0:").count(), 1);
    assert!(!sliced.body.contains("let typed1:"));
    assert!(sliced.body.contains("let typed2:"));
    assert!(sliced.body.contains("let typed3:"));
    assert_eq!(sliced.expression(4).unwrap(), "typed2");
    assert_eq!(sliced.expression(5).unwrap(), "typed3");
    let full = TypedProducer::emit(&plan, &program, ScalarType::F64, leaves(), None).unwrap();
    assert!(full.body.contains("let typed1:"));
    assert_eq!(
        full.preamble(&program).unwrap(),
        sliced.preamble(&program).unwrap()
    );
    // Slice identity conservatively retains every admitted fact, while shader
    // text independently captures the different selected producer body.
    assert_ne!(full.body, sliced.body);
}

#[test]
fn full_elementwise_source_and_rounding_are_unchanged() {
    let plan = plan(vec![add(0, 1, 2), add(2, 1, 3)], 3);
    assert_eq!(plan.complete_single_output_root(&[3]).unwrap(), Some(3));
    assert_eq!(plan.complete_single_output_root(&[2]).unwrap(), None);
    assert_eq!(plan.complete_single_output_root(&[2, 3]).unwrap(), None);
    let program = resolve(
        &plan,
        &[Value::Num(1.0), Value::Num(2.0)],
        ProviderPrecision::F64,
    );
    let producer = TypedProducer::emit(&plan, &program, ScalarType::F64, leaves(), None).unwrap();
    let checked = TypedProducer::emit_checked_program(&plan, &program, ScalarType::F64, 3)
        .unwrap()
        .unwrap();
    assert_eq!(checked.wire().nodes.len(), 4);
    let lowered = TypedProducer::lower_checked_wgsl(&checked, ScalarType::F64, leaves())
        .unwrap()
        .unwrap();
    assert_eq!(producer.body, lowered.body);
    let selected =
        TypedProducer::emit(&plan, &program, ScalarType::F64, leaves(), Some(&[3])).unwrap();
    assert_eq!(selected.body, lowered.body);
    let expected_body = "    let typed0: f64 = (f64(input0.data[i0]) + f64(input1.data[i1]));\n    let typed1: f64 = (f64(typed0) + f64(input1.data[i1]));\n";
    assert_eq!(producer.body, expected_body);
    let expected = format!(
        "// {}\n\n{}",
        program.cache_identity().unwrap(),
        plan.build_wgsl_shader(
            "f64",
            "@group(0) @binding(2) var<storage, read_write> output0: Tensor;\n",
            3,
            expected_body,
            "    output0.data[g] = f64(typed1);\n",
            false,
        )
    );
    assert_eq!(
        plan.generate_typed_wgsl(&[3], &program, ScalarType::F64)
            .unwrap(),
        expected
    );

    let single = resolve(
        &plan,
        &[
            Value::Tensor(runmat_value::Tensor::from_f32(vec![16_777_216.0], vec![1, 1]).unwrap()),
            Value::Tensor(runmat_value::Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap()),
        ],
        ProviderPrecision::F64,
    );
    let producer =
        TypedProducer::emit(&plan, &single, ScalarType::F64, leaves(), Some(&[3])).unwrap();
    assert!(producer.body.contains("let typed0: f32"));
    assert!(producer.body.contains("let typed1: f32"));
    assert_eq!((16_777_216.0_f32 + 1.0) + 1.0, 16_777_216.0);
    assert_eq!((16_777_216.0_f64 + 1.0) + 1.0, 16_777_218.0);
    let strict = resolve(
        &plan,
        &[Value::Num(1.0), Value::Num(2.0)],
        ProviderPrecision::F32,
    );
    assert!(TypedProducer::emit(&plan, &strict, ScalarType::F32, leaves(), Some(&[3])).is_err());
}

#[test]
fn checked_program_uses_captured_cast_binding_and_exact_constant_bits() {
    use runmat_accelerate_api::scalar_program::{ScalarLiteral, ScalarNode};
    use runmat_types::ScalarOperation;

    let cast = plan(
        vec![FusionOp::Builtin {
            origin: Default::default(),
            name: "single".into(),
            inputs: vec![0],
            output: Some(2),
        }],
        2,
    );
    let facts = resolve(
        &cast,
        &[Value::Num(16_777_217.0), Value::Num(0.0)],
        ProviderPrecision::F64,
    );
    let checked = TypedProducer::emit_checked_program(&cast, &facts, ScalarType::F64, 2)
        .unwrap()
        .unwrap();
    assert!(matches!(
        &checked.wire().nodes[1],
        ScalarNode::Apply {
            operation: ScalarOperation::Cast(NumericClass::Single),
            ..
        }
    ));
    let before = TypedProducer::emit(&cast, &facts, ScalarType::F64, leaves(), Some(&[2]))
        .unwrap()
        .body;
    assert!(before.contains("f32(input0.data[i0])"));

    let mut signed_zero = plan(vec![add(0, 1, 2)], 2);
    signed_zero.inputs = vec![0];
    signed_zero.const_values.insert(1, Value::Num(-0.0));
    let facts = resolve(&signed_zero, &[Value::Num(1.0)], ProviderPrecision::F64);
    let checked = TypedProducer::emit_checked_program(&signed_zero, &facts, ScalarType::F64, 2)
        .unwrap()
        .unwrap();
    assert!(
        TypedProducer::lower_checked_wgsl(&checked, ScalarType::F64, leaves())
            .unwrap()
            .is_none()
    );
    assert!(matches!(
        &checked.wire().nodes[1],
        ScalarNode::Literal {
            value: ScalarLiteral::F64Bits(bits),
            ..
        } if *bits == (-0.0f64).to_bits()
    ));
}

#[test]
fn malformed_advertised_cast_cannot_fall_back_to_legacy_expression() {
    let cast = plan(
        vec![FusionOp::Builtin {
            origin: Default::default(),
            name: "single".into(),
            inputs: vec![0],
            output: Some(2),
        }],
        2,
    );
    let facts = resolve(
        &cast,
        &[Value::Num(3.0), Value::Num(4.0)],
        ProviderPrecision::F64,
    );
    let malformed = plan(
        vec![FusionOp::Builtin {
            origin: Default::default(),
            name: "single".into(),
            inputs: vec![0, 1],
            output: Some(2),
        }],
        2,
    );
    assert!(
        TypedProducer::emit(&malformed, &facts, ScalarType::F64, leaves(), Some(&[2])).is_err()
    );
}

#[test]
fn later_closed_binding_is_validated_after_uncounted_legacy_operation() {
    let source = plan(
        vec![
            FusionOp::Builtin {
                origin: Default::default(),
                name: "round".into(),
                inputs: vec![0],
                output: Some(2),
            },
            FusionOp::Builtin {
                origin: Default::default(),
                name: "single".into(),
                inputs: vec![2],
                output: Some(3),
            },
        ],
        3,
    );
    let facts = resolve(
        &source,
        &[Value::Num(2.5), Value::Num(1.0)],
        ProviderPrecision::F64,
    );
    let mut malformed = source;
    let FusionOp::Builtin { inputs, .. } = &mut malformed.operations[1] else {
        unreachable!()
    };
    inputs.push(1);
    let selected = producer_indices(&malformed, &[0, 1], &[3]).unwrap();
    let error = program::selected_resolution_receipt(&malformed, &facts, &selected).unwrap_err();
    assert!(error.to_string().contains("arity"), "{error:#}");
}

#[test]
fn unsupported_scalar_contract_does_not_change_existing_wgsl_route() {
    let source = plan(
        vec![FusionOp::Builtin {
            origin: Default::default(),
            name: "round".into(),
            inputs: vec![0],
            output: Some(2),
        }],
        2,
    );
    let facts = resolve(
        &source,
        &[Value::Num(2.5), Value::Num(0.0)],
        ProviderPrecision::F64,
    );
    assert!(
        TypedProducer::emit_checked_program(&source, &facts, ScalarType::F64, 2)
            .unwrap()
            .is_none()
    );
    assert!(
        TypedProducer::emit(&source, &facts, ScalarType::F64, leaves(), Some(&[2]))
            .unwrap()
            .body
            .contains("rm_round_f64")
    );

    // Round's inferred output carrier is not itself a physical output buffer.
    // Finite geometry must use the same represented Dense fact as the emitter
    // when the next selected arithmetic operation consumes that value.
    let chain = plan(
        vec![
            FusionOp::Builtin {
                origin: Default::default(),
                name: "round".into(),
                inputs: vec![0],
                output: Some(2),
            },
            add(2, 1, 3),
        ],
        3,
    );
    let first = Value::Tensor(runmat_value::Tensor::from_f32(vec![2.5, 3.5], vec![1, 2]).unwrap());
    let second = Value::Tensor(runmat_value::Tensor::from_f32(vec![1.0, 1.0], vec![1, 2]).unwrap());
    let program = resolve(&chain, &[first, second], ProviderPrecision::F32);
    assert!(matches!(
        program.fact(2).unwrap().storage,
        runmat_types::StorageFact::Unknown | runmat_types::StorageFact::Opaque
    ));
    for outputs in [&[3][..], &[2, 3][..]] {
        let bound = chain
            .measured_typed_shader_preparation_bytes(outputs, &program, ScalarType::F32, 2)
            .expect("the selected represented chain has finite geometry");
        let shader = chain
            .prepare_typed_shader_with_inputs_bounded(
                outputs,
                &program,
                ScalarType::F32,
                &[0, 1],
                Some(bound.body_bytes),
            )
            .expect("the measured represented chain emits");
        assert!(bound.retained_bytes >= shader.source().len() as u64);
    }
}

#[test]
fn input_root_keeps_supplied_physical_index_expression() {
    let plan = plan(vec![add(0, 1, 2)], 2);
    let program = resolve(
        &plan,
        &[Value::Num(1.0), Value::Num(2.0)],
        ProviderPrecision::F64,
    );
    let mut inputs = leaves();
    inputs.insert(0, "input0.data[broadcast_index]".into());
    let producer =
        TypedProducer::emit(&plan, &program, ScalarType::F64, inputs, Some(&[0])).unwrap();
    assert!(producer.body.is_empty());
    assert_eq!(
        producer.expression(0).unwrap(),
        "input0.data[broadcast_index]"
    );
    assert!(TypedProducer::emit(&plan, &program, ScalarType::F32, leaves(), Some(&[0])).is_err());
}

#[test]
fn canonical_helper_is_shared_by_full_and_sliced_emission() {
    let plan = plan(
        vec![FusionOp::Builtin {
            origin: Default::default(),
            name: "round".into(),
            inputs: vec![0],
            output: Some(2),
        }],
        2,
    );
    let program = resolve(
        &plan,
        &[Value::Num(2.5), Value::Num(0.0)],
        ProviderPrecision::F64,
    );
    let full = TypedProducer::emit(&plan, &program, ScalarType::F64, leaves(), None).unwrap();
    let sliced =
        TypedProducer::emit(&plan, &program, ScalarType::F64, leaves(), Some(&[2])).unwrap();
    assert_eq!(full.body, sliced.body);
    assert_eq!(
        full.preamble(&program).unwrap(),
        sliced.preamble(&program).unwrap()
    );
    let expected = crate::kernel_math::rounding::functions(ProviderPrecision::F64);
    assert_eq!(
        full.preamble(&program).unwrap(),
        format!("// {}\n{expected}\n", program.cache_identity().unwrap())
    );
    assert!(full.body.contains("rm_round_f64"));
}

#[test]
fn malformed_slice_dependencies_fail_closed() {
    let inputs = [0, 1];
    let ordered =
        producer_indices(&plan(vec![add(0, 1, 2), add(2, 1, 3)], 3), &inputs, &[3]).unwrap();
    assert_eq!(ordered.iter().collect::<Vec<_>>(), vec![0, 1]);
    assert!(producer_indices(&plan(vec![add(0, 1, 2)], 2), &inputs, &[99]).is_err());
    assert!(producer_indices(&plan(vec![add(0, 99, 2)], 2), &inputs, &[2]).is_err());
    assert!(producer_indices(&plan(vec![add(0, 1, 2), add(1, 0, 2)], 2), &inputs, &[2]).is_err());
    assert!(producer_indices(&plan(vec![add(0, 1, 0)], 0), &inputs, &[0]).is_err());
    assert!(producer_indices(&plan(vec![add(3, 1, 2), add(2, 0, 3)], 3), &inputs, &[3]).is_err());
    assert!(producer_indices(&plan(vec![add(2, 1, 2)], 2), &inputs, &[2]).is_err());
}

#[test]
fn complete_emission_does_not_hide_errors_in_unrequested_operations() {
    let valid = plan(vec![add(0, 1, 2)], 2);
    let program = resolve(
        &valid,
        &[Value::Num(1.0), Value::Num(2.0)],
        ProviderPrecision::F64,
    );
    let malformed = plan(vec![add(0, 1, 2), add(99, 0, 3)], 2);
    // Full mode must still reject the later operation, even though only 2
    // would be written. It must not silently enable dead-code elimination.
    assert!(TypedProducer::emit(&malformed, &program, ScalarType::F64, leaves(), None).is_err());
}
