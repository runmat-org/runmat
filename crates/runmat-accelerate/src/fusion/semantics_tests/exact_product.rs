//! Original scalar words, scalar geometry and physical word carrier agree.
use super::*;
use runmat_types::OperatorKind;
use runmat_value::Tensor;

fn source(named: Option<&str>, reverse: bool) -> FusionGroupPlan {
    let inputs = if reverse { vec![1, 0] } else { vec![0, 1] };
    plan(
        vec![match named {
            Some(name) => FusionOp::Builtin {
                name: name.into(),
                origin: crate::graph::FusionCallOrigin::named(name),
                inputs,
                output: Some(2),
            },
            None => FusionOp::Primitive {
                op: PrimitiveOp::Mul,
                inputs,
                output: Some(2),
            },
        }],
        2,
    )
}

#[test]
fn scalar_multiply_uses_exact_words_without_changing_double_semantics() {
    for name in [None, Some("times"), Some("mtimes")] {
        for reverse in [false, true] {
            let source = source(name, reverse);
            let data = Value::Tensor(
                Tensor::from_f32(vec![f32::from_bits(1), -0.0, f32::MAX], vec![1, 3]).unwrap(),
            );
            let inputs = [data, Value::Num(4.0)];
            let resolved = resolve_policy(
                &source,
                &inputs,
                ProviderPrecision::F32,
                PrecisionPolicy::Exact,
            )
            .unwrap();
            assert_eq!(resolved.floating_class(1).unwrap(), NumericClass::Double);
            let receipt = resolved.exact_product(2).unwrap();
            assert_eq!((receipt.base, receipt.control), (0, 1));
            assert_eq!(receipt.exact.original_double_bits(), 4.0_f64.to_bits());
            resolved
                .require_exact_product_controls(&source, &inputs)
                .unwrap();
            assert!(resolved
                .require_exact_product_controls(&source, &[inputs[0].clone(), Value::Num(8.0)])
                .is_err());
            assert!(resolved
                .embedded_scalar_controls(&source, &[1, 2])
                .is_empty());
            assert!(source
                .prepare_typed_shader(&[1, 2], &resolved, ScalarType::F32)
                .is_err());
            let shader = source
                .prepare_typed_shader(&[2], &resolved, ScalarType::F32)
                .unwrap();
            assert_eq!(shader.input_indices(), &[0]);
            assert!(shader.uses_exact_product_words());
            assert_eq!(resolved.host_scalar_work_units(&source).unwrap(), 9);
            assert_eq!(
                resolved
                    .physical_scalar_work_units(&source, 3, shader.implementations())
                    .unwrap(),
                3 * crate::kernel_math::product_exact::work_prior().algorithmic_units()
            );
            assert!(shader.source().contains("struct Tensor { data: array<u32>"));
            assert!(shader
                .source()
                .contains("rm_exact_product_bits(input0.data[i0], 0x40800000u)"));
            assert!(shader.source().contains("output0.data[g] = typed0_bits;"));
            assert!(!shader.source().contains("array<f64>"));
            let checked = crate::fusion::typed_producer::TypedProducer::emit_checked_program(
                &source,
                &resolved,
                ScalarType::F32,
                2,
            )
            .unwrap()
            .unwrap();
            assert!(
                matches!(checked.wire().nodes.last(), Some(runmat_accelerate_api::scalar_program::ScalarNode::Apply {
                operation: runmat_types::ScalarOperation::RealArithmetic(recipe), ..
            }) if recipe.operator() == OperatorKind::ElementwiseMultiply && recipe.computation_class() == NumericClass::Double)
            );
            #[cfg(feature = "wgpu")]
            crate::wgsl::validation_tests::validate_generated(
                "exact-scalar-product",
                shader.source(),
                64,
                8,
                crate::wgsl::validation_tests::RequiredCaps::None,
            );
        }
    }
}

#[test]
fn nested_single_intrinsics_retain_the_exact_product_geometry() {
    let source = plan(
        vec![
            FusionOp::Primitive {
                op: PrimitiveOp::UPlus,
                inputs: vec![0],
                output: Some(2),
            },
            FusionOp::Primitive {
                op: PrimitiveOp::Neg,
                inputs: vec![2],
                output: Some(3),
            },
            FusionOp::Primitive {
                op: PrimitiveOp::Mul,
                inputs: vec![1, 3],
                output: Some(4),
            },
        ],
        4,
    );
    let inputs = [
        Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        Value::Num(2.0),
    ];
    let resolved = resolve_policy(
        &source,
        &inputs,
        ProviderPrecision::F32,
        PrecisionPolicy::Exact,
    )
    .unwrap();
    assert!(resolved.exact_product(4).is_some());

    let bound = source
        .measured_typed_shader_preparation_bytes(&[4], &resolved, ScalarType::F32, 1)
        .expect("exact product geometry must not measure its unreachable Double expression");
    let shader = source
        .prepare_typed_shader_with_inputs_bounded(
            &[4],
            &resolved,
            ScalarType::F32,
            &[0],
            Some(bound.body_bytes),
        )
        .unwrap();
    assert!(shader.uses_exact_product_words());
    assert!(shader
        .source()
        .contains("rm_exact_product_bits(typed1_bits, 0x40000000u)"));
}

#[test]
fn emitted_work_tracks_actual_slice_not_all_semantic_receipts() {
    let mut source = source(None, false);
    source.operations.push(FusionOp::Primitive {
        op: PrimitiveOp::Mul,
        inputs: vec![1, 0],
        output: Some(3),
    });
    source.output = Some(3);
    let inputs = [
        Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap()),
        Value::Num(4.0),
    ];
    let resolved = resolve_policy(
        &source,
        &inputs,
        ProviderPrecision::F32,
        PrecisionPolicy::Exact,
    )
    .unwrap();
    assert!(resolved.exact_product(2).is_some() && resolved.exact_product(3).is_some());
    for (roots, expected) in [(Some(&[2][..]), vec![2]), (None, vec![2, 3])] {
        let producer = crate::fusion::typed_producer::TypedProducer::emit_with_words(
            &source,
            &resolved,
            ScalarType::F32,
            HashMap::from([
                (0, "bitcast<f32>(input0.data[i0])".to_owned()),
                (1, String::new()),
            ]),
            roots,
            HashMap::from([(0, "input0.data[i0]".to_owned())]),
        )
        .unwrap();
        assert_eq!(
            producer
                .implementations()
                .iter()
                .map(|work| work.output)
                .collect::<Vec<_>>(),
            expected
        );
        if roots.is_none() {
            let mut emitted = producer.implementations().to_vec();
            let base = resolved
                .physical_scalar_work_units(&source, 1, &emitted)
                .unwrap();
            let boundary = crate::wgsl::ieee_words::binary32_widening_implementation(3);
            emitted.push(boundary);
            assert_eq!(
                resolved
                    .physical_scalar_work_units(&source, 1, &emitted)
                    .unwrap(),
                base + boundary.algorithmic_units_per_invocation
            );
            emitted.push(boundary);
            assert!(resolved
                .physical_scalar_work_units(&source, 1, &emitted)
                .is_err());
            emitted.pop();
            emitted.push(emitted[0]);
            assert!(resolved
                .physical_scalar_work_units(&source, 1, &emitted)
                .is_err());
        }
    }
}

#[test]
fn captured_single_literals_seed_words_without_invocation_buffers() {
    for computed in [false, true] {
        let mut source = source(None, false);
        source.inputs.clear();
        source.const_values.insert(
            0,
            Value::Tensor(Tensor::from_f32(vec![f32::from_bits(1)], vec![1, 1]).unwrap()),
        );
        source.const_values.insert(1, Value::Num(4.0));
        if computed {
            source.operations.insert(
                0,
                FusionOp::Primitive {
                    op: PrimitiveOp::Add,
                    inputs: vec![0, 0],
                    output: Some(8),
                },
            );
            let FusionOp::Primitive { inputs, .. } = &mut source.operations[1] else {
                unreachable!()
            };
            inputs[0] = 8;
        }
        let program =
            resolve_policy(&source, &[], ProviderPrecision::F32, PrecisionPolicy::Exact).unwrap();
        let shader = source
            .prepare_typed_shader(&[2], &program, ScalarType::F32)
            .unwrap();
        assert!(shader.input_indices().is_empty());
        assert!(shader
            .source()
            .contains("let control0_bits: u32 = 0x00000001u;"));
        assert!(shader.source().contains(if computed {
            "rm_exact_product_bits(typed0_bits,"
        } else {
            "rm_exact_product_bits(control0_bits,"
        }));
        assert_eq!(shader.implementations().len(), 1);
        #[cfg(feature = "wgpu")]
        crate::wgsl::validation_tests::validate_generated(
            "captured-word-product",
            shader.source(),
            64,
            8,
            crate::wgsl::validation_tests::RequiredCaps::None,
        );
    }
}

#[test]
fn scalar_product_projection_preserves_shape_and_never_omits_data_fanout() {
    for shape in [vec![0, 3], vec![2, 1, 3]] {
        let source = source(None, false);
        let inputs = [
            Value::Tensor(
                Tensor::from_f32(vec![1.0; shape.iter().product()], shape.clone()).unwrap(),
            ),
            Value::Tensor(Tensor::new(vec![4.0], vec![1, 1, 1]).unwrap()),
        ];
        let resolved = resolve_policy(
            &source,
            &inputs,
            ProviderPrecision::F32,
            PrecisionPolicy::Exact,
        )
        .unwrap();
        assert_eq!(
            resolved.fact(2).unwrap().shape.known_dims().unwrap(),
            shape.iter().copied().map(Some).collect::<Vec<_>>()
        );
        let shader = source
            .prepare_typed_shader(&[2], &resolved, ScalarType::F32)
            .unwrap();
        assert_eq!(shader.input_indices(), &[0]);
        assert!(source
            .prepare_typed_shader_with_inputs(&[2], &resolved, ScalarType::F32, &[])
            .is_err());
    }
    let mut source = source(None, false);
    source.operations.push(FusionOp::Primitive {
        op: PrimitiveOp::Add,
        inputs: vec![2, 1],
        output: Some(3),
    });
    source.output = Some(3);
    let inputs = [
        Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        Value::Num(4.0),
    ];
    let resolved = resolve_policy(
        &source,
        &inputs,
        ProviderPrecision::F32,
        PrecisionPolicy::Exact,
    )
    .unwrap();
    assert!(resolved.embedded_scalar_controls(&source, &[3]).is_empty());
    assert!(source
        .prepare_typed_shader_with_inputs(&[3], &resolved, ScalarType::F32, &[0])
        .is_err());
}

#[test]
fn actual_invocation_controls_win_and_general_double_data_requires_f64() {
    let mut source = source(None, false);
    source.const_values.insert(1, Value::Num(4.0));
    let data = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap());
    for control in [
        Value::Num(0.1),
        Value::Tensor(Tensor::new(vec![4.0, 4.0], vec![2, 1]).unwrap()),
    ] {
        let inputs = [data.clone(), control];
        let resolved = resolve_policy(
            &source,
            &inputs,
            ProviderPrecision::F32,
            PrecisionPolicy::Exact,
        );
        if matches!(inputs[1], Value::Tensor(_)) {
            assert!(matches!(resolved, Err(FusionResolveFailure::Structural(_))));
            continue;
        }
        let resolved = resolved.unwrap();
        assert!(resolved.exact_product(2).is_none());
        assert!(source
            .prepare_typed_shader(&[2], &resolved, ScalarType::F32)
            .is_err());
        // Array×array matrix multiplication is not an elementwise product.
        if matches!(inputs[1], Value::Num(_)) {
            let resolved = resolve(&source, &inputs).unwrap();
            let shader = source
                .prepare_typed_shader(&[2], &resolved, ScalarType::F64)
                .unwrap();
            assert!(!shader.uses_exact_product_words());
            assert!(shader.source().contains("f32((f64("));
        }
    }
    let first = resolve(&source, &[data.clone(), Value::Num(4.0)]).unwrap();
    let second = resolve(&source, &[data, Value::Num(8.0)]).unwrap();
    assert_ne!(
        first.cache_identity().unwrap(),
        second.cache_identity().unwrap()
    );
}
