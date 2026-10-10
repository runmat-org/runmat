use super::*;
use runmat_value::Tensor;

fn binary(op: PrimitiveOp) -> FusionGroupPlan {
    plan(
        vec![FusionOp::Primitive {
            op,
            inputs: vec![0, 1],
            output: Some(2),
        }],
        2,
    )
}
fn named_plus() -> FusionGroupPlan {
    plan(
        vec![FusionOp::Builtin {
            origin: crate::graph::FusionCallOrigin::named("plus"),
            name: "plus".into(),
            inputs: vec![0, 1],
            output: Some(2),
        }],
        2,
    )
}
fn tensor(single: bool, data: &[f64], shape: &[usize]) -> Value {
    Value::Tensor(if single {
        Tensor::from_f32(data.iter().map(|&x| x as f32).collect(), shape.to_vec()).unwrap()
    } else {
        Tensor::new(data.to_vec(), shape.to_vec()).unwrap()
    })
}
fn host(name: &str, inputs: &[Value]) -> Value {
    let result = futures::executor::block_on(runmat_runtime::call_builtin_async_with_outputs(
        name, inputs, 1,
    ))
    .unwrap();
    match result {
        Value::OutputList(mut values) => {
            assert_eq!(values.len(), 1);
            values.remove(0)
        }
        value => value,
    }
}
fn host_values(value: Value) -> Vec<f64> {
    match value {
        Value::Num(value) => vec![value],
        Value::Tensor(value) => value.materialize_f64(),
        value => panic!("unexpected numerical carrier {value:?}"),
    }
}

#[test]
fn mixed_add_uses_double_sum_then_single_result_in_checked_and_legacy_emission() {
    runmat_gc::gc_test_context(|| {
        let below_one = f64::from(f32::from_bits(0x3f7f_ffff));
        let half_ulp = f64::from(f32::EPSILON) / 4.0;
        for plan in [binary(PrimitiveOp::Add), named_plus()] {
            for scalar in [half_ulp - f64::EPSILON, half_ulp + f64::EPSILON] {
                for reverse in [false, true] {
                    let single = tensor(true, &[below_one, below_one], &[1, 2]);
                    let double = Value::Num(scalar);
                    let inputs = if reverse {
                        [double, single]
                    } else {
                        [single, double]
                    };
                    let program = resolve(&plan, &inputs, ProviderPrecision::F64);
                    assert_eq!(program.floating_class(2).unwrap(), NumericClass::Single);
                    let checked =
                        TypedProducer::emit_checked_program(&plan, &program, ScalarType::F64, 2)
                            .unwrap()
                            .unwrap();
                    assert!(matches!(
                        checked.wire().nodes.last(),
                        Some(runmat_accelerate_api::scalar_program::ScalarNode::Apply {
                            operation: runmat_types::ScalarOperation::MixedRealAddToSingle,
                            ..
                        })
                    ));
                    let selected =
                        TypedProducer::emit(&plan, &program, ScalarType::F64, leaves(), Some(&[2]))
                            .unwrap();
                    let legacy =
                        TypedProducer::emit(&plan, &program, ScalarType::F64, leaves(), None)
                            .unwrap();
                    assert_eq!(selected.body, legacy.body);
                    assert!(selected
                        .body
                        .contains("f32((f64(input0.data[i0]) + f64(input1.data[i1])))"));
                    assert!(
                        TypedProducer::emit(&plan, &program, ScalarType::F32, leaves(), None)
                            .is_err()
                    );
                    let expected = f64::from((below_one + scalar) as f32);
                    assert_eq!(host_values(host("plus", &inputs)), vec![expected; 2]);
                }
            }
        }
    });
}

#[test]
fn mixed_add_broadcast_array_is_typed_but_incompatible_shape_is_not() {
    let single = tensor(true, &[0.25, 0.5], &[2, 1]);
    let double = tensor(false, &[0.5, 0.25, 0.125], &[1, 3]);
    let bad = tensor(false, &[0.5, 0.25, 0.125], &[3, 1]);
    for plan in [binary(PrimitiveOp::Add), named_plus()] {
        let program = resolve(
            &plan,
            &[single.clone(), double.clone()],
            ProviderPrecision::F64,
        );
        assert_eq!(program.floating_class(2).unwrap(), NumericClass::Single);
        assert_eq!(
            program.fact(2).unwrap().shape,
            runmat_types::ShapeFact::from(vec![Some(2), Some(3)])
        );
        assert!(
            futures::executor::block_on(ResolvedProgram::resolve_with_policy(
                &plan,
                &[single.clone(), bad.clone()],
                ProviderPrecision::F64,
                PrecisionPolicy::Exact,
            ))
            .is_err()
        );
    }
}

#[test]
fn left_division_emits_rhs_over_lhs_for_scalars_and_broadcast_arrays() {
    runmat_gc::gc_test_context(|| {
        let plan = binary(PrimitiveOp::ElemLeftDiv);
        let expressions = HashMap::from([(0, "left".into()), (1, "right".into())]);
        assert_eq!(
            crate::fusion::primitive_expr(PrimitiveOp::ElemLeftDiv, &[0, 1], &expressions).unwrap(),
            "(right / left)"
        );
        assert_eq!(
            crate::fusion::primitive_expr(PrimitiveOp::ElemDiv, &[0, 1], &expressions).unwrap(),
            "(left / right)"
        );
        for single in [false, true] {
            let cases = [
                (
                    [
                        tensor(single, &[2.0], &[1, 1]),
                        tensor(single, &[8., 16.], &[1, 2]),
                    ],
                    vec![4., 8.],
                ),
                (
                    [
                        tensor(single, &[2., 4.], &[2, 1]),
                        tensor(single, &[32.0], &[1, 1]),
                    ],
                    vec![16., 8.],
                ),
                (
                    [
                        tensor(single, &[2., 4.], &[2, 1]),
                        tensor(single, &[8., 16., 32.], &[1, 3]),
                    ],
                    vec![4., 2., 8., 4., 16., 8.],
                ),
            ];
            for (inputs, expected) in cases {
                let program = resolve(&plan, &inputs, ProviderPrecision::F64);
                let producer =
                    TypedProducer::emit(&plan, &program, ScalarType::F64, leaves(), None).unwrap();
                let ty = if single { "f32" } else { "f64" };
                assert!(producer
                    .body
                    .contains(&format!("({ty}(input1.data[i1]) / {ty}(input0.data[i0]))")));
                assert_eq!(host_values(host("ldivide", &inputs)), expected);
            }
        }
    });
}

#[test]
fn mixed_class_left_division_keeps_unknown_admission_and_host_semantics() {
    runmat_gc::gc_test_context(|| {
        let plan = binary(PrimitiveOp::ElemLeftDiv);
        for (inputs, expected) in [
            (
                [Value::Num(2.0), tensor(true, &[8., 16.], &[1, 2])],
                vec![4., 8.],
            ),
            (
                [tensor(true, &[2., 4.], &[2, 1]), Value::Num(32.0)],
                vec![16., 8.],
            ),
        ] {
            let program = resolve(&plan, &inputs, ProviderPrecision::F64);
            // The shared operator owner does not yet prove mixed-class
            // promotion. Preserve that fact rather than inventing a class in
            // the emitter to make this fixture compile a kernel.
            assert!(matches!(
                program.fact(2).unwrap().kind,
                runmat_types::ValueKindFact::Unknown
            ));
            assert!(TypedProducer::emit(&plan, &program, ScalarType::F64, leaves(), None).is_err());
            let actual = host("ldivide", &inputs);
            assert_eq!(
                runmat_runtime::value_fact::value_fact(&actual)
                    .numeric()
                    .unwrap()
                    .class,
                runmat_types::NumericClass::Single
            );
            assert_eq!(host_values(actual), expected);
        }
    });
}

#[test]
fn typed_matrix_multiply_retains_scalar_forms_but_not_contractions() {
    runmat_gc::gc_test_context(|| {
        let matrix = tensor(false, &[1., 2., 3., 4.], &[2, 2]);
        for inputs in [
            [Value::Num(2.), matrix.clone()],
            [matrix.clone(), Value::Num(2.)],
        ] {
            let plan = binary(PrimitiveOp::Mul);
            let program = resolve(&plan, &inputs, ProviderPrecision::F64);
            assert!(TypedProducer::emit(&plan, &program, ScalarType::F64, leaves(), None).is_ok());
            assert_eq!(host_values(host("mtimes", &inputs)), vec![2., 4., 6., 8.]);
        }
        let outer = [
            tensor(false, &[-0., 2.], &[2, 1]),
            tensor(false, &[1., 3.], &[1, 2]),
        ];
        for inputs in [[matrix.clone(), matrix], outer.clone()] {
            assert!(
                futures::executor::block_on(ResolvedProgram::resolve_with_policy(
                    &binary(PrimitiveOp::Mul),
                    &inputs,
                    ProviderPrecision::F64,
                    PrecisionPolicy::Exact,
                ))
                .is_err()
            );
        }
        let elementwise = binary(PrimitiveOp::ElemMul);
        let program = resolve(&elementwise, &outer, ProviderPrecision::F64);
        assert!(
            TypedProducer::emit(&elementwise, &program, ScalarType::F64, leaves(), None).is_ok()
        );
        // A one-term matrix sum starts at +0; a scalar product does not.
        let product = host_values(host("mtimes", &outer));
        let elementwise = host_values(host("times", &outer));
        assert_eq!(product, vec![0., 2., 0., 6.]);
        assert!(product[0].is_sign_positive());
        assert!(elementwise[0].is_sign_negative());
    });
}
