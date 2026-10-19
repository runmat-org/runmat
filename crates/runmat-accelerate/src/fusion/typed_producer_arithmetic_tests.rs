//! The original host operation, not an output-class cast, is the oracle.
use super::*;
use runmat_types::OperatorKind;

fn named(name: &str) -> FusionGroupPlan {
    plan(
        vec![FusionOp::Builtin {
            origin: crate::graph::FusionCallOrigin::named(name),
            name: name.into(),
            inputs: vec![0, 1],
            output: Some(2),
        }],
        2,
    )
}

#[test]
fn represented_binary_arithmetic_matches_host_without_early_narrowing() {
    runmat_gc::gc_test_context(|| {
        let single = 1.0 + f64::from(f32::EPSILON);
        let double = 1.0 + f64::from(f32::EPSILON) / 2.0;
        for (primitive, name, operator) in [
            (PrimitiveOp::Add, "plus", OperatorKind::Add),
            (PrimitiveOp::Sub, "minus", OperatorKind::Subtract),
            (
                PrimitiveOp::ElemMul,
                "times",
                OperatorKind::ElementwiseMultiply,
            ),
            (
                PrimitiveOp::ElemDiv,
                "rdivide",
                OperatorKind::ElementwiseDivide,
            ),
            (
                PrimitiveOp::ElemLeftDiv,
                "ldivide",
                OperatorKind::ElementwiseLeftDivide,
            ),
        ] {
            for source in [binary(primitive), named(name)] {
                for reverse in [false, true] {
                    let inputs = if reverse {
                        [
                            tensor(false, &[double; 2], &[1, 2]),
                            tensor(true, &[single; 2], &[1, 2]),
                        ]
                    } else {
                        [
                            tensor(true, &[single; 2], &[1, 2]),
                            tensor(false, &[double; 2], &[1, 2]),
                        ]
                    };
                    let (a, b) = if reverse {
                        (double, single)
                    } else {
                        (single, double)
                    };
                    let expected = f64::from(match operator {
                        OperatorKind::Add => (a + b) as f32,
                        OperatorKind::Subtract => (a - b) as f32,
                        OperatorKind::ElementwiseMultiply => (a * b) as f32,
                        OperatorKind::ElementwiseDivide => (a / b) as f32,
                        OperatorKind::ElementwiseLeftDivide => (b / a) as f32,
                        _ => unreachable!(),
                    });
                    assert_eq!(host_values(host(name, &inputs)), vec![expected; 2]);
                    if operator == OperatorKind::ElementwiseMultiply {
                        assert_ne!(expected, f64::from((a as f32) * (b as f32)));
                    }
                    let program = resolve(&source, &inputs, ProviderPrecision::F64);
                    // A mixed operation widens one operand and rounds its
                    // result. Both placement candidates consume that recipe.
                    assert_eq!(program.scalar_work_units(&source, 2).unwrap(), 6);
                    assert_eq!(program.host_scalar_work_units(&source).unwrap(), 6);
                    let checked =
                        TypedProducer::emit_checked_program(&source, &program, ScalarType::F64, 2)
                            .unwrap()
                            .unwrap();
                    let runmat_accelerate_api::scalar_program::ScalarNode::Apply {
                        operation: runmat_types::ScalarOperation::RealArithmetic(recipe),
                        ..
                    } = checked.wire().nodes.last().unwrap()
                    else {
                        panic!("represented arithmetic required")
                    };
                    assert_eq!(recipe.operator(), operator);
                    assert_eq!(recipe.computation_class(), NumericClass::Double);
                    assert_eq!(recipe.result_class(), NumericClass::Single);
                    for roots in [Some(&[2][..]), None] {
                        let emitted = TypedProducer::emit(
                            &source,
                            &program,
                            ScalarType::F64,
                            leaves(),
                            roots,
                        )
                        .unwrap();
                        assert!(emitted.body.contains("let typed0: f32 = f32("));
                        assert!(emitted.body.contains("f64(input0.data[i0])"));
                        assert!(emitted.body.contains("f64(input1.data[i1])"));
                        assert!(TypedProducer::emit(
                            &source,
                            &program,
                            ScalarType::F32,
                            leaves(),
                            roots
                        )
                        .is_err());
                        #[cfg(feature = "wgpu")]
                        crate::wgsl::validation_tests::validate_generated(
                            "represented-real-arithmetic",
                            &source
                                .generate_typed_wgsl(&[2], &program, ScalarType::F64)
                                .unwrap(),
                            64,
                            8,
                            crate::wgsl::validation_tests::RequiredCaps::Float64,
                        );
                    }
                    let opted = futures::executor::block_on(ResolvedProgram::resolve_with_policy(
                        &source,
                        &inputs,
                        ProviderPrecision::F32,
                        PrecisionPolicy::AllowDoubleToSingle,
                    ))
                    .unwrap();
                    assert_eq!(opted.scalar_work_units(&source, 2).unwrap(), 2);
                    assert_eq!(opted.host_scalar_work_units(&source).unwrap(), 2);
                    let projected =
                        TypedProducer::emit_checked_program(&source, &opted, ScalarType::F32, 2)
                            .unwrap()
                            .unwrap();
                    assert_ne!(checked.identity(), projected.identity());
                    let runmat_accelerate_api::scalar_program::ScalarNode::Apply {
                        operation: runmat_types::ScalarOperation::RealArithmetic(recipe),
                        ..
                    } = projected.wire().nodes.last().unwrap()
                    else {
                        panic!("projected recipe required")
                    };
                    assert_eq!(recipe.computation_class(), NumericClass::Single);
                    assert!(
                        !TypedProducer::emit(&source, &opted, ScalarType::F32, leaves(), None)
                            .unwrap()
                            .body
                            .contains("f64(")
                    );
                }
            }
        }
    });
}

#[test]
fn single_product_intermediate_is_rounded_before_following_double_arithmetic() {
    runmat_gc::gc_test_context(|| {
        let source = plan(
            vec![
                FusionOp::Primitive {
                    op: PrimitiveOp::ElemMul,
                    inputs: vec![0, 1],
                    output: Some(2),
                },
                FusionOp::Primitive {
                    op: PrimitiveOp::Sub,
                    inputs: vec![2, 1],
                    output: Some(3),
                },
            ],
            3,
        );
        let inputs = [
            tensor(true, &[1.0 + f64::from(f32::EPSILON); 2], &[1, 2]),
            tensor(false, &[1.0 + f64::from(f32::EPSILON) / 2.0; 2], &[1, 2]),
        ];
        let program = resolve(&source, &inputs, ProviderPrecision::F64);
        for roots in [Some(&[3][..]), None, Some(&[2, 3][..])] {
            let emitted =
                TypedProducer::emit(&source, &program, ScalarType::F64, leaves(), roots).unwrap();
            assert!(emitted.body.contains("let typed0: f32 = f32("));
            assert!(emitted
                .body
                .contains("let typed1: f32 = f32((f64(typed0) - "));
        }
    });
}
