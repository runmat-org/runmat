//! Canonical declaration identity, checked wire and legacy expression agree.
use super::*;
use runmat_accelerate_api::scalar_program::ScalarNode;
use runmat_types::{ScalarOperation, ScalarUnaryMath};

#[test]
fn canonical_unary_instructions_preserve_precision_and_existing_domain_owners() {
    runmat_gc::gc_test_context(|| {
        for (name, instruction, value) in [
            ("exp", ScalarUnaryMath::Exp, 0.5),
            ("sin", ScalarUnaryMath::Sin, 0.5),
            ("cos", ScalarUnaryMath::Cos, 0.5),
            ("tan", ScalarUnaryMath::Tan, 0.5),
            ("sinh", ScalarUnaryMath::Sinh, 0.5),
            ("cosh", ScalarUnaryMath::Cosh, 0.5),
            ("tanh", ScalarUnaryMath::Tanh, 0.5),
            ("atan", ScalarUnaryMath::Atan, 0.5),
            ("asinh", ScalarUnaryMath::Asinh, 0.5),
            ("acosh", ScalarUnaryMath::Acosh, 2.0),
            ("atanh", ScalarUnaryMath::Atanh, 0.5),
        ] {
            let mut source = plan(
                vec![FusionOp::Builtin {
                    name: name.into(),
                    origin: crate::graph::FusionCallOrigin::named(name),
                    inputs: vec![0],
                    output: Some(2),
                }],
                2,
            );
            source.inputs = vec![0];
            for single in [false, true] {
                let (input, precision, scalar, class) = if single {
                    (
                        Value::Tensor(
                            runmat_value::Tensor::from_f32(vec![value as f32], vec![1, 1]).unwrap(),
                        ),
                        ProviderPrecision::F32,
                        ScalarType::F32,
                        NumericClass::Single,
                    )
                } else {
                    (
                        Value::Num(value),
                        ProviderPrecision::F64,
                        ScalarType::F64,
                        NumericClass::Double,
                    )
                };
                let program = resolve(&source, &[input], precision);
                assert_eq!(program.floating_class(2).unwrap(), class);
                let declaration = program.declaration(2).unwrap().semantics.unwrap();
                assert_eq!(
                    declaration.scalar_operation,
                    Some(ScalarOperation::UnaryMath(instruction))
                );
                if matches!(instruction, ScalarUnaryMath::Acosh | ScalarUnaryMath::Atanh) {
                    assert!(
                        declaration.admit.is_some() && declaration.required_real_unary.is_some()
                    );
                }
                let expressions = HashMap::from([(0, "input0.data[i0]".to_owned())]);
                let legacy =
                    TypedProducer::emit(&source, &program, scalar, expressions.clone(), None)
                        .unwrap();
                let checked =
                    TypedProducer::emit(&source, &program, scalar, expressions, Some(&[2]))
                        .unwrap();
                assert_eq!(legacy.body, checked.body);
                assert!(
                    matches!(checked.checked_program().unwrap().wire().nodes.last(), Some(ScalarNode::Apply { operation: ScalarOperation::UnaryMath(actual), .. }) if *actual == instruction)
                );
                let shader = source.prepare_typed_shader(&[2], &program, scalar).unwrap();
                assert!(shader.source().contains("typed0"));
                if !single {
                    assert!(
                        source
                            .prepare_typed_shader(&[2], &program, ScalarType::F32)
                            .is_err(),
                        "a typed unary identity cannot narrow original Double data"
                    );
                }
                #[cfg(feature = "wgpu")]
                crate::wgsl::validation_tests::validate_generated(
                    "canonical-unary-math",
                    shader.source(),
                    64,
                    8,
                    if single {
                        crate::wgsl::validation_tests::RequiredCaps::None
                    } else {
                        crate::wgsl::validation_tests::RequiredCaps::Float64
                    },
                );
            }
        }
        for name in ["expm1", "log", "log10", "log1p", "log2", "asin", "acos"] {
            let binding = runmat_builtins::BuiltinBindingIdentity {
                builtin: runmat_builtins::builtin_catalog_entry_by_name(name)
                    .unwrap()
                    .identity,
                variant: "default",
            };
            let declaration =
                runmat_runtime::builtins::common::spec::canonical_fusion_declaration(binding)
                    .unwrap();
            assert!(
                declaration.spec.elementwise.is_none(),
                "disabled family must not gain a guessed unary recipe"
            );
            assert!(declaration
                .semantics
                .and_then(|semantics| semantics.scalar_operation)
                .is_none());
        }
    });
}
