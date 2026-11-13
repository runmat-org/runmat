//! Public interpreted and forced-generic-native max calls share the catalog binding.

use super::execute_text_request;
use crate::{
    ExecutableSource, InvocationControl, ProcedureInvocation, ProcedureTarget, RunError,
    RunMatSession,
};
use futures::executor::block_on;
use runmat_builtins::{BuiltinExampleHarness, BuiltinExampleVerification, MAX_CATALOG_ENTRY};
use runmat_value::{ComplexTensor, IntValue, IntegerStorage, Tensor, Value};

const SOURCE: &str = r#"
function [m, i, p] = nativeMaxForms(A, B)
  [m, i] = max(A, [], 1);
  p = max(A, B);
end
function y = nativeMaxInvalid(A)
  y = max(A, [], 0);
end
"#;

fn invocation(name: &str, arguments: Vec<Value>, outputs: usize) -> ProcedureInvocation {
    ProcedureInvocation {
        target: ProcedureTarget::Function(name.into()),
        arguments: crate::ProcedureArguments::Source(arguments),
        requested_outputs: outputs,
    }
}

#[test]
fn public_interpreter_and_forced_native_max_forms_agree_on_values_and_error_owner() {
    runmat_gc::gc_test_context(|| {
        let wide = 9_007_199_254_740_992u64;
        let cases = vec![
            (
                "double",
                Value::Tensor(Tensor::new(vec![3.0, 1.0, 4.0, 2.0], vec![2, 2]).unwrap()),
                Value::Num(2.5),
                [
                    Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![1, 2]).unwrap()),
                    Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
                    Value::Tensor(Tensor::new(vec![3.0, 2.5, 4.0, 2.5], vec![2, 2]).unwrap()),
                ],
            ),
            (
                "exact uint64",
                Value::Tensor(
                    Tensor::new_integer(
                        IntegerStorage::U64(vec![wide, wide + 1, 4, 3]),
                        vec![2, 2],
                    )
                    .unwrap(),
                ),
                Value::Int(IntValue::U64(wide)),
                [
                    Value::Tensor(
                        Tensor::new_integer(IntegerStorage::U64(vec![wide + 1, 4]), vec![1, 2])
                            .unwrap(),
                    ),
                    Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap()),
                    Value::Tensor(
                        Tensor::new_integer(
                            IntegerStorage::U64(vec![wide, wide + 1, wide, wide]),
                            vec![2, 2],
                        )
                        .unwrap(),
                    ),
                ],
            ),
            (
                "single broadcast",
                Value::Tensor(Tensor::from_f32(vec![3.0, 1.0], vec![2, 1]).unwrap()),
                Value::Tensor(Tensor::from_f32(vec![2.0, 4.0], vec![1, 2]).unwrap()),
                [
                    Value::Tensor(Tensor::from_f32(vec![3.0], vec![1, 1]).unwrap()),
                    Value::Num(1.0),
                    Value::Tensor(Tensor::from_f32(vec![3.0, 2.0, 4.0, 4.0], vec![2, 2]).unwrap()),
                ],
            ),
            (
                "complex",
                Value::ComplexTensor(
                    ComplexTensor::new(vec![(1.0, 2.0), (3.0, 0.0)], vec![2, 1]).unwrap(),
                ),
                Value::Complex(0.0, 5.0),
                [
                    Value::Complex(3.0, 0.0),
                    Value::Num(2.0),
                    Value::ComplexTensor(
                        ComplexTensor::new(vec![(0.0, 5.0), (0.0, 5.0)], vec![2, 1]).unwrap(),
                    ),
                ],
            ),
        ];
        for native in [false, true] {
            let mut session = RunMatSession::with_options(false, false).expect("session");
            let unit = block_on(session.compile_executable_unit(
                ExecutableSource::new("native-max-forms@1", "nativeMaxForms.m", SOURCE),
                None,
            ))
            .expect("compile max forms");
            let control = if native {
                InvocationControl::default().force_generic_native()
            } else {
                InvocationControl::default()
            };
            for (label, input, other, expected) in &cases {
                let actual = block_on(session.invoke_executable(
                    &unit,
                    invocation("nativeMaxForms", vec![input.clone(), other.clone()], 3),
                    &control,
                ))
                .unwrap_or_else(|error| panic!("native={native}, {label}: {error:?}"));
                assert_eq!(
                    actual.as_slice(),
                    expected.as_slice(),
                    "native={native}, {label}"
                );
            }

            let error = block_on(session.invoke_executable(
                &unit,
                invocation("nativeMaxInvalid", vec![cases[0].1.clone()], 1),
                &control,
            ))
            .expect_err("invalid max dimension");
            let RunError::Runtime(error) = error else {
                panic!("native={native}: expected RuntimeError, got {error:?}");
            };
            assert_eq!(error.identifier(), Some("RunMat:max:InvalidArgument"));
        }
    });
}

#[test]
fn all_six_max_catalog_examples_execute_with_their_assertions() {
    runmat_gc::gc_test_context(|| {
        let examples = MAX_CATALOG_ENTRY.documentation.examples;
        assert_eq!(examples.len(), 6);
        for example in examples {
            assert_eq!(
                example.harness,
                BuiltinExampleHarness::Portable,
                "{}",
                example.id
            );
            let BuiltinExampleVerification::Assertions { source } = example.verification else {
                panic!("{}: missing executable assertions", example.id);
            };
            let mut session = RunMatSession::with_options(false, false).expect("session");
            let program = format!("{}\n{}", example.program, source);
            let outcome = execute_text_request(&mut session, &program)
                .unwrap_or_else(|error| panic!("{}: {error:?}", example.id));
            assert!(
                !outcome.diagnostics.iter().any(|diagnostic| {
                    diagnostic.severity == crate::abi::DiagnosticSeverity::Error
                }),
                "{}: {:?}",
                example.id,
                outcome.diagnostics
            );
        }
    });
}
