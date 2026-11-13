//! Forced generic-native calls must use the catalog arity binding, including
//! the requested outputs of the builtin call rather than its enclosing function.

use super::*;

const SOURCE: &str = r#"
function y = nativeArityOk(x)
  narginchk(1, 1);
  nargoutchk(1, 1);
  y = x;
end
function y = nativeArityBadInput(x)
  check = @narginchk;
  y = check(1, 1);
end
function y = nativeArityBadOutput(x)
  check = @nargoutchk;
  y = check(1, 1);
end
function [a, b] = nativeArityBadLegacy()
  check = @nargoutchk;
  [a, b] = check(0, 1, 2);
end
"#;

fn call(
    function: &str,
    arguments: Vec<runmat_value::Value>,
    requested_outputs: usize,
) -> ProcedureInvocation {
    ProcedureInvocation {
        target: ProcedureTarget::Function(function.into()),
        arguments: crate::ProcedureArguments::Source(arguments),
        requested_outputs,
    }
}

#[test]
fn forced_generic_native_arity_forms_keep_exact_builtin_output_contracts() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).expect("session");
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new("native-arity@1", "nativeArityOk.m", SOURCE),
            None,
        ))
        .expect("compile arity functions");
        let control = InvocationControl::default().force_generic_native();
        let value = block_on(session.invoke_executable(
            &unit,
            call("nativeArityOk", vec![runmat_value::Value::Num(7.0)], 1),
            &control,
        ))
        .expect("discarded zero-output checks");
        assert_eq!(value.as_slice(), &[runmat_value::Value::Num(7.0)]);

        for (name, args, outputs) in [
            (
                "nativeArityBadInput",
                vec![runmat_value::Value::Num(1.0)],
                1,
            ),
            (
                "nativeArityBadOutput",
                vec![runmat_value::Value::Num(1.0)],
                1,
            ),
            ("nativeArityBadLegacy", Vec::new(), 2),
        ] {
            let error =
                block_on(session.invoke_executable(&unit, call(name, args, outputs), &control))
                    .expect_err("builtin form must reject extra result");
            let RunError::Runtime(runtime_error) = error else {
                panic!("{name}: expected a runtime output error, got {error:?}");
            };
            assert_eq!(
                runtime_error.identifier(),
                Some("RunMat:TooManyOutputs"),
                "{name}: {}",
                runtime_error.message(),
            );
        }
    });
}
