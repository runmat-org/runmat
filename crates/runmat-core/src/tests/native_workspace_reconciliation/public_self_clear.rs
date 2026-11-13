use super::*;

#[derive(Clone, Copy, Debug)]
enum ExecutionMode {
    Interpreted,
    ForcedGenericNative,
}

#[test]
fn first_public_function_invocation_self_clear_does_not_resurrect_persistent() {
    runmat_gc::gc_test_context(|| {
        for mode in [
            ExecutionMode::Interpreted,
            ExecutionMode::ForcedGenericNative,
        ] {
            // Compile, then invoke in this fresh session. No interactive text
            // request primes its language services or function-clear owner.
            let mut session = RunMatSession::with_options(false, false).expect("fresh session");
            let source = crate::ExecutableSource::new(
                "public-self-clear-first@1",
                "selfClearFirst.m",
                "function y = selfClearFirst()\npersistent p\nif isempty(p)\np = 17;\nend\ny = p + 6;\nclear('selfClearFirst');\nend\n",
            );
            let unit = block_on(session.compile_executable_unit(source, None))
                .expect("compile first-session function");
            let invocation = crate::ProcedureInvocation {
                target: crate::ProcedureTarget::Function("selfClearFirst".into()),
                arguments: crate::ProcedureArguments::Source(Vec::new()),
                requested_outputs: 1,
            };
            let control = match mode {
                ExecutionMode::Interpreted => crate::InvocationControl::default(),
                ExecutionMode::ForcedGenericNative => {
                    crate::InvocationControl::default().force_generic_native()
                }
            };
            let native_before = session.generic_native_cache_counts().0;
            let output = block_on(session.invoke_executable(&unit, invocation, &control))
                .unwrap_or_else(|error| panic!("{mode:?} first public invocation: {error}"));
            assert_eq!(output.kind(), runmat_value::ValueSequenceKind::Single);
            assert_eq!(
                output.as_slice(),
                &[runmat_value::Value::Num(23.0)],
                "{mode:?} fresh persistent must initialize as empty before its first assignment"
            );
            if matches!(mode, ExecutionMode::ForcedGenericNative) {
                assert!(
                    session.generic_native_cache_counts().0 > native_before,
                    "forced-native public path must compile and invoke native code"
                );
            }
            let selected =
                session.execution_runtime_context(unit.revision().program_revision.clone());
            assert_eq!(
                selected
                    .session_variable_authority()
                    .persistent_named_value("selfClearFirst", "p")
                    .expect("selected session persistent lookup"),
                None,
                "{mode:?} self-clear must retire canonical persistent state"
            );
        }
    });
}
