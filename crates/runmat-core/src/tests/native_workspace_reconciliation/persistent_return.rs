use super::*;

#[test]
fn forced_native_recursive_persistent_output_returns_latest_canonical_value() {
    runmat_gc::gc_test_context(|| {
        // The output local is also the declared persistent. Each recursive
        // invocation writes the same canonical value while the outer frame's
        // prior SSA value remains an already-evaluated snapshot.
        let source = crate::ExecutableSource::new(
            "native-persistent-recursion@1",
            "nativePersistentReturn.m",
            "function count = nativePersistentReturn(depth)\npersistent count\nif isempty(count)\ncount = 0;\nend\ncount = count + 1;\nif depth > 0\nignored = nativePersistentReturn(depth - 1);\nend\nend\n",
        );
        let mut interpreted =
            RunMatSession::with_options(false, false).expect("interpreter session");
        let unit = block_on(interpreted.compile_executable_unit(source, None))
            .expect("compile recursive persistent output");
        let mut native = RunMatSession::with_options(false, false).expect("native session");

        for (depth, expected) in [(2.0, 3.0), (1.0, 5.0)] {
            let invocation = crate::ProcedureInvocation {
                target: crate::ProcedureTarget::Function("nativePersistentReturn".into()),
                arguments: crate::ProcedureArguments::Source(vec![runmat_value::Value::Num(depth)]),
                requested_outputs: 1,
            };
            let baseline = block_on(interpreted.invoke_executable(
                &unit,
                invocation.clone(),
                &crate::InvocationControl::default(),
            ))
            .expect("disabled-JIT interpreter recursive persistent return");
            let compiled = block_on(native.invoke_executable(
                &unit,
                invocation,
                &crate::InvocationControl::default().force_generic_native(),
            ))
            .expect("forced generic-native recursive persistent return");
            super::super::assert_single_output(&baseline, runmat_value::Value::Num(expected));
            super::super::assert_single_output(&compiled, runmat_value::Value::Num(expected));
        }

        for session in [&mut interpreted, &mut native] {
            let cleared = crate::execute_text_request_for_testing(
                session,
                "clear('nativePersistentReturn');",
            )
            .expect("typed function clear request");
            assert!(cleared.error.is_none(), "{:?}", cleared.error);
        }
        let invocation = crate::ProcedureInvocation {
            target: crate::ProcedureTarget::Function("nativePersistentReturn".into()),
            arguments: crate::ProcedureArguments::Source(vec![runmat_value::Value::Num(0.0)]),
            requested_outputs: 1,
        };
        let after_clear = block_on(interpreted.invoke_executable(
            &unit,
            invocation.clone(),
            &crate::InvocationControl::default(),
        ))
        .expect("interpreter after selected function clear");
        let native_after_clear = block_on(native.invoke_executable(
            &unit,
            invocation,
            &crate::InvocationControl::default().force_generic_native(),
        ))
        .expect("native after selected function clear");
        super::super::assert_single_output(&after_clear, runmat_value::Value::Num(1.0));
        super::super::assert_single_output(&native_after_clear, runmat_value::Value::Num(1.0));
    });
}
