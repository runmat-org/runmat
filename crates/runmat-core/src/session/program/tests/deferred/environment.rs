//! Original deferred environment across caller mutation and both execution engines.
use super::*;
use runmat_value::CharArray;

#[tokio::test]
async fn original_future_environment_survives_caller_mutation_without_mutating_parent() {
    let name = "RUNMAT_DEFERRED_ORIGINAL";
    let host = std::env::var_os(name);
    for native in [false, true] {
        let mut session = RunMatSession::with_frozen_environment(
            false,
            false,
            vec![(name.into(), "captured".into())],
        )
        .unwrap();
        session.set_compat_mode(crate::CompatMode::RunMat);
        let service = Rc::new(CapturingService::new(&session));
        session.install_execution_services(service.clone());
        let unit = session.compile_executable_unit(ExecutableSource::new(
            "deferred-environment-owner", "environment.m",
            "function f = entry(); f = originalEnvironment(); end\nasync function y = originalEnvironment(); y = getenv('RUNMAT_DEFERRED_ORIGINAL'); setenv('RUNMAT_DEFERRED_ORIGINAL','child'); end\nasync function y = finish(f); y = await(f); end"), None).await.unwrap();
        let control = if native {
            crate::InvocationControl::default().force_generic_native()
        } else {
            crate::InvocationControl::default()
        };
        let future = session
            .invoke_executable(
                &unit,
                crate::ProcedureInvocation {
                    target: crate::ProcedureTarget::Function("entry".into()),
                    arguments: crate::ProcedureArguments::Source(vec![]),
                    requested_outputs: 1,
                },
                &control,
            )
            .await
            .unwrap();
        let original = service.take_original();
        assert_eq!(
            original.call.environment.entries(),
            [(name.into(), "captured".into())]
        );
        session
            .install_environment_snapshot(vec![(name.into(), "later".into())])
            .unwrap();
        let value = session
            .invoke_executable(
                &unit,
                crate::ProcedureInvocation {
                    target: crate::ProcedureTarget::Function("finish".into()),
                    arguments: crate::ProcedureArguments::Source(
                        vec![future.as_slice()[0].clone()],
                    ),
                    requested_outputs: 1,
                },
                &control,
            )
            .await
            .unwrap();
        assert_eq!(
            value.as_slice(),
            &[Value::CharArray(CharArray::new_row("captured"))],
            "native={native}"
        );
        assert_eq!(
            session.environment_snapshot(),
            [(name.into(), "later".into())]
        );
    }
    assert_eq!(std::env::var_os(name), host);
}
