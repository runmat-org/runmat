use crate::{
    execute_text_request_for_testing, ExecutableSource, InvocationControl, ProcedureInvocation,
    ProcedureTarget, RunMatSession,
};
use futures::executor::block_on;
use runmat_jit::tiering::TieringConfig;
use runmat_value::Value;

fn config(max_profile_bytes: usize) -> TieringConfig {
    TieringConfig {
        generic_hot_threshold: 1,
        specialized_hot_threshold: 2,
        deterministic: true,
        max_profile_bytes,
        ..Default::default()
    }
}

#[test]
fn generic_native_executes_complete_workspace_beyond_profile_value_bound() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(true, false).unwrap();
        let seed = (0..65)
            .map(|index| format!("v{index} = {index};"))
            .collect::<String>();
        let result = execute_text_request_for_testing(&mut session, &seed).unwrap();
        assert!(result.error.is_none(), "{:?}", result.error);
        assert!(session.get_variables().unwrap().len() >= 65);
        session.set_native_tiering_config_for_testing(config(64 * 1024));
        session.use_generic_ready_route_for_testing();
        let source = "v0 = v0 + 1; assert(v64 == 64);";
        let mut native = 0;
        for _ in 0..4 {
            let result = execute_text_request_for_testing(&mut session, source).unwrap();
            assert!(result.error.is_none(), "{:?}", result.error);
            native += usize::from(result.used_jit);
        }
        assert!(native >= 2, "actual generic native execution is required");
        assert_eq!(
            session.get_variables().unwrap().get("v0"),
            Some(&Value::Num(4.0))
        );
        assert_eq!(
            session.get_variables().unwrap().get("v64"),
            Some(&Value::Num(64.0))
        );
        assert_eq!(session.generic_native_cache_counts(), (1, 1));
        assert_eq!(session.specialized_native_version_count_for_testing(), 0);
        let snapshot = session.native_tiering_snapshot_for_testing();
        assert_eq!(snapshot.sites.len(), 1);
        assert_eq!(snapshot.sites[0].invocations, 4);
        assert!(snapshot.sites[0].profiles.is_empty());
    });
}

fn invoke_function_without_profile(argument_count: usize, max_profile_bytes: usize) {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(true, false).unwrap();
        session.set_native_tiering_config_for_testing(config(max_profile_bytes));
        session.use_generic_ready_route_for_testing();
        let names = (0..argument_count)
            .map(|index| format!("x{index}"))
            .collect::<Vec<_>>();
        let source = format!(
            "function y = unprofiled({})\ny = x0 + {};\nend\n",
            names.join(","),
            names.last().unwrap()
        );
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new("optional-profile@1", "unprofiled.m", source),
            None,
        ))
        .unwrap();
        for _ in 0..4 {
            let result = block_on(
                session.invoke_executable(
                    &unit,
                    ProcedureInvocation {
                        target: ProcedureTarget::Function("unprofiled".into()),
                        arguments: crate::ProcedureArguments::Source(
                            (0..argument_count)
                                .map(|index| Value::Num(index as f64))
                                .collect(),
                        ),
                        requested_outputs: 1,
                    },
                    &InvocationControl::default(),
                ),
            )
            .unwrap();
            assert_eq!(
                result.as_slice(),
                &[Value::Num((argument_count - 1) as f64)]
            );
        }
        assert!(session.stats().jit_compiled >= 2);
        assert_eq!(session.generic_native_cache_counts(), (1, 1));
        assert_eq!(session.specialized_native_version_count_for_testing(), 0);
        let snapshot = session.native_tiering_snapshot_for_testing();
        assert_eq!(snapshot.sites.len(), 1);
        assert_eq!(snapshot.sites[0].invocations, 4);
        assert!(snapshot.sites[0].profiles.is_empty());
    });
}

#[test]
fn exact_function_entry_tiers_without_an_oversized_argument_profile() {
    invoke_function_without_profile(65, 64 * 1024);
}

#[test]
fn profile_byte_bound_does_not_disable_generic_function_execution() {
    invoke_function_without_profile(1, 1);
}

#[test]
fn profile_byte_bound_does_not_disable_empty_argument_entrypoint() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(true, false).unwrap();
        session.set_native_tiering_config_for_testing(config(1));
        session.use_generic_ready_route_for_testing();
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new(
                "optional-profile-entry@1",
                "entry.m",
                "assert(21 + 21 == 42);",
            ),
            None,
        ))
        .unwrap();
        for _ in 0..4 {
            let result = block_on(session.invoke_executable(
                &unit,
                ProcedureInvocation {
                    target: ProcedureTarget::Entrypoint,
                    arguments: crate::ProcedureArguments::Source(vec![]),
                    requested_outputs: 0,
                },
                &InvocationControl::default(),
            ))
            .unwrap();
            assert!(result.is_empty());
        }
        assert!(session.stats().jit_compiled >= 2);
        assert_eq!(session.specialized_native_version_count_for_testing(), 0);
        let snapshot = session.native_tiering_snapshot_for_testing();
        assert_eq!(snapshot.sites[0].invocations, 4);
        assert!(snapshot.sites[0].profiles.is_empty());
    });
}

#[test]
fn cache_selects_generic_when_specialized_code_exists_but_current_profile_is_unavailable() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(true, false).unwrap();
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new(
                "optional-profile-cache@1",
                "profiled.m",
                "function y = profiled(x)\ny = x + 1;\nend\n",
            ),
            None,
        ))
        .unwrap();
        let mut cache =
            crate::generic_native::GenericNativeCache::with_tiering_config(config(64 * 1024));
        let oversized_arguments = vec![Value::Num(1.0); 65];
        assert!(cache.representation_profile(&oversized_arguments).is_none());
        let profile = cache.representation_profile(&[Value::Num(1.0)]).unwrap();
        cache
            .observe_invocation(&unit, Some("profiled"), Some(&profile), 10)
            .unwrap();
        let generic = cache
            .resolve_or_schedule(&unit, Some("profiled"), Some(&profile))
            .unwrap()
            .unwrap();
        assert!(matches!(
            generic.tier,
            runmat_jit::entry::PublishedTier::Generic
        ));
        cache
            .observe_invocation(&unit, Some("profiled"), Some(&profile), 20)
            .unwrap();
        let specialized = cache
            .resolve_or_schedule(&unit, Some("profiled"), Some(&profile))
            .unwrap()
            .unwrap();
        assert!(!matches!(
            specialized.tier,
            runmat_jit::entry::PublishedTier::Generic
        ));
        let unprofiled = cache
            .resolve_or_schedule(&unit, Some("profiled"), None)
            .unwrap()
            .unwrap();
        assert!(matches!(
            unprofiled.tier,
            runmat_jit::entry::PublishedTier::Generic
        ));
        assert!(cache
            .resolve_or_schedule_osr(
                &unit,
                Some("profiled"),
                None,
                runmat_native_codegen::NativeLivenessAdmission::AuthorizedUnbounded,
            )
            .unwrap()
            .is_none());
        cache
            .observe_invocation(&unit, Some("profiled"), None, 30)
            .unwrap();
        let snapshot = cache.tiering_snapshot();
        assert_eq!(snapshot.sites[0].invocations, 3);
        assert_eq!(snapshot.sites[0].profiles[0].samples, 2);
        let bounded = crate::generic_native::GenericNativeCache::with_tiering_config(config(1));
        assert!(bounded.representation_profile(&[]).is_none());
        assert!(bounded.representation_profile(&[Value::Num(1.0)]).is_none());
    });
}
