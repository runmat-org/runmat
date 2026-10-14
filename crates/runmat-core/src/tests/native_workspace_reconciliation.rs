use crate::{execute_text_request_for_testing, RunMatSession};
use futures::executor::block_on;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

#[path = "native_workspace_reconciliation/persistent_return.rs"]
mod persistent_return;
#[path = "native_workspace_reconciliation/public_self_clear.rs"]
mod public_self_clear;

fn execute_workspace_outcome(
    session: &mut RunMatSession,
    source: &str,
) -> runmat_runtime::execution::RootedResult<crate::abi::ExecutionOutcome> {
    let outcome = super::execute_text_request(session, source).expect("workspace execution");
    assert!(
        !outcome
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.severity == crate::abi::DiagnosticSeverity::Error),
        "{:?}",
        outcome.diagnostics
    );
    outcome
}

#[test]
fn completed_register_metadata_cannot_overwrite_newer_canonical_state() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let outcome = execute_text_request_for_testing(&mut session, "x = 99;").unwrap();
        assert!(outcome.error.is_none(), "{:?}", outcome.error);

        session.refresh_workspace_binding_slots(
            &HashMap::from([("x".to_string(), (0, runmat_value::Value::Num(1.0)))]),
            &Default::default(),
        );

        assert_eq!(
            session.get_variables().unwrap().get("x"),
            Some(&runmat_value::Value::Num(99.0))
        );
        assert_eq!(session.lowering_workspace_bindings().get("x"), Some(&0));
    });
}

#[test]
fn top_level_global_binding_survives_requests_without_copying_its_value_into_base() {
    runmat_gc::gc_test_context(|| {
        for jit in [false, true] {
            let mut session = RunMatSession::with_options(jit, false).unwrap();
            if jit {
                session.set_native_tiering_config_for_testing(runmat_jit::tiering::TieringConfig {
                    generic_hot_threshold: 1,
                    specialized_hot_threshold: 100,
                    deterministic: true,
                    ..Default::default()
                });
            }
            let workspace = session.workspace_handle().0;

            let initial = execute_workspace_outcome(&mut session, "g = 1;");
            let local_key = initial
                .workspace_delta
                .upserts
                .iter()
                .find(|entry| workspace_key_name(&entry.key) == "g")
                .expect("local g upsert")
                .key
                .clone();
            assert!(matches!(
                &local_key,
                crate::abi::WorkspaceBindingKey::Interactive { session, name }
                    if *session == workspace && name.0 == "g"
            ));
            let binding = execute_workspace_outcome(&mut session, "global g; g = 9; observed = g;");
            assert!(binding.workspace_delta.upserts.iter().any(|entry| {
                matches!(
                    &entry.key,
                    crate::abi::WorkspaceBindingKey::Global {
                        scope: crate::abi::GlobalScopeKey::Session(session),
                        name,
                    } if *session == workspace
                        && name.0 == "g"
                        && entry.value == runmat_value::Value::Num(9.0)
                )
            }));
            assert!(binding.workspace_delta.removals.contains(&local_key));
            assert_eq!(session.base_workspace_snapshot().unwrap().get("g"), None);

            let read =
                execute_text_request_for_testing(&mut session, "observed_again = g;").unwrap();
            assert!(read.error.is_none(), "jit={jit}: {:?}", read.error);
            assert_eq!(
                session.get_variables().unwrap().get("observed_again"),
                Some(&runmat_value::Value::Num(9.0))
            );
            assert!(
                session
                    .workspace_snapshot()
                    .unwrap()
                    .values
                    .iter()
                    .any(|entry| entry.name == "g"),
                "logical global binding should remain visible in the workspace snapshot"
            );

            let bytes = block_on(session.export_workspace_state(crate::WorkspaceExportMode::Force))
                .unwrap()
                .expect("forced export");
            let mut restored = RunMatSession::with_options(false, false).unwrap();
            restored.import_workspace_state(&bytes).unwrap();
            assert_eq!(
                restored.get_variables().unwrap().get("g"),
                Some(&runmat_value::Value::Num(9.0))
            );
        }
    });
}

#[test]
fn clearing_a_durable_global_binding_restores_local_then_redeclare_reads_global_store() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let workspace = session.workspace_handle().0;
        let bind = execute_workspace_outcome(&mut session, "global g; g = 9;");
        let global_key = bind
            .workspace_delta
            .upserts
            .iter()
            .find(|entry| {
                matches!(
                    &entry.key,
                    crate::abi::WorkspaceBindingKey::Global {
                        scope: crate::abi::GlobalScopeKey::Session(session),
                        name,
                    } if *session == workspace && name.0 == "g"
                )
            })
            .expect("global g upsert")
            .key
            .clone();

        let local = execute_workspace_outcome(&mut session, "clear g; g = 3;");
        assert!(local.workspace_delta.removals.contains(&global_key));
        assert!(local.workspace_delta.upserts.iter().any(|entry| {
            matches!(
                &entry.key,
                crate::abi::WorkspaceBindingKey::Interactive { session, name }
                    if *session == workspace
                        && name.0 == "g"
                        && entry.value == runmat_value::Value::Num(3.0)
            )
        }));
        assert_eq!(
            session.get_variables().unwrap().get("g"),
            Some(&runmat_value::Value::Num(3.0))
        );

        let rebound =
            execute_text_request_for_testing(&mut session, "global g; retained_global = g;")
                .unwrap();
        assert!(rebound.error.is_none(), "{:?}", rebound.error);
        let values = session.get_variables().unwrap();
        assert_eq!(values.get("g"), Some(&runmat_value::Value::Num(9.0)));
        assert_eq!(
            values.get("retained_global"),
            Some(&runmat_value::Value::Num(9.0))
        );
    });
}

fn workspace_key_name(key: &crate::abi::WorkspaceBindingKey) -> &str {
    match key {
        crate::abi::WorkspaceBindingKey::SourceBinding { binding, .. } => &binding.0,
        crate::abi::WorkspaceBindingKey::Interactive { name, .. }
        | crate::abi::WorkspaceBindingKey::Global { name, .. }
        | crate::abi::WorkspaceBindingKey::Persistent { name, .. } => &name.0,
    }
}

#[test]
fn procedure_global_binding_does_not_reclassify_the_top_level_binding() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(false, false).unwrap();
        let definition = execute_text_request_for_testing(
            &mut session,
            "g = 4; result = updateGlobal();\nfunction out = updateGlobal()\nglobal g; g = 11; out = g;\nend",
        )
        .unwrap();
        assert!(definition.error.is_none(), "{:?}", definition.error);

        let local = execute_text_request_for_testing(&mut session, "top_level_seen = g;").unwrap();
        assert!(local.error.is_none(), "{:?}", local.error);
        assert_eq!(
            session.get_variables().unwrap().get("top_level_seen"),
            Some(&runmat_value::Value::Num(4.0))
        );

        let global =
            execute_text_request_for_testing(&mut session, "global g; session_global_seen = g;")
                .unwrap();
        assert!(global.error.is_none(), "{:?}", global.error);
        assert_eq!(
            session.get_variables().unwrap().get("session_global_seen"),
            Some(&runmat_value::Value::Num(11.0))
        );
    });
}

#[test]
fn native_calls_publish_dynamic_workspace_writes_before_following_reads() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(true, false).unwrap();
        session.set_native_tiering_config_for_testing(runmat_jit::tiering::TieringConfig {
            generic_hot_threshold: 1,
            specialized_hot_threshold: 100,
            deterministic: true,
            ..Default::default()
        });
        let source = "alpha = 17; syms alpha; assert(isa(alpha, 'sym')); assert(strcmp(char(alpha), 'alpha'));";
        let mut entered_native = false;
        for _ in 0..6 {
            let outcome = execute_text_request_for_testing(&mut session, source).unwrap();
            assert!(outcome.error.is_none(), "{:?}", outcome.error);
            entered_native |= outcome.used_jit;
        }
        assert!(
            entered_native,
            "interpreter fallback is not native evidence"
        );
    });
}

#[test]
fn declared_global_sees_workspace_builtin_assignment_before_following_read() {
    runmat_gc::gc_test_context(|| {
        let source = "global alpha; alpha = 17; syms alpha; assert(isa(alpha, 'sym'));";
        for jit in [false, true] {
            let mut session = crate::RunMatSession::with_options(jit, false).unwrap();
            if jit {
                session.set_native_tiering_config_for_testing(runmat_jit::tiering::TieringConfig {
                    generic_hot_threshold: 1,
                    specialized_hot_threshold: 100,
                    deterministic: true,
                    ..Default::default()
                });
            }
            let mut entered_native = false;
            for attempt in 0..6 {
                let outcome = execute_text_request_for_testing(&mut session, source).unwrap();
                assert!(
                    outcome.error.is_none(),
                    "jit={jit} attempt={attempt} error={:?}",
                    outcome.error
                );
                entered_native |= outcome.used_jit;
            }
            if jit {
                assert!(
                    entered_native,
                    "native workspace-global witness must execute native"
                );
            } else {
                assert!(!entered_native);
            }
        }
    });
}

#[test]
fn clearing_a_global_alias_keeps_the_session_value_and_later_local_writes_local() {
    runmat_gc::gc_test_context(|| {
        let source = "global alpha; alpha = 17; clear('alpha'); alpha = 19; assert(alpha == 19);";
        let check_store = "global alpha; assert(alpha == 17);";
        for jit in [false, true] {
            let mut session = RunMatSession::with_options(jit, false).unwrap();
            if jit {
                session.set_native_tiering_config_for_testing(runmat_jit::tiering::TieringConfig {
                    generic_hot_threshold: 1,
                    specialized_hot_threshold: 100,
                    deterministic: true,
                    ..Default::default()
                });
            }
            let mut entered_native = false;
            for attempt in 0..6 {
                let outcome = execute_text_request_for_testing(&mut session, source).unwrap();
                assert!(
                    outcome.error.is_none(),
                    "jit={jit} attempt={attempt} error={:?}",
                    outcome.error
                );
                entered_native |= outcome.used_jit;
                let stored = execute_text_request_for_testing(&mut session, check_store).unwrap();
                assert!(stored.error.is_none(), "{:?}", stored.error);
            }
            if jit {
                assert!(entered_native, "global unbinding must execute native");
            } else {
                assert!(!entered_native);
            }
        }
    });
}

#[test]
fn native_failed_declarations_preserve_existing_workspace_values() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(true, false).unwrap();
        session.set_native_tiering_config_for_testing(runmat_jit::tiering::TieringConfig {
            generic_hot_threshold: 1,
            specialized_hot_threshold: 100,
            deterministic: true,
            ..Default::default()
        });
        let source = "sentinel = 19; rejected = false; try; syms('sentinel invalid-name'); catch; rejected = true; end; assert(rejected); assert(sentinel == 19); syms sentinel; assert(isa(sentinel, 'sym'));";
        let mut entered_native = false;
        for _ in 0..6 {
            let outcome = execute_text_request_for_testing(&mut session, source).unwrap();
            assert!(outcome.error.is_none(), "{:?}", outcome.error);
            entered_native |= outcome.used_jit;
        }
        assert!(
            entered_native,
            "interpreter fallback is not native evidence"
        );
    });
}

#[test]
fn dynamic_clear_and_load_reconcile_before_following_numeric_region() {
    runmat_gc::gc_test_context(|| {
        let temp = tempfile::TempDir::new().expect("create MAT workspace fixture directory");
        let path = temp
            .path()
            .join("workspace-reconciliation.mat")
            .to_string_lossy()
            .replace('\\', "\\\\");
        let source = format!(
            "x = [4, 5, 6]; save('{path}', 'x'); x = [1, 2, 3]; clear('x'); load('{path}'); y = x + 1; assert(isequal(y, [5, 6, 7]));"
        );
        for jit in [false, true] {
            let mut session = RunMatSession::with_options(jit, false).unwrap();
            if jit {
                session.set_native_tiering_config_for_testing(runmat_jit::tiering::TieringConfig {
                    generic_hot_threshold: 1,
                    specialized_hot_threshold: 100,
                    deterministic: true,
                    ..Default::default()
                });
            }
            let mut entered_native = false;
            for attempt in 0..6 {
                let outcome = execute_text_request_for_testing(&mut session, &source).unwrap();
                assert!(
                    outcome.error.is_none(),
                    "jit={jit} attempt={attempt} error={:?}",
                    outcome.error
                );
                entered_native |= outcome.used_jit;
            }
            if jit {
                assert!(
                    entered_native,
                    "workspace reconciliation must execute native"
                );
            }
        }
    });
}

#[test]
fn ordinary_assignment_survives_a_later_error_in_interpreter_and_native_execution() {
    runmat_gc::gc_test_context(|| {
        let failing = "committed = 42; error('RunMat:test:later', 'later failure');";
        let verify = "assert(committed == 42);";
        for jit in [false, true] {
            let mut session = RunMatSession::with_options(jit, false).unwrap();
            if jit {
                session.set_native_tiering_config_for_testing(runmat_jit::tiering::TieringConfig {
                    generic_hot_threshold: 1,
                    specialized_hot_threshold: 100,
                    deterministic: true,
                    ..Default::default()
                });
            }
            let mut entered_native = false;
            for attempt in 0..6 {
                let outcome = execute_text_request_for_testing(&mut session, failing).unwrap();
                assert!(outcome.error.is_some(), "attempt {attempt} must fail");
                entered_native |= outcome.used_jit;
                let verification = execute_text_request_for_testing(&mut session, verify).unwrap();
                assert!(
                    verification.error.is_none(),
                    "jit={jit} attempt={attempt} lost the earlier assignment: {:?}",
                    verification.error
                );
            }
            if jit {
                assert!(entered_native, "partial-error witness must execute native");
            }
        }
    });
}

#[test]
fn native_assignment_before_suspended_cancellation_remains_in_base_workspace() {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(true, false).unwrap();
        session.set_native_tiering_config_for_testing(runmat_jit::tiering::TieringConfig {
            generic_hot_threshold: 1,
            specialized_hot_threshold: 100,
            deterministic: true,
            ..Default::default()
        });
        execute_text_request_for_testing(&mut session, "delay = 0;").unwrap();
        let source = "committed_before_cancel = 91; pause(delay);";
        let mut entered_native = false;
        for _ in 0..6 {
            let outcome = execute_text_request_for_testing(&mut session, source).unwrap();
            assert!(outcome.error.is_none(), "{:?}", outcome.error);
            entered_native |= outcome.used_jit;
        }
        assert!(
            entered_native,
            "cancellation witness must warm native execution"
        );
        execute_text_request_for_testing(&mut session, "delay = 30;").unwrap();

        let interrupt = session.interrupt_handle();
        let request = crate::abi::ExecutionRequest::for_source(
            crate::abi::SourceInput::Text {
                name: "<native-workspace-cancel>".into(),
                text: source.into(),
            },
            session.compat_mode(),
            crate::abi::HostExecutionPolicy::default(),
            session.workspace_handle(),
        );
        let mut execution = Box::pin(session.execute_request(request));
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        let first = Future::poll(Pin::as_mut(&mut execution), &mut context);
        assert!(matches!(first, Poll::Pending));
        interrupt.cancel(runmat_execution::CancellationReason::User);
        let response = futures::executor::block_on(execution);
        match response.result {
            Ok(outcome) => assert!(outcome.diagnostics.iter().any(|diagnostic| {
                diagnostic.code == "RunMat:ExecutionCancelled"
                    || diagnostic.code == "ExecutionCancelled"
            })),
            Err(crate::RunError::Runtime(error)) => assert!(matches!(
                error.identifier(),
                Some("RunMat:ExecutionCancelled" | "ExecutionCancelled")
            )),
            Err(error) => panic!("unexpected cancellation error: {error}"),
        }

        let verification = execute_text_request_for_testing(
            &mut session,
            "assert(committed_before_cancel == 91);",
        )
        .unwrap();
        assert!(verification.error.is_none(), "{:?}", verification.error);
    });
}
