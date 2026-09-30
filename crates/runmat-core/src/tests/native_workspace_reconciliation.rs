use crate::{execute_text_request_for_testing, RunMatSession};

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
