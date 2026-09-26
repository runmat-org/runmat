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
