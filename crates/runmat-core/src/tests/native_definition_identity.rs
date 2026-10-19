use crate::{execute_text_request_for_testing, RunMatSession};
use runmat_jit::tiering::TieringConfig;
use runmat_value::Value;

fn execute(session: &mut RunMatSession, source: &str) -> bool {
    let outcome = execute_text_request_for_testing(session, source).unwrap();
    assert!(outcome.error.is_none(), "{:?}", outcome.error);
    outcome.used_jit
}

fn session() -> RunMatSession {
    let mut session = RunMatSession::with_options(true, false).unwrap();
    session.set_native_tiering_config_for_testing(TieringConfig {
        generic_hot_threshold: 1,
        specialized_hot_threshold: 100,
        deterministic: true,
        ..Default::default()
    });
    session
}

#[test]
fn repeated_callback_witness_uses_native_and_preserves_old_capture_instances() {
    runmat_gc::gc_test_context(|| {
        let mut session = session();
        execute(&mut session, "offset = 1;");
        let source = "f = @(x) x + offset; answer = f(2); assert(answer == offset + 2);";
        execute(&mut session, source);
        execute(&mut session, "saved = f;");
        let mut native = 0;
        let mut saved_native = 0;
        for _ in 0..5 {
            execute(&mut session, "offset = offset + 1;");
            native += usize::from(execute(&mut session, source));
            saved_native += usize::from(execute(&mut session, "assert(saved(2) == 3);"));
        }
        assert!(
            native >= 3,
            "repeated callback witness must actually execute native"
        );
        assert!(
            saved_native >= 3,
            "retained closure calls must originate in native code"
        );
        assert_eq!(
            session.get_variables().unwrap().get("answer"),
            Some(&Value::Num(8.0))
        );
    });
}

#[test]
fn changed_definition_invalidates_native_dynamic_lookup() {
    runmat_gc::gc_test_context(|| {
        let mut session = session();
        let initial = "f = @(x) x + 1; answer = f(2); assert(answer == 3);";
        execute(&mut session, initial);
        assert!(execute(&mut session, initial));
        let changed = "f = @(x) x + 100; answer = f(2); assert(answer == 102);";
        execute(&mut session, changed);
        assert!(execute(&mut session, changed));
        assert_eq!(
            session.get_variables().unwrap().get("answer"),
            Some(&Value::Num(102.0))
        );
        // Restoring the original authored source is another real publication,
        // not permission to execute a cached entry against stale callable IDs.
        execute(&mut session, initial);
        assert_eq!(
            session.get_variables().unwrap().get("answer"),
            Some(&Value::Num(3.0))
        );
    });
}

#[test]
fn native_retained_named_handle_preserves_outputs_and_caught_errors() {
    runmat_gc::gc_test_context(|| {
        let mut session = session();
        execute(&mut session,
            "seed = 0; function [a,b] = retained_pair(x)\nif x < 0\nerror('RunMat:RetainedFailure', 'negative input');\nend\na = x; b = x + 1;\nend");
        execute(&mut session, "saved = @retained_pair;");
        let outputs = "[a,b] = saved(3); assert(a == 3 && b == 4);";
        execute(&mut session, outputs);
        assert!(
            execute(&mut session, outputs),
            "multi-output caller must execute native"
        );
        let caught = "caught = false; try\n[a,b] = saved(-1);\ncatch err\ncaught = strcmp(err.identifier, 'RunMat:RetainedFailure');\nend\nassert(caught);";
        execute(&mut session, caught);
        assert!(
            execute(&mut session, caught),
            "catching caller must execute native"
        );
        assert_eq!(
            session.get_variables().unwrap().get("a"),
            Some(&Value::Num(3.0))
        );
        assert_eq!(
            session.get_variables().unwrap().get("b"),
            Some(&Value::Num(4.0))
        );
    });
}
