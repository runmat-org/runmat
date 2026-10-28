//! Public object-subscript behavior through actual hot native execution.

use crate::{execute_text_request_for_testing, RunMatSession};
use runmat_jit::tiering::TieringConfig;
use runmat_value::Value;

fn assert_hot(setup: &str, source: &str, expected: &[(&str, Value)]) {
    runmat_gc::gc_test_context(|| {
        let mut session = RunMatSession::with_options(true, false).unwrap();
        session.set_native_tiering_config_for_testing(TieringConfig {
            generic_hot_threshold: 1,
            specialized_hot_threshold: 100,
            deterministic: true,
            ..Default::default()
        });
        if !setup.is_empty() {
            let outcome = execute_text_request_for_testing(&mut session, setup).unwrap();
            assert!(outcome.error.is_none(), "{:?}", outcome.error);
        }
        let mut native_calls = 0;
        for _ in 0..5 {
            let outcome = execute_text_request_for_testing(&mut session, source).unwrap();
            assert!(outcome.error.is_none(), "{:?}", outcome.error);
            native_calls += usize::from(outcome.used_jit);
            for (name, value) in expected {
                assert_eq!(
                    session.get_variables().unwrap().get(*name),
                    Some(value),
                    "{name}"
                );
            }
        }
        assert!(
            native_calls >= 2,
            "public indexing must execute native code"
        );
    });
}

#[test]
fn native_public_subscript_numeric_defaults_preserve_values() {
    assert_hot(
        "",
        "values = [10, 20]; S = struct('type', '()', 'subs', {{2}}); selected = subsref(values, S); changed = subsasgn(values, S, 99); assigned = changed(2); untouched = changed(1);",
        &[
            ("selected", Value::Num(20.0)),
            ("assigned", Value::Num(99.0)),
            ("untouched", Value::Num(10.0)),
        ],
    );
}

#[test]
fn native_dotted_semantic_fallback_uses_bound_local_function() {
    assert_hot(
        "",
        "obj = 2; result = obj.bump(3);\nfunction out = bump(receiver, value)\n out = receiver + value;\nend",
        &[("result", Value::Num(5.0))],
    );
}

#[test]
fn native_class_reference_method_errors_keep_their_owning_identifiers() {
    assert_hot(
        "__register_test_classes();",
        "missing = ''; nonstatic = '';\ntry\nclassref('Point').definitely_missing(1);\ncatch err\nmissing = err.identifier;\nend\ntry\nclassref('Point').move(1, 2);\ncatch err\nnonstatic = err.identifier;\nend",
        &[
            ("missing", Value::CharArray(runmat_value::CharArray::new_row("RunMat:UndefinedFunction"))),
            ("nonstatic", Value::CharArray(runmat_value::CharArray::new_row("RunMat:MethodNotStatic"))),
        ],
    );
}

#[test]
fn native_public_subscript_object_growth_uses_canonical_scatter() {
    assert_hot(
        "__register_test_classes();",
        "p = new_object('Point'); q = new_object('Point'); q.x = 77; S = struct('type', '()', 'subs', {{2}}); p = subsasgn(p, S, q); count = numel(p); second = p(2); first = p(1); assigned = second.x; untouched = first.x;",
        &[
            ("count", Value::Num(2.0)),
            ("assigned", Value::Num(77.0)),
            ("untouched", Value::Num(0.0)),
        ],
    );
}

#[test]
fn native_public_subscript_rejects_malformed_selector_with_owning_identifier() {
    assert_hot(
        "",
        "S = struct('type', '()', 'subs', 42); read_id = ''; write_id = ''; try\nselected = subsref([10,20], S);\ncatch err\nread_id = err.identifier;\nend\ntry\nchanged = subsasgn([10,20], S, 99);\ncatch err\nwrite_id = err.identifier;\nend",
        &[
            ("read_id", Value::CharArray(runmat_value::CharArray::new_row("RunMat:InvalidObjectSubscriptPath"))),
            ("write_id", Value::CharArray(runmat_value::CharArray::new_row("RunMat:InvalidObjectSubscriptPath"))),
        ],
    );
}
