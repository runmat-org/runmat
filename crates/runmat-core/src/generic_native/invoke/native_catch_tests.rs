//! Public forced-generic-native witnesses for lexical catch authority.

use std::error::Error as _;

use super::native_failure_tests::forced_native_failure;
use runmat_runtime::RuntimeError;

fn assert_origin(error: &RuntimeError, source: &str, identifier: &str, message: &str) {
    assert_eq!(error.identifier(), Some(identifier));
    assert_eq!(error.message(), message);
    assert_eq!(error.context.call_frames.len(), 1);
    assert_eq!(error.context.call_frames[0].function, "native_exception");
    let origin = source.find("error(").expect("origin error call");
    let span = error.span.as_ref().expect("native origin span");
    assert!(span.offset() <= origin && origin < span.offset() + span.len());
}

#[test]
fn direct_zero_argument_rethrow_keeps_original_failure_and_cause() {
    runmat_gc::gc_test_context(|| {
        const DIRECT: &str = "function native_exception()\nerror('bad %q', 1);\nend\n";
        const SOURCE: &str = "function native_exception()\ntry\n error('bad %q', 1);\ncatch\n rethrow();\nend\nend\n";
        let direct = forced_native_failure("native-lexical-cause-direct@1", DIRECT);
        let error = forced_native_failure("native-lexical-direct@1", SOURCE);
        assert_eq!(error.identifier(), Some("RunMat:error"));
        assert_eq!(error.message(), direct.message());
        assert_eq!(error.context.call_frames.len(), 1);
        assert_eq!(error.context.call_frames[0].function, "native_exception");
        let origin = SOURCE.find("error(").unwrap();
        let rethrow = SOURCE.find("rethrow()").unwrap();
        let span = error.span.as_ref().expect("original failure span");
        assert!(span.offset() <= origin && origin < span.offset() + span.len());
        assert!(span.offset() + span.len() <= rethrow);
        let mut original = &error;
        while let Some(shared) = original.source().and_then(|source| {
            source.downcast_ref::<runmat_runtime::runtime_error::SharedRuntimeErrorSource>()
        }) {
            original = shared.original();
        }
        assert!(original
            .source()
            .is_some_and(|source| source.is::<RuntimeError>()));
    });
}

#[test]
fn feval_rethrow_inherits_native_lexical_catch() {
    runmat_gc::gc_test_context(|| {
        const SOURCE: &str = "function native_exception()\ntry\n error('RunMat:IndirectOrigin', 'indirect boom');\ncatch\n feval('rethrow');\nend\nend\n";
        let error = forced_native_failure("native-lexical-feval@1", SOURCE);
        assert_origin(&error, SOURCE, "RunMat:IndirectOrigin", "indirect boom");
        let span = error.span.as_ref().unwrap();
        assert!(span.offset() + span.len() <= SOURCE.find("feval(").unwrap());
    });
}

#[test]
fn feval_rethrow_after_inner_catch_uses_outer_native_exception() {
    runmat_gc::gc_test_context(|| {
        const SOURCE: &str = "function native_exception()\ntry\n error('RunMat:OuterIndirect', 'outer boom');\ncatch\n try\n  error('RunMat:InnerIndirect', 'inner boom');\n catch\n end\n feval('rethrow');\nend\nend\n";
        let error = forced_native_failure("native-lexical-feval-nested@1", SOURCE);
        assert_origin(&error, SOURCE, "RunMat:OuterIndirect", "outer boom");
    });
}

#[test]
fn function_handle_rethrow_inherits_native_lexical_catch() {
    runmat_gc::gc_test_context(|| {
        const SOURCE: &str = "function native_exception()\ntry\n error('RunMat:HandleOrigin', 'handle boom');\ncatch\n callback = @rethrow;\n callback();\nend\nend\n";
        let error = forced_native_failure("native-lexical-handle@1", SOURCE);
        assert_origin(&error, SOURCE, "RunMat:HandleOrigin", "handle boom");
        let span = error.span.as_ref().unwrap();
        assert!(span.offset() + span.len() <= SOURCE.find("callback =").unwrap());
    });
}

#[test]
fn exiting_inner_catch_restores_outer_native_exception() {
    runmat_gc::gc_test_context(|| {
        const SOURCE: &str = "function native_exception()\ntry\n error('RunMat:OuterOrigin', 'outer boom');\ncatch\n try\n  error('RunMat:InnerOrigin', 'inner boom');\n catch\n end\n rethrow();\nend\nend\n";
        let error = forced_native_failure("native-lexical-nested@1", SOURCE);
        assert_origin(&error, SOURCE, "RunMat:OuterOrigin", "outer boom");
        let span = error.span.as_ref().unwrap();
        assert!(span.offset() + span.len() <= SOURCE.find("InnerOrigin").unwrap());
    });
}

#[test]
fn exited_native_catch_cannot_supply_stale_rethrow_authority() {
    runmat_gc::gc_test_context(|| {
        const SOURCE: &str = "function native_exception()\ntry\n error('RunMat:PriorOrigin', 'prior boom');\ncatch\nend\nrethrow();\nend\n";
        let error = forced_native_failure("native-lexical-stale@1", SOURCE);
        assert_eq!(error.identifier(), Some("RunMat:rethrow:NoActiveCatch"));
        assert_ne!(error.identifier(), Some("RunMat:PriorOrigin"));
    });
}

#[test]
fn feval_after_native_catch_cannot_supply_stale_rethrow_authority() {
    runmat_gc::gc_test_context(|| {
        const SOURCE: &str = "function native_exception()\ntry\n error('RunMat:PriorIndirect', 'prior boom');\ncatch\nend\nfeval('rethrow');\nend\n";
        let error = forced_native_failure("native-lexical-feval-stale@1", SOURCE);
        assert_eq!(error.identifier(), Some("RunMat:rethrow:NoActiveCatch"));
        assert_ne!(error.identifier(), Some("RunMat:PriorIndirect"));
    });
}
