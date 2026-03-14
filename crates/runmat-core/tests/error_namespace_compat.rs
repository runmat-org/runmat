// None of these tests use #[wasm_bindgen_test], so they cannot run in the
// browser via wasm-pack. Excluding them from wasm32 avoids compiling a full
// runmat-runtime wasm binary per test file with zero executable tests.
#![cfg(not(target_arch = "wasm32"))]

use futures::executor::block_on;
use runmat_core::{RunError, RunMatSession};
use runmat_gc::gc_test_context;

fn extract_identifier_and_message(error: RunError) -> (Option<String>, String) {
    match error {
        RunError::Semantic(err) => (err.identifier, err.message),
        RunError::Compile(err) => (err.identifier, err.message),
        RunError::Runtime(err) => (
            err.identifier().map(ToString::to_string),
            err.message().to_string(),
        ),
        RunError::Syntax(err) => (None, err.message),
    }
}

fn assert_error_prefix(namespace: &str, code: &str) {
    let mut session = RunMatSession::new().expect("create session");
    session.set_error_namespace(namespace.to_string());

    let (identifier, message) = match block_on(session.execute(code)) {
        Ok(result) => match result.error {
            Some(error) => (
                error.identifier().map(ToString::to_string),
                error.message().to_string(),
            ),
            None => panic!("expected failure for code: {code}"),
        },
        Err(error) => extract_identifier_and_message(error),
    };

    let prefix = format!("{namespace}:");
    let identifier_ok = identifier
        .as_deref()
        .is_some_and(|value| value.starts_with(&prefix));
    let message_ok = message.contains(&prefix);
    assert!(
        identifier_ok || message_ok,
        "expected namespace prefix {prefix} in identifier/message. identifier={identifier:?} message={message:?} code={code}"
    );
}

#[test]
fn error_identifiers_follow_selected_namespace() {
    gc_test_context(|| {
        let cases = [
            // Runtime-generated identifiers should also use active namespace.
            // This one comes from runtime indexing helpers that use static identifiers.
            "a = [1, 2]; a(100);",
            "c = {1}; c{100};",
        ];

        for namespace in ["RunMat", "MATLAB"] {
            for code in cases {
                assert_error_prefix(namespace, code);
            }
        }
    });
}
