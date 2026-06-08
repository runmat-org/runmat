// None of these tests use #[wasm_bindgen_test], so they cannot run in the
// browser via wasm-pack. Excluding them from wasm32 avoids compiling a full
// runmat-runtime wasm binary per test file with zero executable tests.
#![cfg(not(target_arch = "wasm32"))]

use runmat_core::{RunError, RunMatSession};
use runmat_gc::gc_test_context;
use runmat_parser::CompatMode;

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

    let request = runmat_core::abi::ExecutionRequest::for_source(
        runmat_core::abi::SourceInput::Text {
            name: "<test>".to_string(),
            text: code.to_string(),
        },
        CompatMode::Matlab,
        runmat_core::abi::HostExecutionPolicy::default(),
        session.workspace_handle(),
    );

    let response = futures::executor::block_on(session.execute_request(request));
    let (identifier, message) = match response.result {
        Ok(outcome) => {
            let diag = outcome
                .diagnostics
                .iter()
                .find(|d| d.severity == runmat_core::abi::DiagnosticSeverity::Error)
                .unwrap_or_else(|| panic!("expected failure for code: {code}"));
            (Some(diag.code.clone()), diag.message.clone())
        }
        Err(error) => extract_identifier_and_message(error),
    };

    let prefix = format!("{namespace}:");
    let identifier_ok = identifier
        .as_deref()
        .is_some_and(|value| value.starts_with(&prefix));
    assert!(
        identifier_ok,
        "expected namespace prefix {prefix} in identifier. identifier={identifier:?} message={message:?} code={code}"
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
