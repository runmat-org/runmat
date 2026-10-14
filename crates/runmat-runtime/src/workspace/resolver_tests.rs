use super::*;
use crate::context::{RuntimeContext, RuntimeContextGuard};
use crate::execution::RuntimeExecutionService;
use crate::runtime_error::{build_runtime_error, semantic_error_with_source};
use std::error::Error;
use std::rc::Rc;

fn missing(_name: &str) -> Option<Value> {
    None
}

fn empty_snapshot() -> Vec<(String, Value)> {
    Vec::new()
}

fn empty_globals() -> Vec<String> {
    Vec::new()
}

fn failing_mutation() -> crate::RuntimeError {
    semantic_error_with_source(
        "RunMat:Workspace:LexicalPublication",
        "committed lexical mutation failed",
        build_runtime_error("exact lexical cell failure")
            .with_identifier("RunMat:Lexical:CellClosed")
            .build(),
    )
}

fn failing_assign(_name: &str, _value: Value) -> Result<(), crate::RuntimeError> {
    Err(failing_mutation())
}

fn failing_clear() -> Result<(), crate::RuntimeError> {
    Err(failing_mutation())
}

fn failing_remove(_name: &str) -> Result<(), crate::RuntimeError> {
    Err(failing_mutation())
}

#[test]
fn registered_resolver_mutations_preserve_exact_typed_cause() {
    let _lock = test_guard();
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let _scope = RuntimeContextGuard::enter(context);
    register_workspace_resolver(WorkspaceResolver {
        lookup: missing,
        snapshot: empty_snapshot,
        globals: empty_globals,
        assign: Some(failing_assign),
        clear: Some(failing_clear),
        remove: Some(failing_remove),
    });

    for error in [
        try_assign("x", Value::Num(1.0)).unwrap_err(),
        try_clear().unwrap_err(),
        try_remove("x").unwrap_err(),
        assign("x", Value::Num(1.0)).unwrap_err(),
    ] {
        assert_eq!(
            error.identifier(),
            Some("RunMat:Workspace:LexicalPublication")
        );
        let source = error
            .source()
            .unwrap()
            .downcast_ref::<crate::RuntimeError>()
            .unwrap();
        assert_eq!(source.identifier(), Some("RunMat:Lexical:CellClosed"));
    }
}
