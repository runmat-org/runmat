//! Value-only exact callable fixtures. Tests install their own typed invoker
//! in the same runtime library copy and session that dispatches the call.

use crate::context::RuntimeContext;
use runmat_value::{BoundFunctionBinding, FunctionDefinitionOwner, LocalExport, Value};
use std::sync::Arc;

pub fn exact_fixture_runtime() -> RuntimeContext {
    let runtime = RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    super::inherit_legacy_call_environment(&runtime);
    runtime
}

pub fn exact_local_binding(runtime: &RuntimeContext, function: usize) -> BoundFunctionBinding {
    let owner = Arc::new(FunctionDefinitionOwner::default());
    BoundFunctionBinding::local(
        function,
        runtime.session_lease(),
        FunctionDefinitionOwner::issue(&owner, function),
        Vec::new(),
        LocalExport::Ephemeral,
    )
}

pub fn exact_portable_binding(function: usize, fixture: &str) -> BoundFunctionBinding {
    let ordinal = u32::try_from(function).expect("fixture function ordinal");
    BoundFunctionBinding::portable(
        function,
        runmat_execution::FunctionBindingOrigin {
            program: runmat_execution::Digest::sha256(fixture),
            function: runmat_types::ProgramFunctionId(ordinal),
        },
    )
}

pub fn exact_local_handle(runtime: &RuntimeContext, name: &str, function: usize) -> Value {
    exact_local_value(name, exact_local_binding(runtime, function))
}

pub fn exact_local_value(name: &str, binding: BoundFunctionBinding) -> Value {
    Value::BoundFunctionHandle {
        name: name.into(),
        function: binding.function,
        authority: binding.authority,
        source_role: runmat_types::FunctionSourceRole::Unknown,
        source_file: None,
    }
}

/// Tests retain the old numeric callback oracle only behind a checked exact
/// binding issued by their own RuntimeContext. Production has no such bridge.
pub fn install_bound_forwarder(expected: BoundFunctionBinding) -> super::BoundFunctionInvokerGuard {
    install_bound_forwarders(vec![expected])
}

pub fn install_bound_forwarders(
    expected: Vec<BoundFunctionBinding>,
) -> super::BoundFunctionInvokerGuard {
    super::install_local_bound_function_invoker(std::rc::Rc::new(
        move |binding, arguments, requested_outputs| {
            assert!(expected.contains(&binding), "wrong exact callback owner");
            let invoker = super::current_semantic_function_invoker()
                .expect("numeric fixture invoker in same runtime");
            invoker(binding.function, arguments, requested_outputs)
        },
    ))
}
