//! Exact callable fixtures. Tests install their own typed invoker in the same
//! runtime library copy and session that dispatches the call.

use crate::context::RuntimeContext;
use runmat_value::{BoundFunctionBinding, FunctionDefinitionOwner, LocalExport, Value};
use std::sync::Arc;

#[path = "test_support/named.rs"]
mod named;
pub use named::*;

/// Publish a semantic fixture's result before its invocation scope retires.
pub fn rooted_single_value_sequence(
    value: Value,
) -> Result<crate::execution::RootedValueSequence, crate::RuntimeError> {
    let sequence = crate::sequence::single_value_sequence(value)?;
    crate::execution::RootedValueSequence::new(sequence).map_err(|error| error.into_runtime_error())
}

pub fn exact_fixture_runtime() -> RuntimeContext {
    let runtime = RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    super::inherit_legacy_call_environment(&runtime);
    runtime
}

#[cfg(not(target_arch = "wasm32"))]
pub type ExactLifecycleSupervisor = crate::context::RuntimeLocalTaskSupervisor;
#[cfg(target_arch = "wasm32")]
pub type ExactLifecycleSupervisor = crate::context::BrowserLocalTaskSupervisor;

pub fn exact_hosted_fixture_runtime() -> (
    RuntimeContext,
    std::rc::Rc<ExactLifecycleSupervisor>,
    crate::context::RuntimeLifecycleHost,
) {
    let runtime = exact_fixture_runtime();
    let supervisor = std::rc::Rc::new(ExactLifecycleSupervisor::new());
    let ports = runtime
        .service_ports()
        .clone()
        .with_local_tasks(supervisor.submission_port(runtime.local_identity()));
    let runtime = runtime.with_service_ports(ports);
    let host = crate::context::RuntimeLifecycleHost::new(&runtime, supervisor.clone())
        .expect("exact lifecycle host");
    (runtime, supervisor, host)
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
    Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
        runmat_value::BoundFunctionHandleData {
            name: name.into(),
            function: binding.function,
            authority: binding.authority,
            source_role: runmat_types::FunctionSourceRole::Unknown,
            source_file: None,
        },
    ))
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
        move |runtime, binding, arguments, requested_outputs, _lexical_environment| {
            assert!(expected.contains(&binding), "wrong exact callback owner");
            let invoker = super::current_semantic_function_invoker()
                .expect("numeric fixture invoker in same runtime");
            invoker(
                Some(runtime),
                binding.function,
                arguments,
                requested_outputs,
            )
        },
    ))
    .expect("test bound-forwarder projection")
}
