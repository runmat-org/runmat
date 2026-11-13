//! Original invocation dispatch for direct numerical reduction witnesses.
use runmat_runtime::context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts};
use std::rc::Rc;

struct CompiledBuiltins;
impl RuntimeBuiltinService for CompiledBuiltins {
    fn bindings_by_name(&self, name: &str) -> Vec<runmat_runtime::builtin::RuntimeBuiltinBinding> {
        runmat_runtime::builtin::with_runtime_builtin_bindings_by_name(name, |bindings| {
            bindings.to_vec()
        })
    }
}

pub fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(CompiledBuiltins)))
}
