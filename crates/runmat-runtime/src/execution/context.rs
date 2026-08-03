use std::rc::Rc;

use super::RuntimeExecutionServices;

/// Explicit per-invocation access to execution capabilities.
///
/// This context is passed through Core and the VM. It is intentionally not a
/// process-global or thread-local authority.
#[derive(Clone)]
pub struct InvocationExecutionContext {
    services: Rc<dyn RuntimeExecutionServices>,
}

impl std::fmt::Debug for InvocationExecutionContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InvocationExecutionContext")
            .field("scope_id", &self.services.scope_id())
            .finish()
    }
}

impl InvocationExecutionContext {
    pub fn new(services: Rc<dyn RuntimeExecutionServices>) -> Self {
        Self { services }
    }

    pub fn services(&self) -> &Rc<dyn RuntimeExecutionServices> {
        &self.services
    }
}
