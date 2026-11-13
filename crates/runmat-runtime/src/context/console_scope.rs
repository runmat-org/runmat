//! Console-only capture without a strong edge back to foreign adapters.
use super::RuntimeContext;
use crate::{console::ConsoleStream, RuntimeError};

pub(crate) struct RuntimeConsoleScope {
    context: RuntimeContext,
}

impl RuntimeConsoleScope {
    pub(super) fn new(context: RuntimeContext) -> Self {
        Self { context }
    }

    pub(crate) fn record(&self, stream: ConsoleStream, text: &str) -> Result<(), RuntimeError> {
        self.context.record_process_console(stream, text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    #[test]
    fn console_scope_retains_console_without_retaining_foreign_dispatch() {
        let foreign = Rc::new(crate::foreign::ForeignRuntime::new(
            crate::foreign::ForeignPlatform::Native,
        ));
        let weak = Rc::downgrade(&foreign);
        let context =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(
                    super::super::RuntimeServicePorts::default().with_foreign(foreign.clone()),
                );
        let scope = context.console_scope();
        assert!(scope.context.service_ports().foreign().is_none());
        drop(context);
        drop(foreign);
        assert!(
            weak.upgrade().is_none(),
            "retained console must not keep its owning foreign adapter alive"
        );
        scope
            .record(ConsoleStream::Stdout, "retained console")
            .unwrap();
    }
}
