use runmat_runtime::{
    context::{
        HostDocumentRequest, HostInteraction, RuntimeContext, RuntimeHostService,
        RuntimeServiceFuture, RuntimeServicePorts,
    },
    interaction::InteractionResponse,
    RuntimeError,
};
use std::{cell::RefCell, rc::Rc};
#[derive(Default)]
pub struct DocumentHost {
    pub requests: Rc<RefCell<Vec<HostDocumentRequest>>>,
}
impl RuntimeHostService for DocumentHost {
    fn console(
        &self,
        _: runmat_runtime::console::ConsoleStream,
        _: runmat_runtime::console::ConsoleText,
    ) {
    }
    fn warning(&self, _: runmat_runtime::warning_store::RuntimeWarning) {}
    fn interact(
        &self,
        _: HostInteraction,
    ) -> RuntimeServiceFuture<Result<InteractionResponse, RuntimeError>> {
        Box::pin(async {
            Err(RuntimeError::new(
                "Unexpected terminal interaction in document fixture",
            ))
        })
    }
    fn open_document(
        &self,
        request: HostDocumentRequest,
    ) -> RuntimeServiceFuture<Result<(), RuntimeError>> {
        let requests = self.requests.clone();
        Box::pin(async move {
            requests.borrow_mut().push(request);
            Ok(())
        })
    }
}
pub fn context() -> (RuntimeContext, Rc<DocumentHost>) {
    let host = Rc::new(DocumentHost::default());
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_host(host.clone()));
    (context, host)
}
