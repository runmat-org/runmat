use super::{
    HostDocumentRequest, RuntimeContext, RuntimeHostService, RuntimeServiceFuture,
    RuntimeServicePorts,
};
use crate::{interaction::InteractionResponse, RuntimeError};
use std::{cell::RefCell, rc::Rc};

#[derive(Default)]
pub(crate) struct DocumentRecorder {
    pub requests: Rc<RefCell<Vec<HostDocumentRequest>>>,
}
impl RuntimeHostService for DocumentRecorder {
    fn console(&self, _: crate::console::ConsoleStream, _: String) {}
    fn warning(&self, _: crate::warning_store::RuntimeWarning) {}
    fn interact(
        &self,
        _: super::HostInteraction,
    ) -> RuntimeServiceFuture<Result<InteractionResponse, RuntimeError>> {
        Box::pin(async { Err(super::host_document::unavailable()) })
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
pub(crate) fn recording_context() -> (RuntimeContext, Rc<DocumentRecorder>) {
    let recorder = Rc::new(DocumentRecorder::default());
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_host(recorder.clone()));
    context.set_runmat_extensions_enabled(crate::compatibility::runmat_extensions_enabled());
    crate::user_functions::inherit_legacy_call_environment(&context);
    (context, recorder)
}
