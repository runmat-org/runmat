use runmat_runtime::{
    context::{
        HostInteraction, RuntimeContext, RuntimeHostService, RuntimeServiceFuture,
        RuntimeServicePorts,
    },
    interaction::{self, InteractionResponse},
    RuntimeError,
};
use std::{
    rc::Rc,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

struct Adapter(Arc<AtomicUsize>);
impl RuntimeHostService for Adapter {
    fn console(
        &self,
        _: runmat_runtime::console::ConsoleStream,
        _: runmat_runtime::console::ConsoleText,
    ) {
    }
    fn warning(&self, _: runmat_runtime::warning_store::RuntimeWarning) {}
    fn interact(
        &self,
        request: HostInteraction,
    ) -> RuntimeServiceFuture<Result<InteractionResponse, RuntimeError>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        interaction::invoke_installed_async_handler(request).expect("execution handler installed")
    }
}

#[test]
fn scoped_handler_bridge_keeps_queue_precedence_errors_and_no_recursive_dispatch() {
    let calls = Arc::new(AtomicUsize::new(0));
    let host_calls = Arc::new(AtomicUsize::new(0));
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_host(Rc::new(Adapter(host_calls.clone()))),
    );
    futures::executor::block_on(context.scope(async {
        assert!(
            interaction::invoke_installed_async_handler(HostInteraction::KeyPress {
                prompt: String::new()
            })
            .is_none()
        );
        let observed = calls.clone();
        let _handler = interaction::replace_async_handler(Some(Arc::new(move |_| {
            observed.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Err("explicit host error".into()) })
        })));
        interaction::push_queued_response(Ok(InteractionResponse::Line("queued".into())));
        assert_eq!(
            interaction::request_line_async("prompt", true)
                .await
                .unwrap(),
            "queued"
        );
        assert_eq!(host_calls.load(Ordering::SeqCst), 0);
        let error = interaction::request_line_async("prompt", true)
            .await
            .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:interaction:AsyncHandlerError")
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(host_calls.load(Ordering::SeqCst), 1);
        drop(
            interaction::invoke_installed_async_handler(HostInteraction::KeyPress {
                prompt: String::new(),
            })
            .unwrap(),
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "unpolled bridge cannot prompt"
        );
    }));
}
