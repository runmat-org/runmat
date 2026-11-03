//! Session-local async host port for the three language FEA domain roots.
use super::fea_lease::{denial_from_js, denied_to_js};
use super::fea_run_context::{ActiveFeaLease, LocalFeaRunContext};
use runmat_runtime::analysis::fea_run_admission::{
    AsyncFeaRunAdmission, AsyncFeaRunLease, FeaRunDenied, FeaRunOutcome, FeaRunRequest, FeaRunScope,
};
use runmat_runtime::context::RuntimeServiceFuture;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
use wasm_bindgen::JsValue;

pub(super) struct BrowserFeaAdmission {
    pub host: Rc<RefCell<Option<JsValue>>>,
    pub active: Rc<RefCell<Option<ActiveFeaLease>>>,
    pub pending: Rc<Cell<bool>>,
}
impl AsyncFeaRunAdmission for BrowserFeaAdmission {
    fn admit(
        &self,
        request: FeaRunRequest,
    ) -> RuntimeServiceFuture<Result<Box<dyn AsyncFeaRunLease>, FeaRunDenied>> {
        let started = start(
            self.host.clone(),
            self.active.clone(),
            self.pending.clone(),
            request,
        );
        Box::pin(async move {
            started
                .await
                .map(|lease| lease as Box<dyn AsyncFeaRunLease>)
        })
    }
}

/// Own start independently of its caller. If VM cancellation drops the
/// receiver, the late lease's Drop still retires and settles that root once.
pub(super) fn start(
    host: Rc<RefCell<Option<JsValue>>>,
    active: Rc<RefCell<Option<ActiveFeaLease>>>,
    pending: Rc<Cell<bool>>,
    request: FeaRunRequest,
) -> RuntimeServiceFuture<Result<Box<BrowserFeaLease>, FeaRunDenied>> {
    let host = host.borrow().clone();
    if host.is_none() {
        return Box::pin(async { Err(FeaRunDenied::SignInRequired) });
    }
    if pending.get() || active.borrow().is_some() {
        return Box::pin(async { Err(FeaRunDenied::LicenseInvalid) });
    }
    pending.set(true);
    let (sender, receiver) = futures::channel::oneshot::channel();
    wasm_bindgen_futures::spawn_local(async move {
        let result = LocalFeaRunContext::admit_request(request, host)
            .await
            .map_err(denial_from_js)
            .map(|context| {
                *active.borrow_mut() = Some(context.active_lease());
                Box::new(BrowserFeaLease {
                    context: Some(context),
                    active,
                })
            });
        pending.set(false);
        // An undelivered successful value drops its original lease owner.
        let _ = sender.send(result);
    });
    Box::pin(async move {
        receiver
            .await
            .unwrap_or(Err(FeaRunDenied::ServerUnreachable))
    })
}

pub(super) struct BrowserFeaLease {
    context: Option<LocalFeaRunContext>,
    active: Rc<RefCell<Option<ActiveFeaLease>>>,
}
impl std::ops::Deref for BrowserFeaLease {
    type Target = LocalFeaRunContext;
    fn deref(&self) -> &Self::Target {
        self.context.as_ref().expect("live FEA lease")
    }
}
impl BrowserFeaLease {
    pub(super) fn finish(
        mut self: Box<Self>,
        outcome: FeaRunOutcome,
    ) -> RuntimeServiceFuture<Result<(), JsValue>> {
        settle_context(
            self.context.take().expect("one FEA settlement"),
            self.active.clone(),
            outcome,
        )
    }
}
impl AsyncFeaRunLease for BrowserFeaLease {
    fn scope(&self) -> &FeaRunScope {
        self.context.as_ref().expect("live FEA lease").scope()
    }
    fn settle(
        self: Box<Self>,
        outcome: FeaRunOutcome,
    ) -> RuntimeServiceFuture<Result<(), FeaRunDenied>> {
        let settled = self.finish(outcome);
        Box::pin(async move { settled.await.map_err(denial_from_js) })
    }
}
impl Drop for BrowserFeaLease {
    fn drop(&mut self) {
        if let Some(context) = self.context.take() {
            // An abandoned async builtin must retire its owned work before
            // the UI host sends the single cancellation settlement.
            drop(settle_context(
                context,
                self.active.clone(),
                FeaRunOutcome::Cancelled,
            ));
        }
    }
}

fn settle_context(
    context: LocalFeaRunContext,
    active: Rc<RefCell<Option<ActiveFeaLease>>>,
    outcome: FeaRunOutcome,
) -> RuntimeServiceFuture<Result<(), JsValue>> {
    let (sender, receiver) = futures::channel::oneshot::channel();
    // Settlement remains owned even if cancellation drops the acknowledgement.
    wasm_bindgen_futures::spawn_local(async move {
        let checked = context.check().await;
        let settled = context.settle(outcome).await;
        let retired = settled
            .as_ref()
            .err()
            .is_none_or(super::fea_lease::workers_retired);
        if retired {
            active.borrow_mut().take();
        }
        // Missing physical proof keeps this session's original slot owned.
        let _ = sender.send(super::fea_lease::merge_js_lease_error(
            checked,
            settled.err(),
        ));
    });
    Box::pin(async move {
        receiver
            .await
            .unwrap_or_else(|_| Err(denied_to_js(FeaRunDenied::ServerUnreachable)))
    })
}
