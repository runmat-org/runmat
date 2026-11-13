//! Session-local async host port for the three language FEA domain roots.
use super::fea_lease::{denial_from_js, denied_to_js};
use super::fea_run_context::{ActiveFeaLease, LocalFeaRunContext};
use runmat_runtime::analysis::fea_run_admission::{
    AsyncFeaRunAdmission, AsyncFeaRunLease, FeaRunDenied, FeaRunOutcome, FeaRunRequest, FeaRunScope,
};
use runmat_runtime::context::RuntimeServiceFuture;
use runmat_simulation_usage::OperationCount;
use std::{
    cell::{Cell, RefCell},
    rc::{Rc, Weak},
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
    fn request_grant(
        &self,
        request: FeaRunRequest,
        usage: runmat_runtime::analysis::fea_run_admission::FeaRunUsage,
        required: OperationCount,
    ) -> RuntimeServiceFuture<
        Result<runmat_runtime::analysis::fea_run_admission::FeaRunBudget, FeaRunDenied>,
    > {
        let host = self.host.borrow().clone();
        let active = self.active.borrow().clone();
        Box::pin(async move {
            let host = host.ok_or(FeaRunDenied::SignInRequired)?;
            let active = active
                .filter(|active| active.run_id == request.run_id)
                .ok_or(FeaRunDenied::LicenseInvalid)?;
            if active.control.usage()?.completed_ops != usage.completed_ops {
                return Err(FeaRunDenied::LicenseInvalid);
            }
            let request_js = crate::wire::json_contract_to_js(&request)
                .map_err(|_| FeaRunDenied::LicenseInvalid)?;
            let usage_js = crate::wire::json_contract_to_js(
                &serde_json::json!({"completedOps": usage.completed_ops}),
            )
            .map_err(|_| FeaRunDenied::LicenseInvalid)?;
            super::fea_lease::call_host(&host, "check", &request_js, &usage_js)
                .await
                .map_err(denial_from_js)?;
            let required_js = crate::wire::json_contract_to_js(&required)
                .map_err(|_| FeaRunDenied::LicenseInvalid)?;
            let state: super::fea_lease::LeaseState = serde_wasm_bindgen::from_value(
                super::fea_lease::call_host(&host, "grant", &request_js, &required_js)
                    .await
                    .map_err(denial_from_js)?,
            )
            .map_err(|_| FeaRunDenied::LicenseInvalid)?;
            if state.lease.lease_id != active.lease_id
                || state.root_started_at_unix_millis != active.root_started_at_unix_millis()
            {
                return Err(FeaRunDenied::LicenseInvalid);
            }
            let budget = state.lease.budget().map_err(denial_from_js)?;
            active.control.renew(budget)?;
            Ok(budget)
        })
    }
}

/// An authenticated child uses its own ledger and asks the UI coordinator for
/// incremental capacity. This port never starts or settles another root.
pub(super) struct BrowserWorkerFeaAdmission {
    pub context: Weak<LocalFeaRunContext>,
}

pub(super) struct BrowserNumericalCapacity {
    runtime: runmat_runtime::context::RuntimeContext,
    control: runmat_runtime::analysis::fea_run_admission::FeaRunControl,
}
impl BrowserNumericalCapacity {
    pub fn new(context: &Rc<LocalFeaRunContext>) -> Self {
        use runmat_runtime::context::{RuntimeContext, RuntimeServicePorts};
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            RuntimeServicePorts::default().with_fea_run_admission(Rc::new(
                BrowserWorkerFeaAdmission {
                    context: Rc::downgrade(context),
                },
            )),
        );
        Self {
            runtime,
            control: context.scope().control(),
        }
    }
}
impl runmat_simulation_usage::AsyncCapacityOwner for BrowserNumericalCapacity {
    fn ensure_capacity(
        &self,
        additional: OperationCount,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<(), runmat_simulation_usage::LedgerError>> + '_,
        >,
    > {
        Box::pin(async move {
            use runmat_simulation_usage::{LedgerError, UsageContext};
            let usage = UsageContext::capture().ok_or(LedgerError::GrantUnavailable)?;
            runmat_runtime::analysis::fea_run_admission::ensure_available_capacity(
                &usage,
                additional,
                Some(&self.runtime),
            )
            .await
            .map_err(|denied| {
                self.control.deny(denied.clone());
                let cause = match denied {
                    FeaRunDenied::Accounting(cause) => cause,
                    _ => LedgerError::GrantUnavailable,
                };
                cause
            })
        })
    }
}
impl AsyncFeaRunAdmission for BrowserWorkerFeaAdmission {
    fn admit(
        &self,
        _: FeaRunRequest,
    ) -> RuntimeServiceFuture<Result<Box<dyn AsyncFeaRunLease>, FeaRunDenied>> {
        Box::pin(async { Err(FeaRunDenied::LicenseInvalid) })
    }
    fn request_grant(
        &self,
        request: FeaRunRequest,
        usage: runmat_runtime::analysis::fea_run_admission::FeaRunUsage,
        required: OperationCount,
    ) -> RuntimeServiceFuture<
        Result<runmat_runtime::analysis::fea_run_admission::FeaRunBudget, FeaRunDenied>,
    > {
        let context = self.context.upgrade();
        Box::pin(async move {
            let context = context.ok_or(FeaRunDenied::LicenseInvalid)?;
            if context.request() != &request
                || context.scope().usage()?.completed_ops != usage.completed_ops
            {
                return Err(FeaRunDenied::LicenseInvalid);
            }
            context
                .acquire_worker_grant(required)
                .await
                .map_err(denial_from_js)
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
