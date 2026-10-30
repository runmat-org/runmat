//! Borrowed, exact-source numeric egress. No compute or raw-device interface escapes.
use crate::{
    AccelNumericDownloadFuture, AccelProvider, ExecutionProviderAdmission,
    ExecutionProviderAuthorization, GpuHandleIdentity, GpuTensorHandle,
    PreparedNumericDownloadStage, ProviderFeasibilityQuery, EXECUTION_PROVIDER_AUTHORIZER,
};
use anyhow::{anyhow, ensure};
use runmat_execution::{resource::RetainedHostAllocationClaim, ResourceByteBound};
use std::{future::Future, pin::Pin, sync::Arc, task::Poll};

/// Validate transfer authority for this exact registered owner and live source.
/// Providers use this at their closed copy/map ingress; it grants no ambient
/// privilege and must never replace ordinary compute or raw-device admission.
pub fn authorize_numeric_readback(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> anyhow::Result<()> {
    let owner = crate::provider_registry::get(handle.device_id())
        .ok_or_else(|| anyhow!("numeric readback source has no registered owner"))?;
    ensure!(
        crate::same_provider_instance(owner, provider),
        "numeric readback requires the exact registered source owner"
    );
    handle.allocation_lease().validate_live()?;
    handle.view().validate(handle.physical())?;
    if let Some(authorizer) = EXECUTION_PROVIDER_AUTHORIZER.get() {
        ensure!(
            authorizer(ExecutionProviderAdmission::NumericReadback { provider, handle })
                != ExecutionProviderAuthorization::Denied,
            "numeric readback source is outside the current execution assignment"
        );
    }
    Ok(())
}

/// Only the numeric download surface for one immutable borrowed tensor view.
/// It neither clones payload/shape metadata nor exposes its provider facade.
pub struct NumericReadback<'h> {
    provider: &'static dyn AccelProvider,
    source: &'h GpuTensorHandle,
    identity: GpuHandleIdentity,
}

pub fn numeric_readback_for_handle(
    handle: &GpuTensorHandle,
) -> anyhow::Result<NumericReadback<'_>> {
    let provider = crate::provider_registry::get(handle.device_id())
        .ok_or_else(|| anyhow!("numeric readback source has no registered owner"))?;
    authorize_numeric_readback(provider, handle)?;
    Ok(NumericReadback {
        provider,
        source: handle,
        identity: handle.view_identity(),
    })
}

impl<'h> NumericReadback<'h> {
    pub fn source(&self) -> &'h GpuTensorHandle {
        self.source
    }

    pub fn device_id(&self) -> u32 {
        self.source.device_id()
    }

    fn authorize(&self) -> anyhow::Result<()> {
        ensure!(
            self.source.view_identity() == self.identity,
            "numeric readback source view changed"
        );
        authorize_numeric_readback(self.provider, self.source)
    }

    /// Validate a provider hook without exposing the retained facade.
    pub fn authorize_provider<P: AccelProvider + ?Sized>(
        &self,
        provider: &P,
    ) -> anyhow::Result<()> {
        ensure!(
            provider.device_id() == self.provider.device_id()
                && std::ptr::addr_eq(
                    provider as *const P as *const (),
                    self.provider as *const dyn AccelProvider as *const ()
                ),
            "numeric readback hook requires its registered source owner"
        );
        self.authorize()
    }

    pub fn download_numeric_host_workspace_bound(&self) -> anyhow::Result<ResourceByteBound> {
        self.authorize()?;
        Ok(self
            .provider
            .download_numeric_host_workspace_bound(self.source))
    }

    pub fn download_numeric_provider_scratch_bound(&self) -> anyhow::Result<ResourceByteBound> {
        self.authorize()?;
        Ok(self
            .provider
            .download_numeric_provider_scratch_bound(self.source))
    }

    pub fn download_numeric_completion_host_bound(&self) -> anyhow::Result<ResourceByteBound> {
        self.authorize()?;
        Ok(self
            .provider
            .download_numeric_completion_host_bound(self.source))
    }

    pub fn download_numeric_queue_slots(&self) -> anyhow::Result<u32> {
        self.authorize()?;
        Ok(self.provider.download_numeric_queue_slots(self.source))
    }

    pub fn prepare_numeric_download_stage(
        &self,
        query: &ProviderFeasibilityQuery,
        completion_claim: Arc<dyn RetainedHostAllocationClaim>,
    ) -> anyhow::Result<PreparedNumericDownloadStage> {
        self.authorize()?;
        ensure!(
            query.matches_numeric_download(self.source),
            "numeric readback query differs from its source"
        );
        let stage = self
            .provider
            .prepare_numeric_readback_stage(self, query, completion_claim)?;
        ensure!(
            stage.matches_source(self.source),
            "numeric readback stage differs from its source"
        );
        Ok(stage)
    }

    pub fn download_numeric(&self) -> NumericReadbackFuture<'_, 'h> {
        self.download(None)
    }

    pub fn download_numeric_admitted_stage(
        &self,
        stage: PreparedNumericDownloadStage,
    ) -> NumericReadbackFuture<'_, 'h> {
        self.download(Some(stage))
    }

    fn download(
        &self,
        stage: Option<PreparedNumericDownloadStage>,
    ) -> NumericReadbackFuture<'_, 'h> {
        NumericReadbackFuture {
            readback: self,
            stage,
            inner: None,
            done: false,
        }
    }
}

/// Stack-pinnable authorization wrapper. The only boxed future is the
/// provider's existing download future, already covered by its callback bound.
/// No additional allocation or copied source metadata is introduced here.
pub struct NumericReadbackFuture<'a, 'h> {
    readback: &'a NumericReadback<'h>,
    stage: Option<PreparedNumericDownloadStage>,
    inner: Option<AccelNumericDownloadFuture<'a>>,
    done: bool,
}

impl Future for NumericReadbackFuture<'_, '_> {
    type Output = anyhow::Result<crate::HostNumericTensorOwned>;

    fn poll(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        assert!(
            !this.done,
            "numeric readback future polled after completion"
        );
        // Invocation scopes can change between polls. Never retain a prior
        // scope's permission across a pending map/copy future.
        if let Err(error) = this.readback.authorize() {
            this.inner.take();
            this.stage.take();
            this.done = true;
            return Poll::Ready(Err(error));
        }
        if this.inner.is_none() {
            this.inner = Some(if let Some(stage) = this.stage.take() {
                if !stage.matches_source(this.readback.source) {
                    this.done = true;
                    return Poll::Ready(Err(anyhow!(
                        "numeric readback stage differs from its source"
                    )));
                }
                this.readback
                    .provider
                    .download_numeric_readback_admitted_stage(this.readback, stage)
            } else {
                this.readback
                    .provider
                    .download_numeric_readback(this.readback)
            });
        }
        let result = this
            .inner
            .as_mut()
            .expect("download future initialized")
            .as_mut()
            .poll(cx);
        if result.is_ready() {
            this.inner.take();
            this.done = true;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_support::TestAllocationOwner, AccelDownloadFuture, HostTensorView};
    use std::future::poll_fn;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct PendingOwner {
        owner: TestAllocationOwner,
        polls: AtomicUsize,
        drops: AtomicUsize,
    }

    impl AccelProvider for PendingOwner {
        fn device_id(&self) -> u32 {
            self.owner.device_id()
        }
        fn device_info(&self) -> String {
            "pending readback fixture".into()
        }
        fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            anyhow::bail!("readback fixture cannot upload")
        }
        fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async { anyhow::bail!("readback must use native numeric download") })
        }
        fn download_numeric<'a>(
            &'a self,
            _: &'a GpuTensorHandle,
        ) -> AccelNumericDownloadFuture<'a> {
            struct Retained<'a>(&'a AtomicUsize);
            impl Drop for Retained<'_> {
                fn drop(&mut self) {
                    self.0.fetch_add(1, Ordering::SeqCst);
                }
            }
            let retained = Retained(&self.drops);
            Box::pin(async move {
                let _retained = retained;
                poll_fn(|_| {
                    self.polls.fetch_add(1, Ordering::SeqCst);
                    Poll::Pending
                })
                .await
            })
        }
    }

    #[test]
    fn exact_owner_readback_rechecks_liveness_each_poll_and_retires_pending_future() {
        let provider = Box::leak(Box::new(PendingOwner {
            owner: TestAllocationOwner::new(),
            polls: AtomicUsize::new(0),
            drops: AtomicUsize::new(0),
        }));
        let allocation = provider
            .owner
            .allocate_zeroed(
                vec![2],
                crate::NumericElementType::F64,
                crate::GpuTensorStorage::Real,
            )
            .unwrap();
        assert!(numeric_readback_for_handle(&allocation.handle).is_err());
        // SAFETY: this uniquely identified fixture facade is retained for the process.
        unsafe { crate::register_device_provider(provider) };
        let impostor = PendingOwner {
            owner: TestAllocationOwner::new(),
            polls: AtomicUsize::new(0),
            drops: AtomicUsize::new(0),
        };
        assert!(authorize_numeric_readback(&impostor, &allocation.handle).is_err());
        let readback = numeric_readback_for_handle(&allocation.handle).unwrap();
        assert_eq!(
            readback.source().view_identity(),
            allocation.handle.view_identity()
        );
        let mut future = std::pin::pin!(readback.download_numeric());
        let waker = futures::task::noop_waker();
        let mut cx = std::task::Context::from_waker(&waker);
        assert!(future.as_mut().poll(&mut cx).is_pending());
        assert_eq!(provider.polls.load(Ordering::SeqCst), 1);
        allocation.probe.mark_unavailable();
        assert!(matches!(future.as_mut().poll(&mut cx), Poll::Ready(Err(_))));
        assert_eq!(provider.polls.load(Ordering::SeqCst), 1);
        assert_eq!(provider.drops.load(Ordering::SeqCst), 1);
        assert!(readback.download_numeric_host_workspace_bound().is_err());
    }
}
