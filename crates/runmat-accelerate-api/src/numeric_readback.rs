//! Borrowed, exact-source numeric egress. No compute or raw-device interface escapes.
use crate::{
    AccelNumericDownloadFuture, AccelProvider, ExecutionProviderAdmission,
    ExecutionProviderAuthorization, GpuHandleIdentity, GpuTensorHandle,
    PreparedNumericDownloadStage, ProviderFeasibilityQuery, EXECUTION_PROVIDER_AUTHORIZER,
};
use anyhow::{anyhow, ensure};
use runmat_execution::{resource::RetainedHostAllocationClaim, ResourceByteBound};
use std::{future::Future, pin::Pin, sync::Arc, task::Poll};

/// Registered owners win. A thread-scoped facade is authority only for its own
/// device; the legacy ambient/default fallback never proves source ownership.
/// Re-resolution on every operation revokes a cap when its scope changes/ends.
pub(crate) fn exact_source_owner(device_id: u32) -> Option<&'static dyn AccelProvider> {
    crate::provider_registry::get(device_id).or_else(|| {
        crate::current_thread_provider().filter(|provider| provider.device_id() == device_id)
    })
}

/// Validate transfer authority for this exact registered/scoped owner and live source.
/// Providers use this at their closed copy/map ingress; it grants no ambient
/// privilege and must never replace ordinary compute or raw-device admission.
pub fn authorize_numeric_readback(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> anyhow::Result<()> {
    let owner = exact_source_owner(handle.device_id())
        .ok_or_else(|| anyhow!("numeric readback source has no exact owner"))?;
    ensure!(
        crate::same_provider_instance(owner, provider),
        "numeric readback requires the exact source owner"
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
    let provider = exact_source_owner(handle.device_id())
        .ok_or_else(|| anyhow!("numeric readback source has no exact owner"))?;
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

    /// Immutable owner metadata, not access to its compute interface.
    pub fn owner_precision(&self) -> anyhow::Result<crate::ProviderPrecision> {
        self.authorize()?;
        Ok(self.provider.precision())
    }

    /// Immutable policy evidence, never access to a raw device or provider.
    pub fn owner_has_wgpu_access(&self) -> anyhow::Result<bool> {
        self.authorize()?;
        #[cfg(feature = "wgpu")]
        {
            Ok(self.provider.wgpu_device_access().is_some())
        }
        #[cfg(not(feature = "wgpu"))]
        {
            Ok(false)
        }
    }

    /// Compare the actual retained owner facades only after both source
    /// capabilities are valid in this poll's assignment. No facade escapes.
    pub fn same_owner(&self, other: &NumericReadback<'_>) -> anyhow::Result<bool> {
        self.authorize()?;
        other.authorize()?;
        Ok(crate::same_provider_instance(self.provider, other.provider))
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
            "numeric readback hook requires its exact source owner"
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

    #[test]
    fn scoped_exact_owner_has_no_ambient_fallback_and_revokes_between_polls() {
        let make_provider = || {
            Box::leak(Box::new(PendingOwner {
                owner: TestAllocationOwner::new(),
                polls: AtomicUsize::new(0),
                drops: AtomicUsize::new(0),
            }))
        };
        let provider = make_provider();
        let other = make_provider();
        let allocation = provider
            .owner
            .allocate_zeroed(
                vec![2],
                crate::NumericElementType::F64,
                crate::GpuTensorStorage::Real,
            )
            .unwrap();
        let scoped = crate::ThreadProviderGuard::set(Some(provider));
        let readback = numeric_readback_for_handle(&allocation.handle).unwrap();
        assert_eq!(readback.owner_precision().unwrap(), provider.precision());
        let mut future = std::pin::pin!(readback.download_numeric());
        let waker = futures::task::noop_waker();
        let mut cx = std::task::Context::from_waker(&waker);
        assert!(future.as_mut().poll(&mut cx).is_pending());
        let wrong_scope = crate::ThreadProviderGuard::set(Some(other));
        assert!(numeric_readback_for_handle(&allocation.handle).is_err());
        assert!(matches!(future.as_mut().poll(&mut cx), Poll::Ready(Err(_))));
        assert_eq!(provider.polls.load(Ordering::SeqCst), 1);
        assert_eq!(provider.drops.load(Ordering::SeqCst), 1);
        drop(wrong_scope);

        // Inject a distinct registered facade for this test-only namespace to
        // prove registration precedence/revocation, without ever executing its
        // payload methods. No shipping owner is replaced or fixture hidden by
        // global registration just to make ordinary thread-scoped egress work.
        struct Replacement(u32);
        impl AccelProvider for Replacement {
            fn device_id(&self) -> u32 {
                self.0
            }
            fn device_info(&self) -> String {
                "registered replacement proof".into()
            }
            fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
                unreachable!()
            }
            fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
                unreachable!()
            }
        }
        let replacement = Box::leak(Box::new(Replacement(provider.device_id())));
        crate::provider_registry::register(provider.device_id(), replacement);
        assert!(readback.owner_precision().is_err());
        assert!(crate::same_provider_instance(
            exact_source_owner(provider.device_id()).unwrap(),
            replacement
        ));
        drop(scoped);
        assert!(readback.owner_has_wgpu_access().is_err());

        let orphan = other
            .owner
            .allocate_zeroed(
                vec![1],
                crate::NumericElementType::F64,
                crate::GpuTensorStorage::Real,
            )
            .unwrap();
        let cap = {
            let _scope = crate::ThreadProviderGuard::set(Some(other));
            numeric_readback_for_handle(&orphan.handle).unwrap()
        };
        assert!(cap.owner_precision().is_err());
    }
}
