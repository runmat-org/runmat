//! Actual-provider witness for reusable managed registration capacity.
#![cfg(not(target_arch = "wasm32"))]

use super::super::backend_types::{WgpuProvider, WgpuProviderOptions};
use runmat_accelerate_api::{
    AccelProvider, GpuTensorStorage, HostNumericDataView, HostNumericTensorView,
};
use runmat_execution::resource::{
    ProviderHostMetadataAdmission, ProviderHostMetadataError, RetainedHostAllocationClaim,
    RetainedHostViewError, RetainedHostViewGrant, RetainedHostViewIssuer,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Debug, Default)]
struct LiveClaims(AtomicUsize);

#[derive(Debug)]
struct Claim {
    live: Arc<LiveClaims>,
}

impl Claim {
    fn new(live: &Arc<LiveClaims>) -> Arc<Self> {
        live.0.fetch_add(1, Ordering::AcqRel);
        Arc::new(Self { live: live.clone() })
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        self.live.0.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Debug)]
struct Issuer {
    live: Arc<LiveClaims>,
}

#[derive(Debug)]
struct ViewGrant {
    bytes: u64,
    issuer: Arc<Issuer>,
    claim: Arc<Claim>,
}

impl RetainedHostViewGrant for ViewGrant {
    fn view_bytes(&self) -> u64 {
        self.bytes
    }

    fn issuer(&self) -> Arc<dyn RetainedHostViewIssuer> {
        self.issuer.clone()
    }

    fn backing_claim(&self) -> Arc<dyn RetainedHostAllocationClaim> {
        self.claim.clone()
    }
}

impl RetainedHostViewIssuer for Issuer {
    fn reserve_view(
        self: Arc<Self>,
        view_bytes: u64,
    ) -> Result<Box<dyn RetainedHostViewGrant>, RetainedHostViewError> {
        Ok(Box::new(ViewGrant {
            bytes: view_bytes,
            claim: Claim::new(&self.live),
            issuer: self,
        }))
    }
}

#[derive(Debug)]
struct Admission {
    live: Arc<LiveClaims>,
    issuer: Arc<Issuer>,
}

impl Admission {
    fn new() -> (Self, Arc<LiveClaims>) {
        let live = Arc::new(LiveClaims::default());
        (
            Self {
                issuer: Arc::new(Issuer { live: live.clone() }),
                live: live.clone(),
            },
            live,
        )
    }
}

impl ProviderHostMetadataAdmission for Admission {
    fn claim_view(
        &self,
        view_bytes: u64,
    ) -> Result<Box<dyn RetainedHostViewGrant>, ProviderHostMetadataError> {
        Ok(Box::new(ViewGrant {
            bytes: view_bytes,
            issuer: self.issuer.clone(),
            claim: Claim::new(&self.live),
        }))
    }

    fn claim_record(
        &self,
        _record_bytes: u64,
    ) -> Result<Arc<dyn RetainedHostAllocationClaim>, ProviderHostMetadataError> {
        Ok(Claim::new(&self.live))
    }
}

fn upload(
    provider: &WgpuProvider,
    admission: &dyn ProviderHostMetadataAdmission,
    value: f32,
) -> runmat_accelerate_api::ProviderExecutionResult<runmat_accelerate_api::GpuTensorHandle> {
    provider.upload_numeric_raw_admitted(
        &HostNumericTensorView {
            data: HostNumericDataView::F32(&[value]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        },
        Some(admission),
    )
}

#[test]
#[ignore = "requires provisioned WGPU adapter; unavailable is failure"]
fn reusable_managed_slots_follow_actual_records_and_recycle_after_retirement() {
    let provider = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("qualification requires a real WGPU adapter");
    let (underlying, live_claims) = Admission::new();
    let admission = provider
        .prepare_reusable_output_host_metadata(Box::new(underlying), 2)
        .expect("two concurrent managed records are pre-admitted");

    let first = upload(&provider, &*admission, 1.0).expect("first managed record");
    let second = upload(&provider, &*admission, 2.0).expect("second managed record");
    assert_eq!(provider.allocations.record_count().unwrap(), 2);

    let refused = upload(&provider, &*admission, 3.0).expect_err(
        "a third concurrent record must not consume capacity reserved for two live records",
    );
    assert!(refused.is_after_effect());
    assert_eq!(provider.allocations.record_count().unwrap(), 2);

    drop(first);
    let third = upload(&provider, &*admission, 3.0)
        .expect("retiring one actual record returns its move-only slot");
    drop(second);
    drop(third);

    for value in [4.0_f32, 5.0, 6.0] {
        let serial = upload(&provider, &*admission, value)
            .expect("total serial registrations may exceed concurrent capacity");
        drop(serial);
    }
    assert_eq!(provider.allocations.record_count().unwrap(), 0);

    let retained = upload(&provider, &*admission, 7.0).expect("record retained past admission");
    assert!(live_claims.0.load(Ordering::Acquire) > 0);
    drop(admission);
    assert!(
        live_claims.0.load(Ordering::Acquire) > 0,
        "the live record retains pool, table and record funding after admission drop"
    );
    assert_eq!(provider.allocations.record_count().unwrap(), 1);

    drop(retained);
    assert_eq!(provider.allocations.record_count().unwrap(), 0);
    assert_eq!(
        live_claims.0.load(Ordering::Acquire),
        0,
        "final physical retirement releases reusable backing funding"
    );
}
