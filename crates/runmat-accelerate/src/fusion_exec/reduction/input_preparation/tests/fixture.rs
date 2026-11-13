//! Actual fixture leases and counted provider hooks; no fabricated GPU handle.
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, GpuTensorStorage, HostNumericTensorView,
    HostTensorView, ProviderExecutionError, ProviderExecutionResult, ProviderFeasibilityQuery,
    ProviderPrecision, ProviderRealElementwiseOutputRoute, ProviderRealElementwisePhysicalProposal,
    ProviderRealElementwiseRequest, ProviderUploadPhysicalEvidence,
};
use runmat_execution::ResourceByteBound;
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Mutex,
};

pub(super) struct Owner {
    pub allocations: TestAllocationOwner,
    reported_device: Option<u32>,
    pub evidence_queries: AtomicUsize,
    pub proposal_queries: AtomicUsize,
    pub upload_calls: AtomicUsize,
    evidence_requested: AtomicU64,
    fail_upload_call: AtomicUsize,
    pub probes: Mutex<Vec<RetirementProbe>>,
}

impl Owner {
    pub fn new(evidence_requested: u64) -> &'static Self {
        Self::with_reported_device(evidence_requested, None)
    }

    pub fn same_device_facade(evidence_requested: u64, device_id: u32) -> &'static Self {
        Self::with_reported_device(evidence_requested, Some(device_id))
    }

    fn with_reported_device(
        evidence_requested: u64,
        reported_device: Option<u32>,
    ) -> &'static Self {
        let owner: &'static Self = Box::leak(Box::new(Self {
            allocations: TestAllocationOwner::new(),
            reported_device,
            evidence_queries: AtomicUsize::new(0),
            proposal_queries: AtomicUsize::new(0),
            upload_calls: AtomicUsize::new(0),
            evidence_requested: AtomicU64::new(evidence_requested),
            fail_upload_call: AtomicUsize::new(0),
            probes: Mutex::new(Vec::new()),
        }));
        if reported_device.is_none() {
            // SAFETY: this leaked owner and its TestAllocationOwner-minted
            // unique device namespace live for the process/worker lifetime.
            // A same-device wrong facade is deliberately never registered.
            unsafe { runmat_accelerate_api::register_device_provider(owner) };
        }
        owner
    }

    pub fn evidence(&self) -> ProviderUploadPhysicalEvidence {
        ProviderUploadPhysicalEvidence {
            output_requested_bytes: self.evidence_requested.load(Ordering::SeqCst),
            host_workspace_bytes: ResourceByteBound::Known(0),
            provider_scratch_bytes: ResourceByteBound::Known(0),
            completion_queue_slots: 1,
        }
    }

    pub fn fail_upload(&self, call: usize) {
        self.fail_upload_call.store(call, Ordering::SeqCst);
    }

    pub fn evidence_count(&self) -> usize {
        self.evidence_queries.load(Ordering::SeqCst)
    }

    pub fn upload_count(&self) -> usize {
        self.upload_calls.load(Ordering::SeqCst)
    }
}

impl AccelProvider for Owner {
    fn device_id(&self) -> u32 {
        self.reported_device
            .unwrap_or_else(|| self.allocations.device_id())
    }

    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }

    fn device_info(&self) -> String {
        "typed-input-owner-fixture".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("typed preparation must use numeric raw upload")
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        let call = self.upload_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail_upload_call.load(Ordering::SeqCst) == call {
            return Err(ProviderExecutionError::from(anyhow::anyhow!(
                "injected later upload failure"
            )));
        }
        let allocation = self
            .allocations
            .allocate_zeroed(
                host.shape.to_vec(),
                host.element_type(),
                GpuTensorStorage::Real,
            )
            .map_err(ProviderExecutionError::from)?;
        self.probes.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }

    fn upload_physical_evidence(
        &self,
        _: &ProviderFeasibilityQuery,
    ) -> Option<ProviderUploadPhysicalEvidence> {
        self.evidence_queries.fetch_add(1, Ordering::SeqCst);
        Some(self.evidence())
    }

    fn propose_real_elementwise(
        &self,
        _: &ProviderRealElementwiseRequest,
    ) -> ProviderExecutionResult<Option<ProviderRealElementwisePhysicalProposal>> {
        self.proposal_queries.fetch_add(1, Ordering::SeqCst);
        Ok(Some(ProviderRealElementwisePhysicalProposal {
            output_route: ProviderRealElementwiseOutputRoute::FreshAllocation,
            output_buffer_bytes: 16,
            workgroup_size: 8,
            chunk_capacity: 8,
            uniform_allocations: 0,
            uniform_buffer_bytes: 0,
            dispatches: 1,
            queue_submissions: 1,
            pipeline_key: Some(1),
        }))
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("typed preparation must not download")
    }
}
