use super::super::{select_initial_route, SelectedInitialNativeBatch};
use crate::context::provider_operation::ExactProviderOrHostSelection;
use crate::context::{
    RuntimeContext, RuntimePlacementService, RuntimeServicePorts, RuntimeWholeWorkSelection,
};
use runmat_accelerate_api::observation::{
    ProviderObservationCapability, ProviderObservationOverhead, ProviderObservationSupport,
    ProviderOperationReceipt,
};
use runmat_accelerate_api::random::{
    GpuBoundedIntegerDomain, GpuIntegerBatchPhysicalProposal, GpuIntegerBatchProgress,
    GpuIntegerBatchRequest, PROVIDER_GPU_INTEGER_BATCH_OPERATION,
};
use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocation, TestAllocationOwner};
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned,
    HostNumericTensorOwned, HostTensorView, NumericElementType, ProviderExecutionFuture,
    ProviderFeasibility, ProviderFeasibilityQuery, ProviderResourceEstimate,
};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::resource::{
    ProviderHostMetadataAdmission, ProviderHostRegistrationMetadata, RetainedHostAllocationClaim,
};
use runmat_execution::{
    Digest, PlacementDecision, PlacementFeedback, PlacementInvalidation, PlacementPlanRequest,
    PlacementRevision, WholeWorkDecision, WholeWorkDecisionKind, WholeWorkPlacementRequest,
};
use std::cell::Cell;
use std::rc::Rc;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

#[derive(Default)]
pub(super) struct Placement {
    pub(super) observations: Cell<usize>,
}

impl RuntimePlacementService for Placement {
    fn whole_work_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        provider: Digest,
    ) -> Option<PlacementRevision> {
        Some(PlacementRevision {
            program,
            catalog: Digest::sha256(b"integer-batch-protocol-catalog"),
            compiler: Digest::sha256(b"integer-batch-protocol-compiler"),
            authority: runmat_execution::PlacementExecutionAuthorityRevision::Provider(provider),
            policy: Digest::sha256(b"integer-batch-protocol-policy"),
        })
    }

    fn select_whole_work(
        &self,
        request: WholeWorkPlacementRequest,
    ) -> Result<RuntimeWholeWorkSelection, crate::RuntimeError> {
        request.validate().expect("valid integer batch placement");
        let selected = request
            .candidates
            .iter()
            .find(|candidate| candidate.kind.is_provider())
            .expect("the fixture offers its selected provider route");
        Ok(RuntimeWholeWorkSelection {
            decision: WholeWorkDecision {
                signature: request.signature.clone(),
                selected: selected.selected(&request.signature.operation),
                kind: WholeWorkDecisionKind::Unmeasured,
                predicted_total_ns: None,
            },
            probe: None,
            observation_permit: None,
        })
    }

    fn plan(&self, _: PlacementPlanRequest) -> Result<PlacementDecision, crate::RuntimeError> {
        panic!("integer batch protocol uses whole-work selection")
    }

    fn observe(&self, _: PlacementFeedback) -> Result<(), crate::RuntimeError> {
        self.observations.set(self.observations.get() + 1);
        Ok(())
    }

    fn invalidate(&self, _: PlacementInvalidation) {}
}

pub(super) struct ScriptedProvider {
    owner: TestAllocationOwner,
    attempts: Mutex<Vec<u64>>,
    acknowledgements: AtomicUsize,
    provisional: Mutex<Vec<RetirementProbe>>,
    stall_on_call: Option<usize>,
}

impl ScriptedProvider {
    pub(super) fn new(stall_on_call: Option<usize>) -> &'static Self {
        Box::leak(Box::new(Self {
            owner: TestAllocationOwner::new(),
            attempts: Mutex::new(Vec::new()),
            acknowledgements: AtomicUsize::new(0),
            provisional: Mutex::new(Vec::new()),
            stall_on_call,
        }))
    }

    pub(super) fn source(&self) -> TestAllocation {
        self.owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::I32(vec![0]),
                shape: vec![1, 1],
                storage: GpuTensorStorage::Real,
            })
            .expect("fixture source has exact native storage")
    }

    fn physical_proposal() -> GpuIntegerBatchPhysicalProposal {
        GpuIntegerBatchPhysicalProposal {
            output_buffer_bytes: 16,
            accepted_buffer_bytes: 16,
            count_buffer_bytes: 0,
            uniform_buffer_bytes: 0,
            staging_buffer_bytes: 0,
            readback_control_host_bytes: 0,
            pending_view_workspace_bytes: 0,
            uniform_allocations: 0,
            queue_submissions: 0,
        }
    }

    fn first_buffers(
        &self,
        request: GpuIntegerBatchRequest<'_>,
        remaining: u64,
    ) -> GpuIntegerBatchProgress {
        let output = self
            .owner
            .allocate_zeroed(
                request.integer.shape.to_vec(),
                request.integer.class,
                GpuTensorStorage::Real,
            )
            .expect("fixture output has exact native storage");
        let accepted = self
            .owner
            .allocate_zeroed(
                request.integer.shape.to_vec(),
                NumericElementType::U32,
                GpuTensorStorage::Real,
            )
            .expect("fixture mask has exact native storage");
        self.provisional
            .lock()
            .unwrap()
            .extend([output.probe, accepted.probe]);
        GpuIntegerBatchProgress::Pending(runmat_accelerate_api::random::GpuIntegerPending {
            output: output.handle,
            accepted: accepted.handle,
            remaining,
        })
    }

    fn continued_buffers(
        &self,
        request: GpuIntegerBatchRequest<'_>,
        remaining: u64,
    ) -> GpuIntegerBatchProgress {
        let previous = request.pending.expect("continuation owns prior buffers");
        let buffers = runmat_accelerate_api::random::GpuIntegerPending {
            output: previous.output.clone(),
            accepted: previous.accepted.clone(),
            remaining,
        };
        if remaining == 0 {
            GpuIntegerBatchProgress::Complete(buffers)
        } else {
            GpuIntegerBatchProgress::Pending(buffers)
        }
    }

    fn acknowledge_selected_batch(&self) {
        let operation = runmat_accelerate_api::ProviderOperationIdentity::new(
            PROVIDER_GPU_INTEGER_BATCH_OPERATION,
        );
        let recorder = runmat_accelerate_api::observation::current_provider_observation()
            .filter(|recorder| recorder.expects_operation(&operation))
            .expect("selected batch executes under its observation receipt");
        recorder.confirm_provider_execution(self.device_id(), &operation);
        self.acknowledgements.fetch_add(1, Ordering::SeqCst);
    }

    pub(super) fn attempts(&self) -> Vec<u64> {
        self.attempts.lock().unwrap().clone()
    }

    pub(super) fn acknowledgements(&self) -> usize {
        self.acknowledgements.load(Ordering::SeqCst)
    }

    pub(super) fn provisional_retirements(&self) -> Vec<usize> {
        self.provisional
            .lock()
            .unwrap()
            .iter()
            .map(RetirementProbe::retirements)
            .collect()
    }
}

impl AccelProvider for ScriptedProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "integer batch coordinator fixture".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("integer batch coordinator fixture does not upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("integer batch coordinator fixture does not download") })
    }

    fn registered_output_host_metadata(
        &self,
        _: usize,
    ) -> Option<ProviderHostRegistrationMetadata> {
        Some(ProviderHostRegistrationMetadata {
            record_payload_bytes: 0,
            view_payload_bytes: 0,
            lease_payload_bytes: 0,
        })
    }

    fn output_registration_table_upper_bytes(&self, _: usize) -> Option<u64> {
        Some(0)
    }

    fn output_registration_preparation_host_bytes(&self) -> Option<u64> {
        Some(0)
    }

    fn prepare_output_host_metadata(
        &self,
        admission: Box<dyn ProviderHostMetadataAdmission>,
        _: usize,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Box<dyn ProviderHostMetadataAdmission>>
    {
        Ok(admission)
    }

    fn query_feasibility(&self, query: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        if query.operation
            == runmat_accelerate_api::ProviderOperationIdentity::new(
                PROVIDER_GPU_INTEGER_BATCH_OPERATION,
            )
            && query.workload.gpu_integer_batch.is_some()
        {
            let batch = query
                .workload
                .gpu_integer_batch
                .expect("supported fixture query has a batch workload");
            let physical = Self::physical_proposal();
            ProviderFeasibility::supported(ProviderResourceEstimate {
                transient_bytes: physical.peak_bytes(),
                output_bytes: physical.newly_retained_output_bytes(batch),
                dispatches: Some(physical.queue_submissions),
            })
        } else {
            ProviderFeasibility::rejected(
                runmat_accelerate_api::ProviderRejectionCode::UnsupportedOperation,
                "fixture.only_integer_batch",
            )
        }
    }

    fn operation_observation_capability(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> ProviderObservationCapability {
        if query.operation
            == runmat_accelerate_api::ProviderOperationIdentity::new(
                PROVIDER_GPU_INTEGER_BATCH_OPERATION,
            )
        {
            ProviderObservationCapability::Supported(ProviderObservationOverhead {
                submissions: 0,
                requested_peak_buffer_bytes: 0,
                clear_commands: 0,
                map_callbacks: 0,
                wgpu_compute_query: None,
            })
        } else {
            ProviderObservationCapability::Unsupported
        }
    }

    fn begin_operation_observation(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> ProviderObservationSupport {
        if query.operation
            == runmat_accelerate_api::ProviderOperationIdentity::new(
                PROVIDER_GPU_INTEGER_BATCH_OPERATION,
            )
        {
            ProviderObservationSupport::Observed(ProviderOperationReceipt::new())
        } else {
            ProviderObservationSupport::Unsupported
        }
    }

    fn propose_gpu_integer_batch(
        &self,
        request: GpuIntegerBatchRequest<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuIntegerBatchPhysicalProposal>>
    {
        request
            .validate()
            .expect("coordinator emits a valid finite batch");
        Ok(Some(Self::physical_proposal()))
    }

    fn execute_gpu_integer_batch<'a>(
        &'a self,
        request: GpuIntegerBatchRequest<'a>,
        _: Option<&'a dyn ProviderHostMetadataAdmission>,
        _: Option<Arc<dyn RetainedHostAllocationClaim>>,
    ) -> ProviderExecutionFuture<'a, GpuIntegerBatchProgress> {
        let call = {
            let mut attempts = self.attempts.lock().unwrap();
            attempts.push(request.first_attempt);
            attempts.len()
        };
        if self.stall_on_call == Some(call) {
            return Box::pin(std::future::pending());
        }
        Box::pin(async move {
            let progress = match call {
                1 => self.first_buffers(request, 4),
                2 => self.continued_buffers(request, 2),
                3 => self.continued_buffers(request, 0),
                _ => panic!("coordinator emitted an unexpected extra batch"),
            };
            self.acknowledge_selected_batch();
            Ok(progress)
        })
    }
}

pub(super) fn runtime(
    placement: Rc<Placement>,
    cancellation: &CancellationSource,
) -> RuntimeContext {
    RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    )
    .with_service_ports(RuntimeServicePorts::default().with_placement(placement))
}

pub(super) fn selected_first(
    source: &GpuTensorHandle,
    domain: GpuBoundedIntegerDomain,
    shape: &[usize],
) -> SelectedInitialNativeBatch {
    let route = select_initial_route(source, domain, shape, NumericElementType::I32, None)
        .expect("select batch route")
        .expect("native route is available");
    let ExactProviderOrHostSelection::Provider(selected) = route.selection else {
        panic!("fixture placement selects the provider")
    };
    SelectedInitialNativeBatch {
        selected,
        snapshot: route.snapshot,
        physical: route.physical,
    }
}
