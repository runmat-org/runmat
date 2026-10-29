//! Actual fixture allocation leases; no device execution or numerical claim.
use super::*;
use crate::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, HostTensorView,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Clone, Copy)]
pub(super) enum Fault {
    BeforeUnsupported,
    AfterCause,
    Pending,
    Alias,
    WrongClass,
    WrongReceipt,
}
pub(super) struct Fixture {
    pub owner: TestAllocationOwner,
    pub decline_at: AtomicUsize,
    pub calls: AtomicUsize,
    pub fault: Mutex<Option<(usize, Fault)>>,
    pub alias: Mutex<Option<GpuTensorHandle>>,
    pub probes: Mutex<Vec<RetirementProbe>>,
    pub controls: Mutex<Vec<ProviderRealElementwiseOperation>>,
    pub retained: Mutex<Vec<Vec<usize>>>,
    pub cleanup_failure_at: AtomicUsize,
}
impl AccelProvider for Fixture {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "compound real Add lifetime fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("no staging")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("no numerical fixture")
    }
    fn propose_real_elementwise(
        &self,
        request: &ProviderRealElementwiseRequest,
    ) -> ProviderExecutionResult<Option<ProviderRealElementwisePhysicalProposal>> {
        let (step, uniform_bytes) = match request.operation {
            ProviderRealElementwiseOperation::FloatingCast {
                target: ProviderPrecision::F64,
            } => (1, 16),
            ProviderRealElementwiseOperation::ScalarAdd {
                control: ScalarLiteral::F64Bits(_),
                mode: runmat_types::RealFloatingAddMode::Double,
            } => (2, 256),
            ProviderRealElementwiseOperation::FloatingCast {
                target: ProviderPrecision::F32,
            } => (3, 16),
            ProviderRealElementwiseOperation::ScalarAdd { .. }
            | ProviderRealElementwiseOperation::ExactSingleBasePower { .. }
            | ProviderRealElementwiseOperation::ScalarMul { .. }
            | ProviderRealElementwiseOperation::ScalarMax { .. }
            | ProviderRealElementwiseOperation::Sqrt
            | ProviderRealElementwiseOperation::Spacing
            | ProviderRealElementwiseOperation::Expm1
            | ProviderRealElementwiseOperation::UnaryMath(_) => return Ok(None),
        };
        if self.decline_at.load(Ordering::SeqCst) == step {
            return Ok(None);
        }
        let (output, route) = output_projection(request);
        Ok(Some(ProviderRealElementwisePhysicalProposal {
            output_route: route,
            output_buffer_bytes: output.checked_byte_len().unwrap(),
            workgroup_size: 64,
            chunk_capacity: 64,
            uniform_allocations: 1,
            uniform_buffer_bytes: uniform_bytes,
            dispatches: 1,
            queue_submissions: 1,
            pipeline_key: Some(step as u64),
        }))
    }
    fn execute_real_elementwise_recipe<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        recipe: &'a ProviderRealElementwiseRecipe,
        _admission: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> crate::ProviderExecutionFuture<'a, ProviderExecutedRealElementwise> {
        Box::pin(async move {
            let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            self.controls
                .lock()
                .unwrap()
                .push(recipe.request().operation);
            self.retained.lock().unwrap().push(
                self.probes
                    .lock()
                    .unwrap()
                    .iter()
                    .map(RetirementProbe::retirements)
                    .collect(),
            );
            let fault = self
                .fault
                .lock()
                .unwrap()
                .filter(|(at, _)| *at == call)
                .map(|(_, fault)| fault);
            if matches!(fault, Some(Fault::BeforeUnsupported)) {
                return Err(crate::unsupported_provider_operation(
                    "fixture unsupported before this hook effect",
                )
                .into());
            }
            let output = if matches!(fault, Some(Fault::Alias)) {
                self.alias.lock().unwrap().as_ref().unwrap().clone()
            } else {
                let expected = match recipe.output().element_type {
                    ProviderElementType::F32 => NumericElementType::F32,
                    ProviderElementType::F64 => NumericElementType::F64,
                    _ => unreachable!("real fixture output"),
                };
                let element = if matches!(fault, Some(Fault::WrongClass)) {
                    if expected == NumericElementType::F32 {
                        NumericElementType::F64
                    } else {
                        NumericElementType::F32
                    }
                } else {
                    expected
                };
                let allocated = self
                    .owner
                    .allocate_zeroed(input.shape().to_vec(), element, GpuTensorStorage::Real)
                    .unwrap();
                if self.cleanup_failure_at.load(Ordering::SeqCst) == call {
                    allocated
                        .probe
                        .fail_retirement("retained compound cleanup cause");
                }
                self.probes.lock().unwrap().push(allocated.probe);
                allocated.handle
            };
            if matches!(fault, Some(Fault::AfterCause)) {
                return Err(ProviderExecutionError::from(std::io::Error::other(
                    "original compound hook cause",
                ))
                .after_effect());
            }
            if matches!(fault, Some(Fault::Pending)) {
                std::future::pending::<()>().await;
            }
            let p = recipe.physical();
            Ok(ProviderExecutedRealElementwise {
                observed: ProviderRealElementwiseObserved {
                    output_buffer_bytes: Some(output.physical().requested_bytes()),
                    uniform_allocations: p.uniform_allocations,
                    uniform_buffer_bytes: p.uniform_buffer_bytes
                        + u64::from(matches!(fault, Some(Fault::WrongReceipt))),
                    dispatches: p.dispatches,
                    queue_submissions: p.queue_submissions,
                },
                output,
            })
        })
    }
}

pub(super) fn fixture() -> &'static Fixture {
    let fixture = Box::leak(Box::new(Fixture {
        owner: TestAllocationOwner::new(),
        decline_at: AtomicUsize::new(0),
        calls: AtomicUsize::new(0),
        fault: Mutex::new(None),
        alias: Mutex::new(None),
        probes: Mutex::new(Vec::new()),
        controls: Mutex::new(Vec::new()),
        retained: Mutex::new(Vec::new()),
        cleanup_failure_at: AtomicUsize::new(0),
    }));
    // SAFETY: leaked process-lifetime provider with its privately minted device.
    unsafe { crate::register_device_provider(fixture) };
    fixture
}
pub(super) fn allocate_source(fixture: &Fixture) -> crate::test_support::TestAllocation {
    fixture
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap()
}
pub(super) fn allocate_double_source(fixture: &Fixture) -> crate::test_support::TestAllocation {
    fixture
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap()
}
pub(super) fn double_source_request(bits: u32) -> ProviderRealElementwiseRequest {
    let mut original = request(ScalarLiteral::F32Bits(bits));
    original.source.element_type = ProviderElementType::F64;
    original.source_buffer_bytes = 16;
    original
}
pub(super) fn request(control: ScalarLiteral) -> ProviderRealElementwiseRequest {
    ProviderRealElementwiseRequest {
        operation: ProviderRealElementwiseOperation::ScalarAdd {
            control,
            mode: runmat_types::RealFloatingAddMode::MixedDoubleThenSingle,
        },
        source: ProviderRepresentation {
            element_type: ProviderElementType::F32,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            residency: ProviderResidency::Device,
            shape: vec![2, 1],
        },
        source_buffer_bytes: 8,
    }
}
pub(super) fn ready<F: std::future::Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    match future
        .as_mut()
        .poll(&mut std::task::Context::from_waker(std::task::Waker::noop()))
    {
        std::task::Poll::Ready(value) => value,
        std::task::Poll::Pending => panic!("unexpected fixture suspension"),
    }
}
