//! A selected host-origin convolution keeps cancellation distinct from a
//! provider failure after its asynchronous dispatch has returned an output.

use super::*;
use crate::builtins::deep_learning::layers::{
    convolution_1d_layer_builtin, sequence_input_layer_builtin,
};
use runmat_accelerate_api::{
    DeepLearningConv1DCapability, DeepLearningConv1DOperands, DeepLearningConv1DRequest,
};
use runmat_execution::{cancellation::CancellationSource, CancellationReason};
use std::rc::Rc;

struct CancelConvolution {
    owner: TestAllocationOwner,
    cancellation: CancellationSource,
    uploads: AtomicUsize,
    launches: AtomicUsize,
    probes: Mutex<Vec<runmat_accelerate_api::test_support::RetirementProbe>>,
}

impl AccelProvider for CancelConvolution {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "selected-convolution-cancellation".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("native host route must not use the legacy floating upload")
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let data = match host.data {
            HostNumericDataView::F32(values) => HostNumericDataOwned::F32(values.to_vec()),
            HostNumericDataView::F64(values) => HostNumericDataOwned::F64(values.to_vec()),
            _ => {
                return Err(
                    anyhow::anyhow!("convolution fixture expects native floating storage").into(),
                )
            }
        };
        let allocated = self.owner.allocate_numeric(HostNumericTensorOwned {
            data,
            shape: host.shape.to_vec(),
            storage: host.storage,
        })?;
        self.uploads.fetch_add(1, Ordering::SeqCst);
        self.probes.lock().unwrap().push(allocated.probe);
        Ok(allocated.handle)
    }

    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        panic!("cancelled selected route must not gather or replay")
    }

    fn deep_learning_forward_route_capability(
        &self,
        route: &DeepLearningForwardRoute<'_>,
    ) -> anyhow::Result<Option<DeepLearningForwardCapability>> {
        let geometry = route
            .analyze()
            .map_err(|error| anyhow::anyhow!("{error:?}"))?;
        Ok(Some(DeepLearningForwardCapability::Ready(
            DeepLearningForwardResources {
                output_bytes: geometry.output_bytes,
                largest_stage_value_bytes: geometry.largest_stage_value_bytes,
                known_normalization_scratch_bytes: 0,
                upload_bytes: 0,
                resident_bytes: 0,
                new_buffer_bytes_upper_bound:
                    runmat_accelerate_api::DeepLearningForwardBufferBound::Unavailable,
            },
        )))
    }

    fn deep_learning_conv1d_capability(
        &self,
        operands: DeepLearningConv1DOperands<'_>,
        _: &DeepLearningConv1DRequest,
    ) -> anyhow::Result<DeepLearningConv1DCapability> {
        operands.anchor.allocation_lease().validate_live()?;
        Ok(DeepLearningConv1DCapability::Ready)
    }

    fn deep_learning_conv1d<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
        request: &'a DeepLearningConv1DRequest,
    ) -> runmat_accelerate_api::AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.launches.fetch_add(1, Ordering::SeqCst);
            let allocated = self.owner.allocate_zeroed(
                request.output_shape.clone(),
                input.physical().element_type(),
                GpuTensorStorage::Real,
            )?;
            self.probes.lock().unwrap().push(allocated.probe);
            std::future::ready(()).await;
            self.cancellation.cancel(CancellationReason::User);
            Ok(allocated.handle)
        })
    }
}

#[test]
fn selected_convolution_cancellation_after_dispatch_abandons_and_retires_only_owned_buffers() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let cancellation = CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    );
    let provider: &'static CancelConvolution = Box::leak(Box::new(CancelConvolution {
        owner: TestAllocationOwner::new(),
        cancellation,
        uploads: AtomicUsize::new(0),
        launches: AtomicUsize::new(0),
        probes: Mutex::new(Vec::new()),
    }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let input_layer = block_on(sequence_input_layer_builtin(
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let conv = block_on(convolution_1d_layer_builtin(
        Value::Num(2.0),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let network_owner = block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input_layer, conv], 2, 1).unwrap(),
    )]))
    .unwrap();
    let Value::Object(network) = network_owner.value() else {
        panic!("dlnetwork constructor must return an object")
    };
    let input = Value::Tensor(Tensor::from_f32(vec![1.0, 4.0], vec![1, 1, 2]).unwrap());
    let host = HostNetworkPlan::admit(network, &input, "forward").unwrap();
    let admitted = host_origin::admit_host_route(&host, host.output_axes(), provider)
        .unwrap()
        .expect("lowered learned convolution route")
        .expect("provider admits the complete pure host route");
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.launches.load(Ordering::SeqCst), 0);
    let failure = block_on(host_origin::execute_host_route_selected(
        admitted,
        Some(&runtime),
    ))
    .expect_err("cancellation after convolution dispatch must abandon the selected route");
    assert!(!failure.is_operational());
    assert_eq!(failure.into_error().identifier(), Some("RunMat:Cancelled"));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 3);
    assert_eq!(provider.launches.load(Ordering::SeqCst), 1);
    let probes = provider.probes.lock().unwrap();
    assert_eq!(probes.len(), 4);
    assert!(probes.iter().all(|probe| probe.retirements() == 1));
}
