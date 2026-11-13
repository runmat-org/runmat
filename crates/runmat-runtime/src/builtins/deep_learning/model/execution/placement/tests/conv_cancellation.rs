//! The public selected-route observation must not turn convolution
//! cancellation into negative provider performance feedback.

use super::*;
use crate::builtins::deep_learning::layers::{
    convolution_1d_layer_builtin, sequence_input_layer_builtin,
};
use runmat_accelerate_api::{
    DeepLearningConv1DCapability, DeepLearningConv1DOperands, DeepLearningConv1DRequest,
};
use runmat_execution::{cancellation::CancellationSource, CancellationReason};

struct CancelledConvolution {
    owner: TestAllocationOwner,
    cancellation: CancellationSource,
    launches: AtomicUsize,
}

impl AccelProvider for CancelledConvolution {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "selected-convolution-observation".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("selected convolution must use native numeric uploads")
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let data = match host.data {
            HostNumericDataView::F32(values) => HostNumericDataOwned::F32(values.to_vec()),
            HostNumericDataView::F64(values) => HostNumericDataOwned::F64(values.to_vec()),
            _ => {
                return Err(anyhow::anyhow!("convolution fixture requires floating storage").into())
            }
        };
        Ok(self
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                data,
                shape: host.shape.to_vec(),
                storage: host.storage,
            })?
            .handle)
    }

    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        panic!("cancelled selected convolution must not gather")
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
                new_buffer_bytes_upper_bound: DeepLearningForwardBufferBound::Unavailable,
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
            let output = self.owner.allocate_zeroed(
                request.output_shape.clone(),
                input.physical().element_type(),
                GpuTensorStorage::Real,
            )?;
            std::future::ready(()).await;
            self.cancellation.cancel(CancellationReason::User);
            Ok(output.handle)
        })
    }
}

#[test]
fn public_selected_convolution_cancellation_produces_no_failure_feedback() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider: &'static CancelledConvolution = Box::leak(Box::new(CancelledConvolution {
        owner: TestAllocationOwner::new(),
        cancellation: CancellationSource::new(),
        launches: AtomicUsize::new(0),
    }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let service = Rc::new(SelectedRoute {
        prefer_provider: true,
        ..SelectedRoute::default()
    });
    let runtime = runtime(service.clone()).with_cancellation_token(provider.cancellation.token());
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
    let error = block_on(execute_network(NetworkExecutionRequest {
        runtime: Some(&runtime),
        network,
        input: &input,
        caller: NetworkCaller::Forward,
        output_layout: NetworkOutputLayout::Preserve,
        requested_outputs: Some(1),
    }))
    .err()
    .expect("selected convolution must stop after cancellation");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(provider.launches.load(Ordering::SeqCst), 1);
    assert!(service.feedback.borrow().is_empty());
}
