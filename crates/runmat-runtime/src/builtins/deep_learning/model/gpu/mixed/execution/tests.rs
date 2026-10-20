use super::*;
use crate::builtins::deep_learning::object;
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelProvider, GpuTensorStorage, HostNumericDataOwned, HostNumericTensorOwned, HostTensorView,
    NumericElementType,
};
use runmat_execution::{cancellation::CancellationSource, CancellationReason};
use runmat_value::{CellArray, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

struct CancellingStageProvider {
    owner: TestAllocationOwner,
    cancellation: CancellationSource,
    stages: AtomicUsize,
    outputs: Mutex<Vec<RetirementProbe>>,
    fail_after_cancel: bool,
}

impl AccelProvider for CancellingStageProvider {
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("admitted resident execution must not upload")
    }

    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        panic!("admitted resident execution must not download")
    }

    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "cancelling-stage-fixture".into()
    }

    fn activation_relu<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            input.allocation_lease().validate_live()?;
            self.stages.fetch_add(1, Ordering::SeqCst);
            let allocation = self.owner.allocate_zeroed(
                input.shape().to_vec(),
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )?;
            self.outputs.lock().unwrap().push(allocation.probe);
            std::future::ready(()).await;
            self.cancellation.cancel(CancellationReason::User);
            if self.fail_after_cancel {
                anyhow::bail!("provider ReLU failed after issuing cancellation");
            }
            Ok(allocation.handle)
        })
    }
}

#[test]
fn cancellation_after_awaited_stage_stops_later_stages_without_publication() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let cancellation = CancellationSource::new();
    let provider = Box::leak(Box::new(CancellingStageProvider {
        owner: TestAllocationOwner::new(),
        cancellation,
        stages: AtomicUsize::new(0),
        outputs: Mutex::new(Vec::new()),
        fail_after_cancel: false,
    }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let allocation = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![2.0]),
            shape: vec![1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let borrowed_input = allocation.handle;
    let network = object(
        "dlnetwork",
        vec![("Layers", Value::Cell(CellArray::new(vec![], 0, 1).unwrap()))],
    )
    .expect("fixture object");
    let Value::Object(network) = network else {
        unreachable!()
    };
    let plan = Plan {
        stages: vec![Stage::Relu, Stage::Relu],
        output_axes: AxisPlan::for_input(
            "BC",
            runmat_builtins::DeepLearningInputKind::Feature,
            &[1, 1],
        )
        .unwrap(),
        consumed_resident_parameters: Vec::new(),
    };
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_cancellation_token(provider.cancellation.token());
    runtime.set_error_namespace("Invocation");
    let result = block_on(execute(
        &network,
        &borrowed_input,
        plan,
        true,
        Some(&runtime),
        "forward",
    ));
    let Err(error) = result else {
        panic!("the first awaited stage must cancel before publication")
    };
    assert!(matches!(&error, StageExecutionFailure::Abandoned(_)));
    assert_eq!(
        error.into_error().identifier(),
        Some("Invocation:Cancelled")
    );
    assert_eq!(provider.stages.load(Ordering::SeqCst), 1);
    borrowed_input.allocation_lease().validate_live().unwrap();
    assert_eq!(allocation.probe.retirements(), 0);
    let outputs = provider.outputs.lock().unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].retirements(), 1);
}

#[test]
fn provider_failure_remains_operational_when_cancellation_races_it() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(CancellingStageProvider {
        owner: TestAllocationOwner::new(),
        cancellation: CancellationSource::new(),
        stages: AtomicUsize::new(0),
        outputs: Mutex::new(Vec::new()),
        fail_after_cancel: true,
    }));
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let allocation = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![2.0]),
            shape: vec![1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let network = object(
        "dlnetwork",
        vec![("Layers", Value::Cell(CellArray::new(vec![], 0, 1).unwrap()))],
    )
    .expect("fixture object");
    let Value::Object(network) = network else {
        unreachable!()
    };
    let plan = Plan {
        stages: vec![Stage::Relu],
        output_axes: AxisPlan::for_input(
            "BC",
            runmat_builtins::DeepLearningInputKind::Feature,
            &[1, 1],
        )
        .unwrap(),
        consumed_resident_parameters: Vec::new(),
    };
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_cancellation_token(provider.cancellation.token());
    let failure = block_on(execute(
        &network,
        &allocation.handle,
        plan,
        false,
        Some(&runtime),
        "forward",
    ))
    .err()
    .expect("the provider must fail the selected stage");
    assert!(provider.cancellation.token().is_cancelled());
    assert!(matches!(&failure, StageExecutionFailure::Operational(_)));
    assert_ne!(failure.into_error().identifier(), Some("RunMat:Cancelled"));
    assert_eq!(provider.stages.load(Ordering::SeqCst), 1);
    assert_eq!(allocation.probe.retirements(), 0);
    let outputs = provider.outputs.lock().unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].retirements(), 1);
}
