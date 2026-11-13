use super::*;
use crate::context::{
    host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
    RuntimeParallelService, RuntimeServicePorts,
};
use futures::executor::block_on;
use std::rc::Rc;
struct Finite(ManagedHostEffectLedger);

#[derive(Default)]
struct TransferProvider {
    owner: runmat_accelerate_api::test_support::TestAllocationOwner,
    allocations: std::sync::Mutex<Vec<runmat_accelerate_api::test_support::TestAllocation>>,
    downloads: std::sync::atomic::AtomicUsize,
}
impl TransferProvider {
    fn value(
        &self,
        data: Vec<f64>,
        shape: Vec<usize>,
        intent: runmat_accelerate_api::GpuHandleProvenance,
    ) -> Value {
        let allocation = self
            .owner
            .allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
                data: runmat_accelerate_api::HostNumericDataOwned::F64(data),
                shape,
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            })
            .unwrap();
        let handle = allocation.handle.clone().with_provenance(intent).unwrap();
        self.allocations.lock().unwrap().push(allocation);
        Value::GpuTensor(handle)
    }
}
impl runmat_accelerate_api::AccelProvider for TransferProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "inferential automatic transfer fixture".into()
    }
    fn upload(
        &self,
        _: &runmat_accelerate_api::HostTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        panic!("fixture owns native allocations directly")
    }
    fn download<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        panic!("native typed download required")
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        self.downloads
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Box::pin(async move {
            let allocations = self.allocations.lock().unwrap();
            let allocation = allocations
                .iter()
                .find(|allocation| {
                    allocation.handle.allocation_identity() == handle.allocation_identity()
                })
                .expect("actual fixture owner");
            let data = allocation.probe.with_data(handle, |data| {
                let runmat_accelerate_api::HostNumericDataView::F64(values) = data else {
                    panic!("real double fixture")
                };
                runmat_accelerate_api::HostNumericDataOwned::F64(values.to_vec())
            })?;
            Ok(runmat_accelerate_api::HostNumericTensorOwned {
                data,
                shape: handle.shape().to_vec(),
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            })
        })
    }
}

#[test]
fn automatic_residency_executes_but_explicit_samples_controls_and_groups_never_download() {
    use runmat_accelerate_api::{
        GpuHandleProvenance::{Automatic, Explicit},
        ThreadProviderGuard,
    };
    use std::sync::atomic::Ordering;
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(TransferProvider::default()));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _provider = ThreadProviderGuard::set(Some(provider));
    let _outputs = crate::output_count::push_output_count(Some(3)).unwrap();
    let sample = provider.value(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Automatic);
    let alpha = provider.value(vec![0.05], vec![1, 1], Automatic);
    let groups = provider.value(vec![1.0, 2.0], vec![2, 1], Automatic);
    let args = crate::builtins::common::test_support::builtin_arguments(vec![
        sample.clone(),
        Value::from("Alpha"),
        alpha,
    ]);
    assert!(block_on(super::super::ttest::evaluate(None, args)).is_ok());
    let args = crate::builtins::common::test_support::builtin_arguments(vec![
        sample.clone(),
        groups.clone(),
        Value::from("off"),
    ]);
    assert!(block_on(super::super::anova1::evaluate(None, args)).is_ok());
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 4);
    let explicit_sample = provider.value(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Explicit);
    let explicit_control = provider.value(vec![0.05], vec![1, 1], Explicit);
    let explicit_groups = provider.value(vec![1.0, 2.0], vec![2, 1], Explicit);
    for (anova, values) in [
        (false, vec![explicit_sample.clone()]),
        (false, vec![sample.clone(), explicit_sample]),
        (false, vec![sample.clone(), explicit_control.clone()]),
        (
            false,
            vec![
                sample.clone(),
                Value::from("Alpha"),
                explicit_control.clone(),
            ],
        ),
        (
            false,
            vec![sample.clone(), Value::from("Dim"), explicit_control],
        ),
        (true, vec![sample, explicit_groups, Value::from("off")]),
    ] {
        let args = crate::builtins::common::test_support::builtin_arguments(values);
        let error = if anova {
            block_on(super::super::anova1::evaluate(None, args))
        } else {
            block_on(super::super::ttest::evaluate(None, args))
        }
        .unwrap_err();
        assert!(error
            .message()
            .contains("host fallback requires automatic residency"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(
            provider.downloads.load(Ordering::SeqCst),
            4,
            "reject all roles before even an earlier automatic argument transfers"
        );
    }
}
impl RuntimeParallelService for Finite {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let s = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(s.free_bytes()),
            epoch: s.epoch,
            ..Default::default()
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}
#[test]
fn both_entries_refuse_finite_before_any_statistical_workspace_and_preserve_typed_cancel() {
    let ledger = ManagedHostEffectLedger::new(1024 * 1024);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Finite(ledger.clone()))),
        );
    for anova in [false, true] {
        let mut values = vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(),
        )];
        if anova {
            values.extend([
                Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap()),
                Value::from("off"),
            ]);
        }
        let args = crate::builtins::common::test_support::builtin_arguments(values);
        let error = if anova {
            block_on(super::super::anova1::evaluate(Some(&runtime), args))
        } else {
            block_on(super::super::ttest::evaluate(Some(&runtime), args))
        }
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:Placement:InsufficientResourceEvidence")
        );
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }
    assert_eq!(ledger.snapshot().peak_reserved_bytes, 0);
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_cancellation_token(cancellation.token());
    runtime.set_error_namespace("StatisticsScope");
    cancellation.cancel(runmat_execution::CancellationReason::User);
    for anova in [false, true] {
        let mut values = vec![Value::Num(1.0)];
        if anova {
            values.extend([
                Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap()),
                Value::from("off"),
            ]);
        }
        let args = crate::builtins::common::test_support::builtin_arguments(values);
        let error = if anova {
            block_on(super::super::anova1::evaluate(Some(&runtime), args))
        } else {
            block_on(super::super::ttest::evaluate(Some(&runtime), args))
        }
        .unwrap_err();
        assert_eq!(error.identifier(), Some("StatisticsScope:Cancelled"));
        assert!(crate::runtime_error::error_has_control_cancellation_origin(
            &error
        ));
    }
}
#[test]
fn shifted_scaled_moments_preserve_small_spreads_and_large_representable_deviations() {
    let moments = block_on(moments(
        &[1e150, 1e150 + 1e140, 1e150 + 2e140],
        &mut Cooperation::new(None),
    ))
    .unwrap();
    let sd = moments.scale * (moments.normalized_squares / 2.0).sqrt();
    assert!((sd / 1e140 - 1.0).abs() < 1e-5);
    let moments = block_on(super::moments(
        &[1e308, -1e308, -1e308],
        &mut Cooperation::new(None),
    ))
    .unwrap();
    let sd = moments.scale * (moments.normalized_squares / 2.0).sqrt();
    assert!(moments.mean.is_finite() && sd.is_finite());
}
