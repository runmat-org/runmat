use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, GpuTensorStorage, HostTensorView, NumericElementType,
};
use std::{
    future::Future,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    task::{Context, Poll},
};

struct ReductionOwner {
    owner: TestAllocationOwner,
    outputs: Mutex<Vec<RetirementProbe>>,
    alias: bool,
    pending: bool,
    first_fault: Option<FirstFault>,
    wrong_dim_shape: bool,
    first_dim_pending: bool,
    dim_calls: Mutex<Vec<usize>>,
    max_dim_calls: Mutex<Vec<usize>>,
    downloads: AtomicUsize,
}

#[derive(Clone, Copy)]
enum FirstFault {
    Unsupported,
    Execution,
    UnprovenUnsupported,
    AfterEffectUnsupported,
}

fn first_fault_error(fault: FirstFault) -> runmat_accelerate_api::ProviderExecutionError {
    match fault {
        FirstFault::Unsupported => runmat_accelerate_api::ProviderExecutionError::from(
            runmat_accelerate_api::unsupported_provider_operation("minimum unavailable"),
        ),
        FirstFault::Execution => runmat_accelerate_api::ProviderExecutionError::from(
            std::io::Error::new(std::io::ErrorKind::BrokenPipe, "minimum driver failed"),
        ),
        FirstFault::UnprovenUnsupported => runmat_accelerate_api::ProviderExecutionError::from(
            runmat_accelerate_api::unsupported_provider_operation("minimum phase unknown"),
        )
        .with_unproven_effects(),
        FirstFault::AfterEffectUnsupported => runmat_accelerate_api::ProviderExecutionError::from(
            runmat_accelerate_api::unsupported_provider_operation(
                "minimum allocated before failure",
            ),
        )
        .after_effect(),
    }
}

impl AccelProvider for ReductionOwner {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "owned range reduction fixture".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        unreachable!("no uploads in reduction lifetime witness")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { panic!("no downloads in reduction lifetime witness") })
    }
    fn reduce_min<'a>(
        &'a self,
        source: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if let Some(fault) = self.first_fault {
                return Err(first_fault_error(fault));
            }
            if self.alias {
                return Ok(source.clone());
            }
            let allocation = self.owner.allocate_zeroed(
                vec![1, 1],
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )?;
            self.outputs.lock().unwrap().push(allocation.probe);
            Ok(allocation.handle)
        })
    }
    fn reduce_max<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.pending {
                futures::future::pending::<()>().await;
            }
            Err(anyhow::anyhow!("injected second reduction failure").into())
        })
    }
    fn reduce_min_dim<'a>(
        &'a self,
        source: &'a GpuTensorHandle,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, runmat_accelerate_api::ReduceDimResult>
    {
        self.dim_calls.lock().unwrap().push(dim);
        Box::pin(async move {
            if self.first_dim_pending {
                futures::future::pending::<()>().await;
            }
            if let Some(fault) = self.first_fault {
                return Err(first_fault_error(fault));
            }
            let mut shape = source.shape().to_vec();
            shape[dim] = 1;
            if self.wrong_dim_shape {
                shape = vec![2, 1];
            }
            let values = self.owner.allocate_zeroed(
                shape.clone(),
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )?;
            let indices = self.owner.allocate_zeroed(
                shape,
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )?;
            let mut outputs = self.outputs.lock().unwrap();
            outputs.push(values.probe);
            outputs.push(indices.probe);
            Ok(runmat_accelerate_api::ReduceDimResult {
                values: values.handle,
                indices: indices.handle,
            })
        })
    }
    fn reduce_max_dim<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, runmat_accelerate_api::ReduceDimResult>
    {
        self.max_dim_calls.lock().unwrap().push(dim);
        Box::pin(async move {
            assert!(
                !self.wrong_dim_shape,
                "wrong-shape minimum must not attempt maximum"
            );
            if self.pending {
                futures::future::pending::<()>().await;
            }
            Err(anyhow::anyhow!("injected second dimensional reduction failure").into())
        })
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn range_intermediate_owns_its_lifetime_across_failure_and_pending_drop() {
    for alias in [false, true] {
        for pending in [false, true] {
            let provider = Box::leak(Box::new(ReductionOwner {
                owner: TestAllocationOwner::new(),
                outputs: Mutex::new(Vec::new()),
                alias,
                pending,
                first_fault: None,
                wrong_dim_shape: false,
                first_dim_pending: false,
                dim_calls: Mutex::new(Vec::new()),
                max_dim_calls: Mutex::new(Vec::new()),
                downloads: AtomicUsize::new(0),
            }));
            let input = provider
                .owner
                .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let mut future = Box::pin(range_gpu_all(provider, &input.handle));
            let waker = futures::task::noop_waker();
            let mut context = Context::from_waker(&waker);
            match future.as_mut().poll(&mut context) {
                Poll::Pending => assert!(pending),
                Poll::Ready(output) => {
                    assert!(!pending);
                    let error = output.expect_err("second reduction fails after first effect");
                    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                    let mut cause: &(dyn std::error::Error + 'static) = &error;
                    loop {
                        if let Some(provider) =
                            cause.downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
                        {
                            assert!(provider.is_after_effect());
                            break;
                        }
                        cause = cause.source().expect("original provider cause retained");
                    }
                }
            }
            assert_eq!(input.probe.retirements(), 0);
            if !alias && pending {
                assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 0);
            }
            drop(future);
            if !alias {
                assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 1);
            }
            assert_eq!(input.probe.retirements(), 0);
            drop(input.handle);
            assert_eq!(input.probe.retirements(), 1);
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn public_range_does_not_download_and_recompute_after_successful_minimum() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let provider = Box::leak(Box::new(ReductionOwner {
        owner: TestAllocationOwner::new(),
        outputs: Mutex::new(Vec::new()),
        alias: false,
        pending: false,
        first_fault: None,
        wrong_dim_shape: false,
        first_dim_pending: false,
        dim_calls: Mutex::new(Vec::new()),
        max_dim_calls: Mutex::new(Vec::new()),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_zeroed(vec![2], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("resident Double input");
    let original = input.handle.wire_descriptor();
    let error = futures::executor::block_on(crate::call_builtin_async(
        "range",
        &[
            Value::GpuTensor(input.handle.clone()),
            Value::from("all"),
            Value::from("includenan"),
        ],
    ))
    .expect_err("post-minimum maximum failure must be terminal");
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(error.message().contains("global maximum"));
    assert_eq!(provider.outputs.lock().unwrap().len(), 1);
    assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 1);
    assert_eq!(input.handle.wire_descriptor(), original);
    input.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(input.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn initial_minimum_falls_back_only_for_certified_pre_effect_unsupported() {
    for fault in [
        FirstFault::Unsupported,
        FirstFault::Execution,
        FirstFault::UnprovenUnsupported,
        FirstFault::AfterEffectUnsupported,
    ] {
        let provider = Box::leak(Box::new(ReductionOwner {
            owner: TestAllocationOwner::new(),
            outputs: Mutex::new(Vec::new()),
            alias: false,
            pending: false,
            first_fault: Some(fault),
            wrong_dim_shape: false,
            first_dim_pending: false,
            dim_calls: Mutex::new(Vec::new()),
            max_dim_calls: Mutex::new(Vec::new()),
            downloads: AtomicUsize::new(0),
        }));
        let input = provider
            .owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let outcome = futures::executor::block_on(range_gpu_all(provider, &input.handle));
        if matches!(fault, FirstFault::Unsupported) {
            assert!(outcome.unwrap().is_none());
        } else {
            let error = outcome.expect_err("first-hook failure is terminal");
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert!(error.message().contains("global minimum"));
        }
        assert!(provider.outputs.lock().unwrap().is_empty());
        input.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input.probe.retirements(), 0);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn wrong_shape_dimensional_minimum_is_terminal_and_retires_both_results() {
    let provider = Box::leak(Box::new(ReductionOwner {
        owner: TestAllocationOwner::new(),
        outputs: Mutex::new(Vec::new()),
        alias: false,
        pending: false,
        first_fault: None,
        wrong_dim_shape: true,
        first_dim_pending: false,
        dim_calls: Mutex::new(Vec::new()),
        max_dim_calls: Mutex::new(Vec::new()),
        downloads: AtomicUsize::new(0),
    }));
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let error =
        futures::executor::block_on(range_gpu_single_dim(provider, &input.handle, 0, &[1, 1]))
            .expect_err("wrong-shaped first result is terminal");
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(error.message().contains("dimensional minimum"));
    let probes = provider.outputs.lock().unwrap();
    assert_eq!(probes.len(), 2);
    assert!(probes.iter().all(|probe| probe.retirements() == 1));
    input.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(input.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn dimensional_first_fault_uses_only_the_canonical_axis() {
    for axis in [0, 1] {
        for fault in [
            FirstFault::Unsupported,
            FirstFault::Execution,
            FirstFault::UnprovenUnsupported,
            FirstFault::AfterEffectUnsupported,
        ] {
            let provider = Box::leak(Box::new(ReductionOwner {
                owner: TestAllocationOwner::new(),
                outputs: Mutex::new(Vec::new()),
                alias: false,
                pending: false,
                first_fault: Some(fault),
                wrong_dim_shape: false,
                first_dim_pending: false,
                dim_calls: Mutex::new(Vec::new()),
                max_dim_calls: Mutex::new(Vec::new()),
                downloads: AtomicUsize::new(0),
            }));
            let input = provider
                .owner
                .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let expected = if axis == 0 { [1, 2] } else { [2, 1] };
            let outcome = futures::executor::block_on(range_gpu_single_dim(
                provider,
                &input.handle,
                axis,
                &expected,
            ));
            if matches!(fault, FirstFault::Unsupported) {
                assert!(outcome.unwrap().is_none());
            } else {
                let error = outcome.expect_err("uncertified first failure is terminal");
                assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                assert!(error.message().contains("dimensional minimum"));
            }
            assert_eq!(*provider.dim_calls.lock().unwrap(), vec![axis]);
            assert!(provider.max_dim_calls.lock().unwrap().is_empty());
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
            assert!(provider.outputs.lock().unwrap().is_empty());
            input.handle.allocation_lease().validate_live().unwrap();
            assert_eq!(input.probe.retirements(), 0);
            drop(input.handle);
            assert_eq!(input.probe.retirements(), 1);
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pending_dimensional_first_hook_drops_without_replay_or_borrowed_retirement() {
    let provider = Box::leak(Box::new(ReductionOwner {
        owner: TestAllocationOwner::new(),
        outputs: Mutex::new(Vec::new()),
        alias: false,
        pending: false,
        first_fault: None,
        wrong_dim_shape: false,
        first_dim_pending: true,
        dim_calls: Mutex::new(Vec::new()),
        max_dim_calls: Mutex::new(Vec::new()),
        downloads: AtomicUsize::new(0),
    }));
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let mut future = Box::pin(range_gpu_single_dim(provider, &input.handle, 0, &[1, 2]));
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
    assert_eq!(*provider.dim_calls.lock().unwrap(), vec![0]);
    assert!(provider.max_dim_calls.lock().unwrap().is_empty());
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert!(provider.outputs.lock().unwrap().is_empty());
    drop(future);
    input.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(input.probe.retirements(), 0);
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn dimensional_second_hook_failure_and_pending_drop_are_terminal() {
    for pending in [false, true] {
        let provider = Box::leak(Box::new(ReductionOwner {
            owner: TestAllocationOwner::new(),
            outputs: Mutex::new(Vec::new()),
            alias: false,
            pending,
            first_fault: None,
            wrong_dim_shape: false,
            first_dim_pending: false,
            dim_calls: Mutex::new(Vec::new()),
            max_dim_calls: Mutex::new(Vec::new()),
            downloads: AtomicUsize::new(0),
        }));
        let input = provider
            .owner
            .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let mut future = Box::pin(range_gpu_single_dim(provider, &input.handle, 1, &[2, 1]));
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        match future.as_mut().poll(&mut context) {
            Poll::Pending => assert!(pending),
            Poll::Ready(outcome) => {
                assert!(!pending);
                let error = outcome.expect_err("second dimensional failure is terminal");
                assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                assert!(error.message().contains("dimensional maximum"));
                let mut source: &(dyn std::error::Error + 'static) = &error;
                loop {
                    if let Some(typed) =
                        source.downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
                    {
                        assert!(typed.is_after_effect());
                        break;
                    }
                    source = source.source().expect("original provider cause");
                }
            }
        }
        assert_eq!(*provider.dim_calls.lock().unwrap(), vec![1]);
        assert_eq!(*provider.max_dim_calls.lock().unwrap(), vec![1]);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.outputs.lock().unwrap().len(), 2);
        if pending {
            assert!(provider
                .outputs
                .lock()
                .unwrap()
                .iter()
                .all(|probe| probe.retirements() == 0));
        }
        drop(future);
        assert!(provider
            .outputs
            .lock()
            .unwrap()
            .iter()
            .all(|probe| probe.retirements() == 1));
        input.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input.probe.retirements(), 0);
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
    }
}
