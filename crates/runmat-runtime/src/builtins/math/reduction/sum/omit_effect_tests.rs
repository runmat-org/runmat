//! Public omit-NaN sum fallback is permitted only before provider work.
use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelNumericDownloadFuture, GpuTensorStorage, HostNumericDataOwned,
    HostNumericTensorOwned, HostTensorView, NumericElementType, ProviderExecutionFuture,
    ProviderExecutionResult, ThreadProviderGuard,
};
use std::{
    future::Future,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    task::{Context, Poll},
};

#[derive(Clone, Copy)]
enum FailurePoint {
    FirstMap,
    AfterWidenMap,
    FirstReduction,
    LaterReduction,
    PendingLaterReduction,
}

struct StageProvider {
    owner: TestAllocationOwner,
    point: FailurePoint,
    downloads: AtomicUsize,
    reductions: AtomicUsize,
    widened: Mutex<Vec<RetirementProbe>>,
    mapped: Mutex<Vec<RetirementProbe>>,
    reduced: Mutex<Vec<RetirementProbe>>,
    restored: Mutex<Vec<RetirementProbe>>,
}

impl StageProvider {
    fn new(point: FailurePoint) -> &'static Self {
        Box::leak(Box::new(Self {
            owner: TestAllocationOwner::new(),
            point,
            downloads: AtomicUsize::new(0),
            reductions: AtomicUsize::new(0),
            widened: Mutex::new(Vec::new()),
            mapped: Mutex::new(Vec::new()),
            reduced: Mutex::new(Vec::new()),
            restored: Mutex::new(Vec::new()),
        }))
    }

    fn output(&self, shape: Vec<usize>, probes: &Mutex<Vec<RetirementProbe>>) -> GpuTensorHandle {
        let allocation = self
            .owner
            .allocate_zeroed(shape, NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        probes.lock().unwrap().push(allocation.probe);
        allocation.handle
    }
}

impl AccelProvider for StageProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "sum omitnan effect-order fixture".into()
    }

    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }

    fn upload(&self, host: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        assert!(matches!(self.point, FailurePoint::FirstMap));
        assert_eq!(host.shape, &[1, 2]);
        assert_eq!(host.data, &[0.0, 0.0]);
        Ok(self.output(host.shape.to_vec(), &self.restored))
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unexpected legacy download") })
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let len = handle.shape().iter().product();
            Ok(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(vec![0.0; len]),
                shape: handle.shape().to_vec(),
                storage: GpuTensorStorage::Real,
            })
        })
    }

    fn unary_double<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Ok(self.output(input.shape().to_vec(), &self.widened)) })
    }

    fn map_nan_to_zero(&self, input: &GpuTensorHandle) -> ProviderExecutionResult<GpuTensorHandle> {
        if matches!(
            self.point,
            FailurePoint::FirstMap | FailurePoint::AfterWidenMap
        ) {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "selected NaN map is unavailable",
            )
            .into());
        }
        Ok(self.output(input.shape().to_vec(), &self.mapped))
    }

    fn reduce_sum_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dimension: usize,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        let ordinal = self.reductions.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            match (self.point, ordinal) {
                (FailurePoint::FirstReduction, 0) | (FailurePoint::LaterReduction, 1) => {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "selected reduction is unavailable",
                    )
                    .into());
                }
                (FailurePoint::PendingLaterReduction, 1) => {
                    return futures::future::pending().await;
                }
                _ => {}
            }
            let mut shape = input.shape().to_vec();
            shape[dimension] = 1;
            Ok(self.output(shape, &self.reduced))
        })
    }
}

async fn selected_sum(
    provider: &'static StageProvider,
    source: GpuTensorHandle,
    selection: DimSelection,
) -> BuiltinResult<Value> {
    let _selected = ThreadProviderGuard::set(Some(provider));
    let selector = match selection {
        DimSelection::Dim(dimension) => Value::Num(dimension as f64),
        DimSelection::All => Value::from("all"),
        _ => panic!("fixture requires an explicit dimension or all-axis selection"),
    };
    sum_builtin(
        None,
        None,
        Value::GpuTensor(source),
        crate::builtins::common::test_support::builtin_arguments(vec![
            selector,
            Value::from("omitnan"),
        ]),
    )
    .await
    .map(|owned| owned.value().clone())
}

fn original_unsupported(error: &RuntimeError) {
    let mut cause: &(dyn std::error::Error + 'static) = error;
    while !cause.is::<runmat_accelerate_api::UnsupportedProviderOperation>() {
        cause = cause
            .source()
            .expect("original typed unsupported marker retained");
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn first_pre_effect_map_refusal_uses_the_admitted_host_boundary() {
    let provider = StageProvider::new(FailurePoint::FirstMap);
    let source = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let result = block_on(selected_sum(
        provider,
        source.handle.clone(),
        DimSelection::Dim(1),
    ))
    .expect("the first certified refusal may use host sum");
    let Value::GpuTensor(result) = result else {
        panic!("the admitted host fallback restores the resident input's output contract");
    };
    assert_eq!(result.shape(), &[1, 2]);
    assert_eq!(
        runmat_accelerate_api::handle_precision(&result),
        Some(ProviderPrecision::F64)
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.restored.lock().unwrap().len(), 1);
    assert!(provider.mapped.lock().unwrap().is_empty());
    assert_eq!(source.probe.retirements(), 0);
    source.handle.allocation_lease().validate_live().unwrap();
    result.allocation_lease().validate_live().unwrap();
    result.release().unwrap();
    assert_eq!(provider.restored.lock().unwrap()[0].retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn unsupported_after_widen_or_map_never_replays_the_original_sum() {
    for (point, element, selection) in [
        (
            FailurePoint::AfterWidenMap,
            NumericElementType::F32,
            DimSelection::Dim(1),
        ),
        (
            FailurePoint::FirstReduction,
            NumericElementType::F64,
            DimSelection::Dim(1),
        ),
        (
            FailurePoint::LaterReduction,
            NumericElementType::F64,
            DimSelection::All,
        ),
    ] {
        let provider = StageProvider::new(point);
        let source = provider
            .owner
            .allocate_zeroed(vec![2, 2], element, GpuTensorStorage::Real)
            .unwrap();
        let error = block_on(selected_sum(provider, source.handle.clone(), selection))
            .expect_err("completed provider work cannot replay on the source");
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        original_unsupported(&error);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert!(provider.restored.lock().unwrap().is_empty());
        assert_eq!(source.probe.retirements(), 0);
        source.handle.allocation_lease().validate_live().unwrap();
        let mut probes = provider.widened.lock().unwrap().clone();
        probes.extend(provider.mapped.lock().unwrap().iter().cloned());
        probes.extend(provider.reduced.lock().unwrap().iter().cloned());
        assert!(!probes.is_empty());
        for probe in probes {
            assert_eq!(probe.retirements(), 1);
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn dropping_a_polled_second_reduction_retires_every_owned_intermediate() {
    let provider = StageProvider::new(FailurePoint::PendingLaterReduction);
    let source = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let mut future = Box::pin(selected_sum(
        provider,
        source.handle.clone(),
        DimSelection::All,
    ));
    let waker = futures::task::noop_waker();
    assert!(matches!(
        future.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert_eq!(provider.reductions.load(Ordering::SeqCst), 2);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert!(provider.restored.lock().unwrap().is_empty());
    drop(future);
    assert_eq!(provider.mapped.lock().unwrap().len(), 1);
    assert_eq!(provider.reduced.lock().unwrap().len(), 1);
    assert_eq!(provider.mapped.lock().unwrap()[0].retirements(), 1);
    assert_eq!(provider.reduced.lock().unwrap()[0].retirements(), 1);
    assert_eq!(source.probe.retirements(), 0);
    source.handle.allocation_lease().validate_live().unwrap();
}
