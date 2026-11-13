//! Resident Single omit-NaN stage and ownership witnesses.
use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, GpuTensorStorage, HostTensorView, NumericElementType, ThreadProviderGuard,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

struct OmitProvider {
    owner: TestAllocationOwner,
    source: Mutex<Option<GpuTensorHandle>>,
    widened: Mutex<Vec<RetirementProbe>>,
    mapped: Mutex<Vec<RetirementProbe>>,
    reduced: Mutex<Vec<RetirementProbe>>,
    downloads: AtomicUsize,
    malformed_map: bool,
    fail_widen_retirement: bool,
}

impl OmitProvider {
    fn new(malformed_map: bool, fail_widen_retirement: bool) -> Self {
        Self {
            owner: TestAllocationOwner::new(),
            source: Mutex::new(None),
            widened: Mutex::new(Vec::new()),
            mapped: Mutex::new(Vec::new()),
            reduced: Mutex::new(Vec::new()),
            downloads: AtomicUsize::new(0),
            malformed_map,
            fail_widen_retirement,
        }
    }

    fn source(&self) -> runmat_accelerate_api::test_support::TestAllocation {
        let source = self
            .owner
            .allocate_zeroed(vec![2, 2], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap();
        *self.source.lock().unwrap() = Some(source.handle.clone());
        source
    }

    fn output(
        &self,
        shape: Vec<usize>,
        element: NumericElementType,
        probes: &Mutex<Vec<RetirementProbe>>,
    ) -> GpuTensorHandle {
        let output = self
            .owner
            .allocate_zeroed(shape, element, GpuTensorStorage::Real)
            .unwrap();
        probes.lock().unwrap().push(output.probe);
        output.handle
    }
}

impl AccelProvider for OmitProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "single sum omit stage fixture".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("sum must not gather in this fixture") })
    }

    fn unary_double<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let output = self.output(
                input.shape().to_vec(),
                NumericElementType::F64,
                &self.widened,
            );
            if self.fail_widen_retirement {
                self.widened
                    .lock()
                    .unwrap()
                    .last()
                    .unwrap()
                    .fail_retirement("widened release failed");
            }
            Ok(output)
        })
    }

    fn map_nan_to_zero(
        &self,
        input: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        assert_eq!(input.physical().element_type(), NumericElementType::F64);
        if self.malformed_map {
            return Ok(self.source.lock().unwrap().as_ref().unwrap().clone());
        }
        Ok(self.output(
            input.shape().to_vec(),
            NumericElementType::F64,
            &self.mapped,
        ))
    }

    fn reduce_sum_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            assert_eq!(input.physical().element_type(), NumericElementType::F64);
            let mut shape = input.shape().to_vec();
            shape[dim] = 1;
            Ok(self.output(shape, NumericElementType::F64, &self.reduced))
        })
    }

    fn unary_single<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            Ok(self.output(
                input.shape().to_vec(),
                NumericElementType::F32,
                &self.reduced,
            ))
        })
    }
}

fn omit_first_axis(source: GpuTensorHandle) -> BuiltinResult<Value> {
    block_on(sum_with_nan_policy(
        Value::GpuTensor(source),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
        Some(ReductionNaN::Omit),
    ))
}

#[test]
fn single_omit_widens_maps_reduces_and_publishes_without_download() {
    let provider = Box::leak(Box::new(OmitProvider::new(false, false)));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let source = provider.source();
    let Value::GpuTensor(output) = omit_first_axis(source.handle.clone()).unwrap() else {
        panic!("resident Single output expected")
    };
    assert_eq!(output.physical().element_type(), NumericElementType::F32);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.widened.lock().unwrap()[0].retirements(), 1);
    assert_eq!(provider.mapped.lock().unwrap()[0].retirements(), 1);
    assert_eq!(source.probe.retirements(), 0);
    source.handle.allocation_lease().validate_live().unwrap();
    output.release().unwrap();
}

#[test]
fn malformed_map_and_widening_retirement_fail_without_gather_or_borrowed_release() {
    for (malformed_map, fail_widen_retirement) in [(true, false), (false, true)] {
        let provider = Box::leak(Box::new(OmitProvider::new(
            malformed_map,
            fail_widen_retirement,
        )));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let source = provider.source();
        let error = omit_first_axis(source.handle.clone()).unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert_eq!(source.probe.retirements(), 0);
        source.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(provider.widened.lock().unwrap()[0].retirements(), 1);
        if fail_widen_retirement {
            assert_eq!(provider.mapped.lock().unwrap()[0].retirements(), 1);
            assert!(error.source.is_some());
        } else {
            assert!(provider.mapped.lock().unwrap().is_empty());
        }
    }
}
