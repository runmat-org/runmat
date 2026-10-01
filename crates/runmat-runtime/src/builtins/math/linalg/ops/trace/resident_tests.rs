//! Capable-provider contracts not exercised by the current production providers.
use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelProvider, AccelProviderFuture,
    GpuTensorHandle, GpuTensorStorage, HostTensorView, NumericElementType, ProviderExecutionFuture,
    ProviderPrecision, ThreadProviderGuard,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct ComplexTraceProvider {
    owner: TestAllocationOwner,
    diagonals: AtomicUsize,
    sums: AtomicUsize,
    casts: AtomicUsize,
}

impl ComplexTraceProvider {
    fn allocate(&self, shape: Vec<usize>, element: NumericElementType) -> GpuTensorHandle {
        self.owner
            .allocate_zeroed(shape, element, GpuTensorStorage::ComplexInterleaved)
            .expect("fixture allocation")
            .handle
    }
}

impl AccelProvider for ComplexTraceProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "complex trace contract fixture".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unexpected download") })
    }

    fn diag_extract(&self, matrix: &GpuTensorHandle, _: isize) -> anyhow::Result<GpuTensorHandle> {
        self.diagonals.fetch_add(1, Ordering::SeqCst);
        if matrix.shape() == [1, 1] {
            return Ok(matrix.clone());
        }
        let element = match runmat_accelerate_api::handle_precision(matrix) {
            Some(ProviderPrecision::F32) => NumericElementType::F32,
            Some(ProviderPrecision::F64) => NumericElementType::F64,
            None => anyhow::bail!("floating input required"),
        };
        Ok(self.allocate(vec![matrix.shape()[0], 1], element))
    }

    fn reduce_sum<'a>(
        &'a self,
        diagonal: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.sums.fetch_add(1, Ordering::SeqCst);
            if diagonal.shape() == [1, 1]
                && runmat_accelerate_api::handle_precision(diagonal) == Some(ProviderPrecision::F64)
            {
                return Ok(diagonal.clone());
            }
            Ok(self.allocate(vec![1, 1], NumericElementType::F64))
        })
    }

    fn unary_single<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.casts.fetch_add(1, Ordering::SeqCst);
            Ok(self.allocate(vec![1, 1], NumericElementType::F32))
        })
    }
}

#[test]
fn complex_diagonal_and_sum_remain_resident_for_capable_provider() {
    for element in [NumericElementType::F64, NumericElementType::F32] {
        let provider: &'static ComplexTraceProvider = Box::leak(Box::new(ComplexTraceProvider {
            owner: TestAllocationOwner::new(),
            diagonals: AtomicUsize::new(0),
            sums: AtomicUsize::new(0),
            casts: AtomicUsize::new(0),
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider.allocate(vec![2, 2], element);
        let output = block_on(super::trace_builtin(Value::GpuTensor(input.clone()))).unwrap();
        let Value::GpuTensor(output) = output else {
            panic!("capable provider must keep complex trace resident")
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&output),
            GpuTensorStorage::ComplexInterleaved
        );
        assert_eq!(
            runmat_accelerate_api::handle_precision(&output),
            runmat_accelerate_api::handle_precision(&input)
        );
        assert_eq!(provider.diagonals.load(Ordering::SeqCst), 1);
        assert_eq!(provider.sums.load(Ordering::SeqCst), 1);
        assert_eq!(
            provider.casts.load(Ordering::SeqCst),
            usize::from(element == NumericElementType::F32)
        );
        input.allocation_lease().validate_live().unwrap();
    }
}

#[test]
fn scalar_complex_diagonal_and_sum_may_keep_the_input_allocation() {
    let provider: &'static ComplexTraceProvider = Box::leak(Box::new(ComplexTraceProvider {
        owner: TestAllocationOwner::new(),
        diagonals: AtomicUsize::new(0),
        sums: AtomicUsize::new(0),
        casts: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider.allocate(vec![1, 1], NumericElementType::F64);
    let output = block_on(super::trace_builtin(Value::GpuTensor(input.clone()))).unwrap();
    let Value::GpuTensor(output) = output else {
        panic!("scalar complex trace must stay resident")
    };
    assert!(output.shares_allocation_with(&input));
    assert_eq!(provider.diagonals.load(Ordering::SeqCst), 1);
    assert_eq!(provider.sums.load(Ordering::SeqCst), 1);
    input.allocation_lease().validate_live().unwrap();
}
