//! Context tests use a real fixture allocation owner for physical backing
//! larger than a legal tensor view. InProcessProvider intentionally does not
//! accept element-count-changing views of its own allocations.

use runmat_accelerate_api::test_support::{TestAllocation, TestAllocationOwner};
use runmat_accelerate_api::{
    unsupported_provider_operation, AccelDownloadFuture, AccelProvider, GpuTensorHandle,
    GpuTensorStorage, GpuTensorView, HostTensorView, NumericElementType,
};

pub(crate) struct OwnedPhysicalProvider {
    allocations: TestAllocationOwner,
}

impl OwnedPhysicalProvider {
    pub(crate) fn new() -> Self {
        Self {
            allocations: TestAllocationOwner::new(),
        }
    }

    pub(crate) fn allocate_f64(&self, backing_shape: &[usize]) -> TestAllocation {
        self.allocations
            .allocate_zeroed(
                backing_shape.to_vec(),
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )
            .expect("fixture owns an actual F64 backing allocation")
    }

    pub(crate) fn allocate_padded_f64_view(
        &self,
        backing_shape: &[usize],
        view_shape: &[usize],
    ) -> (TestAllocation, GpuTensorHandle) {
        let allocation = self.allocate_f64(backing_shape);
        let view = allocation
            .handle
            .with_view(GpuTensorView::contiguous(view_shape.to_vec()))
            .expect("fixture owner admits a view within its real backing payload");
        assert_eq!(view.shape(), view_shape);
        assert_eq!(view.physical().element_type(), NumericElementType::F64);
        assert_eq!(view.physical().storage(), GpuTensorStorage::Real);
        assert_eq!(
            allocation
                .probe
                .with_data(&allocation.handle, |data| data.len())
                .expect("fixture payload remains owned"),
            usize::try_from(view.physical().element_capacity()).expect("test capacity fits usize")
        );
        (allocation, view)
    }
}

impl AccelProvider for OwnedPhysicalProvider {
    fn device_id(&self) -> u32 {
        self.allocations.device_id()
    }

    fn device_info(&self) -> String {
        "owned physical context-test allocation".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        Err(unsupported_provider_operation("fixture upload unavailable"))
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async {
            Err(unsupported_provider_operation(
                "fixture download unavailable",
            ))
        })
    }
}
