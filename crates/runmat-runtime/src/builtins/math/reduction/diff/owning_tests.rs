use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, GpuTensorStorage, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};
use std::sync::Mutex;

#[derive(Clone, Copy)]
enum Fault {
    Execution,
    Alias,
    Shape,
}
struct Provider {
    owner: TestAllocationOwner,
    fault: Fault,
    outputs: Mutex<Vec<RetirementProbe>>,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "diff owning result fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("execution failures must not gather") })
    }
    fn diff_dim(
        &self,
        input: &GpuTensorHandle,
        _: usize,
        _: usize,
    ) -> anyhow::Result<GpuTensorHandle> {
        match self.fault {
            Fault::Execution => {
                Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "difference failed").into())
            }
            Fault::Alias => Ok(input.with_view(input.view().clone().with_shape(vec![2, 1]))?),
            Fault::Shape => {
                let allocation = self.owner.allocate_zeroed(
                    vec![3, 1],
                    NumericElementType::F64,
                    GpuTensorStorage::Real,
                )?;
                self.outputs.lock().unwrap().push(allocation.probe);
                Ok(allocation.handle)
            }
        }
    }
}

#[test]
fn rejected_difference_outputs_release_only_owned_result_and_keep_execution_source() {
    for fault in [Fault::Execution, Fault::Alias, Fault::Shape] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            fault,
            outputs: Mutex::new(Vec::new()),
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(vec![3, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let error =
            futures::executor::block_on(diff_gpu(input.handle.clone(), 1, Some(1))).unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        if matches!(fault, Fault::Execution) {
            assert!(error
                .source
                .as_ref()
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .is_some());
        } else {
            assert!(error.message().contains("malformed or aliased"));
        }
        for output in provider.outputs.lock().unwrap().iter() {
            assert_eq!(output.retirements(), 1);
        }
        assert_eq!(input.probe.retirements(), 0);
        input.handle.allocation_lease().validate_live().unwrap();
    }
}
