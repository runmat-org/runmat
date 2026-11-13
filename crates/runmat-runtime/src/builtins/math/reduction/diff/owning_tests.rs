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
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result: anyhow::Result<_> = (|| match self.fault {
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
        })();
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
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
        let error = futures::executor::block_on(diff_gpu(
            input.handle.clone(),
            1,
            Some(1),
            &super::super::numerical_execution::NumericalCall::new(
                runmat_builtins::NumericalReductionKind::Difference,
                None,
            )
            .unwrap(),
        ))
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        if matches!(fault, Fault::Execution) {
            let source = error.source.as_ref().expect("provider source retained");
            let provider = source
                .downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
                .expect("typed provider execution carrier retained");
            let cause = std::error::Error::source(provider)
                .expect("original provider cause retained")
                .downcast_ref::<std::io::Error>()
                .expect("original I/O error remains downcastable");
            assert_eq!(cause.kind(), std::io::ErrorKind::BrokenPipe);
            assert_eq!(cause.to_string(), "difference failed");
        } else {
            assert_eq!(error.identifier(), DIFF_ERROR_INTERNAL.identifier);
        }
        for output in provider.outputs.lock().unwrap().iter() {
            assert_eq!(output.retirements(), 1);
        }
        assert_eq!(input.probe.retirements(), 0);
        input.handle.allocation_lease().validate_live().unwrap();
    }
}
