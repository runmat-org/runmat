use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned,
    HostNumericTensorOwned, HostTensorView, NumericElementType, ProviderTrapezoidSpacing,
    ThreadProviderGuard,
};
use std::sync::Mutex;

#[derive(Clone, Copy)]
enum Fault {
    Failure,
    InputAlias,
    SpacingAlias,
}
struct Provider {
    owner: TestAllocationOwner,
    uploads: Mutex<Vec<RetirementProbe>>,
    fault: Fault,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "cumtrapz spacing ownership fixture".into()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(host.data.to_vec()),
            shape: host.shape.to_vec(),
            storage: GpuTensorStorage::Real,
        })?;
        self.uploads.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("failure must not fall back to download") })
    }
    fn cumtrapz_dim(
        &self,
        input: &GpuTensorHandle,
        _: usize,
        spacing: ProviderTrapezoidSpacing<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result: anyhow::Result<_> = (|| match self.fault {
            Fault::Failure => Err(anyhow::Error::new(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "cumtrapz execution failed",
            ))),
            Fault::InputAlias => Ok(input.with_view(input.view().clone())?),
            Fault::SpacingAlias => {
                let handle = match spacing {
                    ProviderTrapezoidSpacing::Vector(handle)
                    | ProviderTrapezoidSpacing::Tensor(handle) => handle,
                    _ => anyhow::bail!("expected uploaded coordinates"),
                };
                Ok(handle.with_view(handle.view().clone())?)
            }
        })();
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
}

#[test]
fn cumulative_integration_retires_spacing_on_execution_failure_and_alias_rejection() {
    for fault in [Fault::Failure, Fault::InputAlias, Fault::SpacingAlias] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            uploads: Mutex::new(Vec::new()),
            fault,
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(vec![3, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let x = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).unwrap();
        let error = futures::executor::block_on(cumtrapz_builtin(
            None,
            Value::Tensor(x),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                input.handle.clone(),
            )]),
        ))
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        match fault {
            Fault::Failure => {
                let source = error.source.as_ref().expect("provider source retained");
                let provider = source
                    .downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
                    .expect("typed provider execution carrier retained");
                let cause = std::error::Error::source(provider)
                    .expect("original provider cause retained")
                    .downcast_ref::<std::io::Error>()
                    .expect("original I/O error remains downcastable");
                assert_eq!(cause.kind(), std::io::ErrorKind::BrokenPipe);
                assert_eq!(cause.to_string(), "cumtrapz execution failed");
            }
            _ => assert_eq!(error.identifier(), CUMTRAPZ_ERROR_INTERNAL.identifier),
        }
        let uploads = provider.uploads.lock().unwrap();
        assert_eq!(uploads.len(), 1);
        assert_eq!(uploads[0].retirements(), 1);
        assert_eq!(input.probe.retirements(), 0);
        input.handle.allocation_lease().validate_live().unwrap();
    }
}
