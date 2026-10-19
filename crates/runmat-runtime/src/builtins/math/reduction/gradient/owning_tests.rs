use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, HostTensorView, NumericElementType, ThreadProviderGuard,
};
use std::sync::Mutex;

struct Provider {
    owner: TestAllocationOwner,
    outputs: Mutex<Vec<RetirementProbe>>,
    alias: bool,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "gradient partial result ownership".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("failure must not silently gather") })
    }
    fn gradient_dim(
        &self,
        input: &GpuTensorHandle,
        dim: usize,
        _: f64,
    ) -> anyhow::Result<GpuTensorHandle> {
        if dim == 1 {
            if self.alias {
                return Ok(input.with_view(input.view().clone())?);
            }
            return Err(anyhow::Error::new(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "second gradient failed",
            )));
        }
        let result = self.owner.allocate_zeroed(
            input.shape().to_vec(),
            NumericElementType::F64,
            GpuTensorStorage::Real,
        )?;
        self.outputs.lock().unwrap().push(result.probe);
        Ok(result.handle)
    }
}

#[test]
fn later_gradient_failure_retires_prior_outputs_without_releasing_input() {
    for alias in [false, true] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            outputs: Mutex::new(Vec::new()),
            alias,
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let error = futures::executor::block_on(gradient_gpu_outputs(
            input.handle.clone(),
            &[1, 2],
            &[GradientSpacing::Scalar(1.0), GradientSpacing::Scalar(1.0)],
            false,
        ))
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        if alias {
            assert!(error.message().contains("malformed or aliased"));
        } else {
            let source = error.source.as_ref().expect("provider source retained");
            let provider = source
                .downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
                .expect("typed provider execution carrier retained");
            let cause = std::error::Error::source(provider)
                .expect("original provider cause retained")
                .downcast_ref::<std::io::Error>()
                .expect("original I/O error remains downcastable");
            assert_eq!(cause.kind(), std::io::ErrorKind::BrokenPipe);
            assert_eq!(cause.to_string(), "second gradient failed");
        }
        let outputs = provider.outputs.lock().unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].retirements(), 1);
        assert_eq!(input.probe.retirements(), 0);
        input.handle.allocation_lease().validate_live().unwrap();
    }
}
