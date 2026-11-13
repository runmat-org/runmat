use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};
use std::sync::Mutex;

struct Provider {
    owner: TestAllocationOwner,
    malformed: bool,
    outputs: Mutex<Vec<RetirementProbe>>,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "moving ownership witness".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("no fallback after provider failure") })
    }
    fn moving_window<'a>(
        &'a self,
        _: &'a ProviderMovingWindowRequest<'a>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if !self.malformed {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "moving execution failed",
                )
                .into());
            }
            let output = self.owner.allocate_zeroed(
                vec![1, 1],
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )?;
            self.outputs.lock().unwrap().push(output.probe);
            Ok(output.handle)
        })
    }
}

#[test]
fn moving_rejects_malformed_results_and_preserves_execution_failure_without_gather() {
    for malformed in [false, true] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            malformed,
            outputs: Mutex::new(Vec::new()),
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(vec![3, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let parsed = ParsedMoving {
            window: WindowSpec::Counts {
                before: 1,
                after: 1,
            },
            dim: Some(1),
            nan_mode: NanMode::Include,
            endpoints: Endpoints::Shrink,
            sample_points: None,
            normalization: VarianceNormalization::Sample,
        };
        let work = admission::MovingCall::new(None, "movsum").unwrap();
        let error = futures::executor::block_on(provider::moving_gpu(
            &work,
            "movsum",
            MovingOp::Sum,
            input.handle.clone(),
            &parsed,
        ))
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        if malformed {
            assert!(error.message().contains("malformed"));
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
            assert_eq!(cause.to_string(), "moving execution failed");
        }
        for output in provider.outputs.lock().unwrap().iter() {
            assert_eq!(output.retirements(), 1);
        }
        assert_eq!(input.probe.retirements(), 0);
        input.handle.allocation_lease().validate_live().unwrap();
    }
}
