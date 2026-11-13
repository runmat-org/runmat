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
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result: anyhow::Result<_> = (|| {
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
        })();
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
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
            &super::super::numerical_execution::NumericalCall::new(
                runmat_builtins::NumericalReductionKind::Gradient,
                None,
            )
            .unwrap(),
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

#[test]
fn between_output_cancellation_retains_failed_physical_retirement_and_primary_cause() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        outputs: Mutex::new(Vec::new()),
        alias: false,
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_cancellation_token(cancellation.token());
    let call = super::super::numerical_execution::NumericalCall::new(
        runmat_builtins::NumericalReductionKind::Gradient,
        Some(&runtime),
    )
    .unwrap();
    // Produce and adopt an actual first result, then inject cancellation at
    // the same checkpoint the multi-output loop uses before its next call.
    let mut outputs = futures::executor::block_on(gradient_gpu_outputs(
        input.handle.clone(),
        &[1],
        &[GradientSpacing::Scalar(1.0)],
        false,
        &call,
    ))
    .unwrap();
    let probes = provider.outputs.lock().unwrap();
    assert_eq!(probes.len(), 1);
    probes[0].fail_retirement("between-output physical retirement witness");
    cancellation.cancel(runmat_execution::CancellationReason::User);
    let primary = call.check().unwrap_err();
    let error = check_gradient_outputs(&call, &mut outputs).unwrap_err();
    assert!(outputs.is_empty());
    assert_eq!(error.identifier(), primary.identifier());
    assert_eq!(error.context.builtin, primary.context.builtin);
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(error.message().starts_with(primary.message()));
    assert!(error
        .message()
        .contains("between-output physical retirement witness"));
    // The rejection carrier retains the original control source alongside
    // the failed physical-release owner, rather than discarding Drop's result.
    let mut source = std::error::Error::source(&error);
    let mut found_control = false;
    while let Some(cause) = source {
        found_control |= matches!(
            cause.downcast_ref::<crate::execution::ExecutionServiceError>(),
            Some(crate::execution::ExecutionServiceError::Cancelled)
        );
        source = cause.source();
    }
    assert!(
        found_control,
        "original cancellation control cause remains downcastable"
    );
    assert_eq!(probes[0].retirements(), 1);
    assert_eq!(input.probe.retirements(), 0);
    input.handle.allocation_lease().validate_live().unwrap();
}
