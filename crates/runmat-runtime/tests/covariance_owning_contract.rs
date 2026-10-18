use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::allocation::WeakAllocationLease;
use runmat_accelerate_api::*;
use runmat_runtime::{call_builtin_async_with_outputs, GpuGatherRetry};
use runmat_value::Value;
use std::sync::Mutex;

struct Provider {
    inner: InProcessProvider,
    fail: bool,
    trace: Mutex<Vec<&'static str>>,
    outputs: Mutex<Vec<(WeakAllocationLease, WeakAllocationLease)>>,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn upload(&self, host: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.trace.lock().unwrap().push("download");
        self.inner.download(input)
    }
    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.trace.lock().unwrap().push("download");
        self.inner.download_numeric(input)
    }
    fn covariance<'a>(
        &'a self,
        matrix: &'a GpuTensorHandle,
        second: Option<&'a GpuTensorHandle>,
        weights: Option<&'a GpuTensorHandle>,
        options: &'a CovarianceOptions,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.trace.lock().unwrap().push("covariance");
        Box::pin(async move {
            if self.fail {
                return Err(ProviderExecutionError::from_source(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "weights must be non-negative finite values: transport failed",
                )));
            }
            self.inner
                .covariance(matrix, second, weights, options)
                .await
        })
    }
    fn covariance_to_correlation(
        &self,
        input: &GpuTensorHandle,
    ) -> anyhow::Result<ProviderCovarianceToCorrelationResult> {
        self.trace.lock().unwrap().push("conversion");
        let result = self.inner.covariance_to_correlation(input)?;
        self.outputs.lock().unwrap().push((
            result.correlation.allocation_lease().downgrade(),
            result.sigma.allocation_lease().downgrade(),
        ));
        Ok(result)
    }
    fn reshape(&self, input: &GpuTensorHandle, shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        self.trace.lock().unwrap().push("reshape");
        self.inner.reshape(input, shape)
    }
}
fn provider(fail: bool) -> &'static Provider {
    Box::leak(Box::new(Provider {
        inner: Default::default(),
        fail,
        trace: Mutex::new(vec![]),
        outputs: Mutex::new(vec![]),
    }))
}
fn upload(provider: &Provider, data: &[f64], shape: &[usize]) -> GpuTensorHandle {
    provider.upload(&HostTensorView { data, shape }).unwrap()
}

#[test]
fn resident_weight_domain_rejections_keep_exact_causes_and_borrowed_inputs() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for (bad, domain) in [
        (-1.0, CovarianceWeightDomainError::Negative),
        (f64::NAN, CovarianceWeightDomainError::NonFinite),
        (f64::INFINITY, CovarianceWeightDomainError::NonFinite),
    ] {
        let provider = provider(false);
        let _active = ThreadProviderGuard::set(Some(provider));
        let input = upload(provider, &[1., 2., 3., 4.], &[2, 2]);
        let weights = upload(provider, &[1., bad], &[2, 1]);
        let input_weak = input.allocation_lease().downgrade();
        let weight_weak = weights.allocation_lease().downgrade();
        let error = block_on(call_builtin_async_with_outputs(
            "cov",
            &[
                Value::GpuTensor(input.clone()),
                Value::GpuTensor(weights.clone()),
            ],
            1,
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:cov:InvalidArgument"));
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
        let carrier = error
            .source
            .as_ref()
            .unwrap()
            .downcast_ref::<ProviderExecutionError>()
            .unwrap();
        let mut source: &(dyn std::error::Error + 'static) = carrier;
        loop {
            if let Some(actual) = source.downcast_ref::<CovarianceWeightDomainError>() {
                assert_eq!(*actual, domain);
                break;
            }
            source = source.source().expect("original typed weight rejection");
        }
        assert_eq!(*provider.trace.lock().unwrap(), ["covariance"]);
        assert!(input_weak.acquire().is_some());
        assert!(weight_weak.acquire().is_some());
        drop((input, weights));
        assert!(input_weak.acquire().is_none());
        assert!(weight_weak.acquire().is_none());
    }
}

#[test]
fn malformed_weights_precede_provider_work_and_untyped_failures_stay_internal() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let provider = provider(true);
    let _active = ThreadProviderGuard::set(Some(provider));
    let input = upload(provider, &[1., 2., 3., 4.], &[2, 2]);
    let malformed = upload(provider, &[1., 1., 1.], &[3, 1]);
    // Explicit third array selects the weight operand independently of the
    // two-array dataset-vs-weight inference.
    let error = block_on(call_builtin_async_with_outputs(
        "cov",
        &[
            Value::GpuTensor(input.clone()),
            Value::GpuTensor(input.clone()),
            Value::GpuTensor(malformed),
        ],
        1,
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:cov:WeightVectorLengthMismatch")
    );
    assert!(provider.trace.lock().unwrap().is_empty());
    let weights = upload(provider, &[1., 1.], &[2, 1]);
    let error = block_on(call_builtin_async_with_outputs(
        "cov",
        &[Value::GpuTensor(input.clone()), Value::GpuTensor(weights)],
        1,
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:cov:Internal"));
    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    let carrier = error
        .source
        .as_ref()
        .unwrap()
        .downcast_ref::<ProviderExecutionError>()
        .unwrap();
    let source = std::error::Error::source(carrier).unwrap();
    assert_eq!(
        source.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::BrokenPipe
    );
    assert_eq!(*provider.trace.lock().unwrap(), ["covariance"]);
}

#[test]
fn conversion_public_arity_retains_only_returned_physical_allocations() {
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for name in ["corrcov", "cov2corr"] {
        for count in 0..=3 {
            let provider = provider(false);
            let _active = ThreadProviderGuard::set(Some(provider));
            let input = upload(provider, &[4., 0., 0., 9.], &[2, 2]);
            let borrowed = input.allocation_lease().downgrade();
            let result = block_on(call_builtin_async_with_outputs(
                name,
                &[Value::GpuTensor(input.clone())],
                count,
            ));
            if count == 3 {
                assert_eq!(
                    result.unwrap_err().identifier(),
                    Some(format!("RunMat:{name}:InvalidArgument").as_str())
                );
                assert!(provider.trace.lock().unwrap().is_empty());
            } else if count == 0 {
                assert_eq!(result.unwrap(), Value::output_list(vec![]));
                assert!(provider.trace.lock().unwrap().is_empty());
            } else {
                let result = result.unwrap();
                let outputs = provider.outputs.lock().unwrap();
                assert_eq!(outputs.len(), 1);
                let (correlation, sigma) = &outputs[0];
                assert_eq!(
                    correlation.acquire().is_some(),
                    name == "corrcov" || count == 2
                );
                assert_eq!(sigma.acquire().is_some(), name == "cov2corr" || count == 2);
                let Value::OutputList(values) = result.value() else {
                    panic!("requested outputs")
                };
                assert_eq!(values.len(), count);
                for (index, value) in values.iter().enumerate() {
                    let Value::GpuTensor(handle) = value else {
                        panic!("resident result")
                    };
                    assert!(!handle.shares_allocation_with(&input));
                    let host = block_on(provider.inner.download(handle)).unwrap();
                    let is_correlation = (name == "corrcov") == (index == 0);
                    if is_correlation {
                        assert_eq!(host.shape, [2, 2]);
                        assert_eq!(host.data, [1.0, 0.0, 0.0, 1.0]);
                    } else {
                        assert_eq!(
                            host.shape,
                            if name == "cov2corr" {
                                vec![1, 2]
                            } else {
                                vec![2, 1]
                            }
                        );
                        assert_eq!(host.data, [2.0, 3.0]);
                    }
                }
                drop(result);
                assert!(correlation.acquire().is_none());
                assert!(sigma.acquire().is_none());
            }
            assert!(borrowed.acquire().is_some());
            drop(input);
            assert!(borrowed.acquire().is_none());
        }
    }
}
