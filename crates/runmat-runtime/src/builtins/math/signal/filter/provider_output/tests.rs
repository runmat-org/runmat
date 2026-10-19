use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorHandle, HostTensorView,
    ProviderIirFilterOptions, ProviderIirFilterResult, ThreadProviderGuard,
};
use runmat_value::Value;

use crate::builtins::common::test_support;

struct NoHookProvider(InProcessProvider);

impl AccelProvider for NoHookProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.0.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.0.download(handle)
    }
    fn device_info(&self) -> String {
        "filter-no-hook".into()
    }
    fn device_id(&self) -> u32 {
        self.0.device_id()
    }
}

struct FailingHookProvider(InProcessProvider);

impl AccelProvider for FailingHookProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.0.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.0.download(handle)
    }
    fn device_info(&self) -> String {
        "filter-failing-hook".into()
    }
    fn device_id(&self) -> u32 {
        self.0.device_id()
    }
    fn query_feasibility(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        self.0.query_feasibility(query)
    }
    fn iir_filter<'a>(
        &'a self,
        _coefficients: runmat_accelerate_api::ProviderIirCoefficientSource<'a>,
        _x: &'a GpuTensorHandle,
        _options: ProviderIirFilterOptions,
    ) -> AccelProviderFuture<'a, ProviderIirFilterResult> {
        Box::pin(async { Err(anyhow::anyhow!("filter device execution failed")) })
    }
}

fn filter_resident(provider: &'static dyn AccelProvider) -> crate::BuiltinResult<Value> {
    let shape = [3, 1];
    let signal = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0],
            shape: &shape,
        })
        .expect("upload source signal");
    block_on(super::super::evaluate(
        Value::Num(1.0),
        Value::Num(1.0),
        Value::GpuTensor(signal),
        &[],
    ))
    .map(|evaluation| evaluation.into_value())
}

#[test]
fn default_no_hook_is_typed_unsupported_and_uses_host_reference() {
    let _lock = test_support::accel_test_lock();
    let provider: &'static NoHookProvider =
        Box::leak(Box::new(NoHookProvider(InProcessProvider::new())));
    let _scope = ThreadProviderGuard::set(Some(provider));
    let output = filter_resident(provider).expect("typed Unsupported fallback");
    let Value::Tensor(output) = output else {
        panic!("expected host reference output");
    };
    assert_eq!(output.shape(), vec![3, 1]);
    assert_eq!(output.materialize_f64(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn operational_hook_error_is_not_replayed_on_host() {
    let _lock = test_support::accel_test_lock();
    let provider: &'static FailingHookProvider =
        Box::leak(Box::new(FailingHookProvider(InProcessProvider::new())));
    let _scope = ThreadProviderGuard::set(Some(provider));
    let error = filter_resident(provider).expect_err("operational failure must terminate");
    assert_eq!(
        error.identifier(),
        super::super::FILTER_ERROR_PROVIDER_EXECUTION.identifier
    );
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(std::error::Error::source(&error).is_some());
}

#[test]
fn provider_result_cannot_publish_borrowed_signal_as_new_output() {
    let _lock = test_support::accel_test_lock();
    let provider: &'static NoHookProvider =
        Box::leak(Box::new(NoHookProvider(InProcessProvider::new())));
    let _scope = ThreadProviderGuard::set(Some(provider));
    let signal = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[2, 1],
        })
        .unwrap();
    let numerator = provider
        .upload(&HostTensorView {
            data: &[1.0],
            shape: &[1, 1],
        })
        .unwrap();
    let denominator = provider
        .upload(&HostTensorView {
            data: &[1.0],
            shape: &[1, 1],
        })
        .unwrap();
    let context = super::ProviderOutputContext {
        provider,
        signal: &signal,
        numerator: &numerator,
        denominator: &denominator,
        original_numerator: None,
        original_denominator: None,
        initial_state: None,
        state_shape: &[0, 1],
        state_len: 0,
    };
    let result = context.admit(ProviderIirFilterResult {
        output: signal.clone(),
        final_state: None,
    });
    let Err(error) = result else {
        panic!("alias must be rejected");
    };
    assert_eq!(
        error.identifier(),
        super::super::FILTER_ERROR_PROVIDER_OUTPUT.identifier
    );
    assert!(signal.allocation_lease().validate_live().is_ok());
}

#[test]
fn provider_result_cannot_retire_original_resident_coefficient() {
    let _lock = test_support::accel_test_lock();
    let provider: &'static NoHookProvider =
        Box::leak(Box::new(NoHookProvider(InProcessProvider::new())));
    let _scope = ThreadProviderGuard::set(Some(provider));
    let signal = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[2, 1],
        })
        .unwrap();
    let original = provider
        .upload(&HostTensorView {
            data: &[4.0, 5.0],
            shape: &[2, 1],
        })
        .unwrap();
    let numerator = provider
        .upload(&HostTensorView {
            data: &[1.0],
            shape: &[1, 1],
        })
        .unwrap();
    let denominator = provider
        .upload(&HostTensorView {
            data: &[1.0],
            shape: &[1, 1],
        })
        .unwrap();
    let context = super::ProviderOutputContext {
        provider,
        signal: &signal,
        numerator: &numerator,
        denominator: &denominator,
        original_numerator: Some(&original),
        original_denominator: None,
        initial_state: None,
        state_shape: &[0, 1],
        state_len: 0,
    };
    let error = context
        .admit(ProviderIirFilterResult {
            output: original.clone(),
            final_state: None,
        })
        .expect_err("original coefficient alias must not be published");
    assert_eq!(
        error.identifier(),
        super::super::FILTER_ERROR_PROVIDER_OUTPUT.identifier
    );
    assert!(original.allocation_lease().validate_live().is_ok());
}

#[test]
fn missing_final_state_retires_only_aliased_result_reference() {
    let _lock = test_support::accel_test_lock();
    let provider: &'static NoHookProvider =
        Box::leak(Box::new(NoHookProvider(InProcessProvider::new())));
    let _scope = ThreadProviderGuard::set(Some(provider));
    let signal = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[2, 1],
        })
        .unwrap();
    let original = provider
        .upload(&HostTensorView {
            data: &[3.0, 4.0],
            shape: &[2, 1],
        })
        .unwrap();
    let numerator = provider
        .upload(&HostTensorView {
            data: &[1.0, 1.0],
            shape: &[2, 1],
        })
        .unwrap();
    let denominator = provider
        .upload(&HostTensorView {
            data: &[1.0],
            shape: &[1, 1],
        })
        .unwrap();
    let context = super::ProviderOutputContext {
        provider,
        signal: &signal,
        numerator: &numerator,
        denominator: &denominator,
        original_numerator: Some(&original),
        original_denominator: None,
        initial_state: None,
        state_shape: &[1, 1],
        state_len: 1,
    };
    let error = context
        .admit(ProviderIirFilterResult {
            output: original.clone(),
            final_state: None,
        })
        .expect_err("missing state must reject the result, including a borrowed alias");
    assert_eq!(
        error.identifier(),
        super::super::FILTER_ERROR_PROVIDER_OUTPUT.identifier
    );
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(original.allocation_lease().validate_live().is_ok());
    assert!(signal.allocation_lease().validate_live().is_ok());
}

#[test]
fn invalid_primary_and_borrowed_state_retire_only_result_references() {
    let _lock = test_support::accel_test_lock();
    let provider: &'static NoHookProvider =
        Box::leak(Box::new(NoHookProvider(InProcessProvider::new())));
    let _scope = ThreadProviderGuard::set(Some(provider));
    let signal = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[2, 1],
        })
        .unwrap();
    let numerator = provider
        .upload(&HostTensorView {
            data: &[1.0, 1.0],
            shape: &[2, 1],
        })
        .unwrap();
    let denominator = provider
        .upload(&HostTensorView {
            data: &[1.0],
            shape: &[1, 1],
        })
        .unwrap();
    let original_state = provider
        .upload(&HostTensorView {
            data: &[9.0],
            shape: &[1, 1],
        })
        .unwrap();
    let invalid_output = provider
        .upload(&HostTensorView {
            data: &[7.0],
            shape: &[1, 1],
        })
        .unwrap();
    let context = super::ProviderOutputContext {
        provider,
        signal: &signal,
        numerator: &numerator,
        denominator: &denominator,
        original_numerator: None,
        original_denominator: None,
        initial_state: Some(&original_state),
        state_shape: &[1, 1],
        state_len: 1,
    };
    let error = context
        .admit(ProviderIirFilterResult {
            output: invalid_output.clone(),
            final_state: Some(original_state.clone()),
        })
        .expect_err("invalid primary must reject its paired borrowed state");
    assert_eq!(
        error.identifier(),
        super::super::FILTER_ERROR_PROVIDER_OUTPUT.identifier
    );
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(original_state.allocation_lease().validate_live().is_ok());
    assert!(signal.allocation_lease().validate_live().is_ok());
    assert!(invalid_output.allocation_lease().validate_live().is_ok());
}
