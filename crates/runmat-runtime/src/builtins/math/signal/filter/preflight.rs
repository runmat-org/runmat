use super::*;
use crate::builtins::math::signal::common::{
    preflight_iir_operands, IirAdmissionFault, IirFaultKind, IirForm, IirOperand,
};

pub(super) fn operands(
    numerator: &Value,
    denominator: &Value,
    signal: &Value,
    initial: Option<&Value>,
) -> BuiltinResult<crate::builtins::math::signal::common::IirAdmittedGeometry> {
    preflight_iir_operands(IirForm::Causal, numerator, denominator, signal, initial)
        .map_err(own_error)
}

pub(super) fn resident_owners(
    numerator: &Value,
    denominator: &Value,
    signal: &Value,
    initial: Option<&Value>,
    dimension: Option<&Value>,
) -> BuiltinResult<()> {
    let roles = [
        (Some(numerator), &FILTER_ERROR_INVALID_COEFFICIENTS),
        (Some(denominator), &FILTER_ERROR_INVALID_COEFFICIENTS),
        (Some(signal), &FILTER_ERROR_INVALID_SIGNAL),
        (initial, &FILTER_ERROR_INVALID_INITIAL_STATE),
        (dimension, &FILTER_ERROR_INVALID_DIMENSION),
    ];
    // Validate ALL exact sources before evaluating compute policy for any
    // role. A later stale/foreign source must dominate an earlier policy-only
    // decline, with the domain's original per-role diagnostic intact.
    for (value, descriptor) in roles {
        if let Some(value @ Value::GpuTensor(handle)) = value {
            crate::builtins::common::validation::validate_resident_metadata(value).map_err(
                |source| {
                    let mut error = filter_error_with_source(
                        descriptor,
                        "resident operand metadata is invalid",
                        source,
                    );
                    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    error
                },
            )?;
            handle
                .view()
                .validate(handle.physical())
                .map_err(|source| {
                    let mut error = filter_error(descriptor);
                    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    crate::runtime_error::with_provider_execution_source(
                        error,
                        runmat_accelerate_api::ProviderExecutionError::from_source(source),
                    )
                })?;
        }
    }
    for (value, descriptor) in roles {
        if let Some(Value::GpuTensor(handle)) = value {
            gpu_helpers::require_resident_compute_owner(handle, || filter_error(descriptor))?;
        }
    }
    Ok(())
}

fn own_error(fault: IirAdmissionFault) -> RuntimeError {
    let descriptor = match (fault.operand, fault.kind) {
        (IirOperand::Numerator, IirFaultKind::Empty) => &FILTER_ERROR_EMPTY_NUMERATOR,
        (IirOperand::Denominator, IirFaultKind::Empty) => &FILTER_ERROR_EMPTY_DENOMINATOR,
        (_, IirFaultKind::LeadingDenominatorZero) => &FILTER_ERROR_DENOMINATOR_A1_ZERO,
        (IirOperand::Numerator | IirOperand::Denominator, _) => &FILTER_ERROR_INVALID_COEFFICIENTS,
        (IirOperand::Signal, _) => &FILTER_ERROR_INVALID_SIGNAL,
        (IirOperand::InitialState, _) => &FILTER_ERROR_INVALID_INITIAL_STATE,
    };
    filter_error_with_detail(descriptor, format!("{:?}: {:?}", fault.operand, fault.kind))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_accelerate::simple_provider::InProcessProvider;
    use runmat_accelerate_api::{
        AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
        HostTensorView, ThreadProviderGuard,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountedProvider {
        delegate: InProcessProvider,
        uploads: AtomicUsize,
        downloads: AtomicUsize,
    }

    impl AccelProvider for CountedProvider {
        fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            self.uploads.fetch_add(1, Ordering::SeqCst);
            self.delegate.upload(view)
        }
        fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            self.delegate.download(handle)
        }
        fn download_numeric<'a>(
            &'a self,
            handle: &'a GpuTensorHandle,
        ) -> AccelNumericDownloadFuture<'a> {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            self.delegate.download_numeric(handle)
        }
        fn device_info(&self) -> String {
            "filter-preflight-counted".to_owned()
        }
        fn device_id(&self) -> u32 {
            self.delegate.device_id()
        }
    }

    #[test]
    fn later_host_invalid_coefficient_and_dimension_precede_resident_readback() {
        let _lock = crate::builtins::common::test_support::accel_test_lock();
        let provider: &'static CountedProvider = Box::leak(Box::new(CountedProvider {
            delegate: InProcessProvider::new(),
            uploads: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
        }));
        let _scope = ThreadProviderGuard::set(Some(provider));
        let b = provider
            .upload(&HostTensorView {
                data: &[1.0, 1.0],
                shape: &[1, 2],
            })
            .expect("resident numerator");
        let empty_a = Tensor::new(Vec::new(), vec![0, 1]).expect("empty denominator");
        let Err(error) = block_on(super::super::evaluate(
            Value::GpuTensor(b.clone()),
            Value::Tensor(empty_a),
            Value::Num(1.0),
            &[],
        )) else {
            panic!("known empty denominator must fail");
        };
        assert_eq!(
            error.identifier(),
            FILTER_ERROR_EMPTY_DENOMINATOR.identifier
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);

        let Err(error) = block_on(super::super::evaluate(
            Value::GpuTensor(b.clone()),
            Value::Num(1.0),
            Value::Num(1.0),
            &[
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Num(0.0),
            ],
        )) else {
            panic!("known invalid dimension must fail");
        };
        assert_eq!(
            error.identifier(),
            FILTER_ERROR_INVALID_DIMENSION.identifier
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        b.release().expect("release borrowed numerator");
    }

    #[test]
    fn later_stale_signal_precedes_numerator_download() {
        let _lock = crate::builtins::common::test_support::accel_test_lock();
        let provider = crate::builtins::math::signal::common::convolution::test_support::RejectTransferProvider::new();
        let _scope = ThreadProviderGuard::set(Some(provider));
        let numerator = provider.input(vec![1.0, 1.0], vec![1, 2]);
        let signal = provider.input(vec![1.0, 2.0], vec![1, 2]);
        signal.probe.mark_unavailable();
        let Err(error) = block_on(super::super::evaluate(
            Value::Num(1.0),
            Value::Num(1.0),
            Value::GpuTensor(signal.handle.clone()),
            &[],
        )) else {
            panic!("signal lease must be checked before coefficient uploads");
        };
        assert_eq!(error.identifier(), FILTER_ERROR_INVALID_SIGNAL.identifier);
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(std::error::Error::source(&error).is_some());
        assert_eq!(provider.downloads(), 0);
        assert!(numerator.handle.allocation_lease().validate_live().is_ok());
        assert_eq!(numerator.probe.retirements(), 0);
    }

    #[test]
    fn mixed_resident_owners_fall_back_without_coefficient_uploads() {
        let _lock = crate::builtins::common::test_support::accel_test_lock();
        let signal_provider: &'static CountedProvider = Box::leak(Box::new(CountedProvider {
            delegate: InProcessProvider::new(),
            uploads: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
        }));
        let state_provider: &'static CountedProvider = Box::leak(Box::new(CountedProvider {
            delegate: InProcessProvider::new(),
            uploads: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
        }));
        // Both leaked fixtures remain valid for the process lifetime and have
        // distinct provider-assigned device identifiers.
        unsafe {
            runmat_accelerate_api::register_device_provider(signal_provider);
            runmat_accelerate_api::register_device_provider(state_provider);
        }
        let _scope = ThreadProviderGuard::set(Some(signal_provider));
        let signal = signal_provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[1, 2],
            })
            .expect("signal");
        let state = state_provider
            .upload(&HostTensorView {
                data: &[10.0],
                shape: &[1, 1],
            })
            .expect("initial state");
        let signal_uploads = signal_provider.uploads.load(Ordering::SeqCst);
        let state_uploads = state_provider.uploads.load(Ordering::SeqCst);
        let result = block_on(super::super::evaluate(
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
            Value::GpuTensor(signal.clone()),
            &[Value::GpuTensor(state.clone())],
        ))
        .expect("mixed-owner inputs retain supported host execution");
        assert_eq!(
            signal_provider.uploads.load(Ordering::SeqCst),
            signal_uploads
        );
        assert_eq!(state_provider.uploads.load(Ordering::SeqCst), state_uploads);
        assert_eq!(signal_provider.downloads.load(Ordering::SeqCst), 1);
        assert_eq!(state_provider.downloads.load(Ordering::SeqCst), 1);
        let (output, _) = result.into_pair();
        let Value::Tensor(output) = output else {
            panic!("mixed-owner host route must produce a host tensor")
        };
        assert_eq!(output.materialize_f64(), vec![11.0, 2.0]);
        assert!(signal.allocation_lease().validate_live().is_ok());
        assert!(state.allocation_lease().validate_live().is_ok());
    }

    #[test]
    fn invalid_host_initial_state_precedes_resident_coefficient_download() {
        let _lock = crate::builtins::common::test_support::accel_test_lock();
        let provider: &'static CountedProvider = Box::leak(Box::new(CountedProvider {
            delegate: InProcessProvider::new(),
            uploads: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
        }));
        let _scope = ThreadProviderGuard::set(Some(provider));
        let numerator = provider
            .upload(&HostTensorView {
                data: &[1.0, 0.0],
                shape: &[1, 2],
            })
            .expect("resident numerator");
        let Err(error) = block_on(super::super::evaluate(
            Value::GpuTensor(numerator.clone()),
            Value::Num(1.0),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            &[Value::Tensor(
                Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap(),
            )],
        )) else {
            panic!("known invalid initial state must precede resident coefficient readback")
        };
        assert_eq!(
            error.identifier(),
            FILTER_ERROR_INVALID_INITIAL_STATE.identifier
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert!(numerator.allocation_lease().validate_live().is_ok());
    }

    #[test]
    fn provider_without_causal_hook_rejects_before_coefficient_uploads() {
        let _lock = crate::builtins::common::test_support::accel_test_lock();
        let provider: &'static CountedProvider = Box::leak(Box::new(CountedProvider {
            delegate: InProcessProvider::new(),
            uploads: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
        }));
        let _scope = ThreadProviderGuard::set(Some(provider));
        let signal = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0],
                shape: &[1, 3],
            })
            .expect("resident signal");
        let uploads_before = provider.uploads.load(Ordering::SeqCst);
        let (output, _) = block_on(super::super::evaluate(
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            Value::GpuTensor(signal.clone()),
            &[],
        ))
        .expect("unsupported provider uses the host reference route")
        .into_pair();
        assert_eq!(provider.uploads.load(Ordering::SeqCst), uploads_before);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
        let Value::Tensor(output) = output else {
            panic!("unsupported provider must return the host reference result")
        };
        assert_eq!(output.materialize_f64(), vec![1.0, 3.0, 5.0]);
        assert!(signal.allocation_lease().validate_live().is_ok());
    }

    #[test]
    fn known_nonzero_complex_imaginary_component_never_qualifies_as_real() {
        let mut cooperation = SignalCooperation::new(None, BUILTIN_NAME);
        assert!(block_on(super::super::preparation::to_real_vec(
            &[Complex::new(1.0, 1e-300)],
            &mut cooperation
        ))
        .unwrap()
        .is_none());
    }
}
