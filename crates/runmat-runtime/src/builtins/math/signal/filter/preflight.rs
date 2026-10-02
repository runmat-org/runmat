use super::*;
use crate::builtins::math::signal::common::{
    preflight_iir_operands, IirAdmissionFault, IirFaultKind, IirForm, IirOperand,
};

pub(super) fn operands(
    numerator: &Value,
    denominator: &Value,
    signal: &Value,
    initial: Option<&Value>,
) -> BuiltinResult<()> {
    preflight_iir_operands(IirForm::Causal, numerator, denominator, signal, initial)
        .map(|_| ())
        .map_err(own_error)
}

pub(super) fn resident_owners(
    numerator: &Value,
    denominator: &Value,
    signal: &Value,
    initial: Option<&Value>,
    dimension: Option<&Value>,
) -> BuiltinResult<()> {
    for (value, descriptor) in [
        (Some(numerator), &FILTER_ERROR_INVALID_COEFFICIENTS),
        (Some(denominator), &FILTER_ERROR_INVALID_COEFFICIENTS),
        (Some(signal), &FILTER_ERROR_INVALID_SIGNAL),
        (initial, &FILTER_ERROR_INVALID_INITIAL_STATE),
        (dimension, &FILTER_ERROR_INVALID_DIMENSION),
    ] {
        if let Some(Value::GpuTensor(handle)) = value {
            super::super::resident_output::owner(handle, || filter_error(descriptor))?;
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
        downloads: AtomicUsize,
    }

    impl AccelProvider for CountedProvider {
        fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
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
        let provider: &'static CountedProvider = Box::leak(Box::new(CountedProvider {
            delegate: InProcessProvider::new(),
            downloads: AtomicUsize::new(0),
        }));
        let _scope = ThreadProviderGuard::set(Some(provider));
        let numerator = provider
            .upload(&HostTensorView {
                data: &[1.0, 1.0],
                shape: &[1, 2],
            })
            .unwrap();
        let signal = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[1, 2],
            })
            .unwrap();
        let stale_signal = signal.clone();
        signal.release().unwrap();
        let Err(error) = block_on(super::super::evaluate(
            Value::GpuTensor(numerator.clone()),
            Value::Num(1.0),
            Value::GpuTensor(stale_signal),
            &[],
        )) else {
            panic!("signal lease must be checked before numerator transfer");
        };
        assert_eq!(error.identifier(), FILTER_ERROR_INVALID_SIGNAL.identifier);
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(std::error::Error::source(&error).is_some());
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert!(numerator.allocation_lease().validate_live().is_ok());
    }

    #[test]
    fn known_nonzero_complex_imaginary_component_never_qualifies_as_real() {
        assert!(super::super::to_real_vec(&[Complex::new(1.0, 1e-300)]).is_none());
    }
}
