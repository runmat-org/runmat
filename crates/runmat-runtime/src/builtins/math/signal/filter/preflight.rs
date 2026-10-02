use super::*;

pub(super) fn operands(
    numerator: &Value,
    denominator: &Value,
    signal: &Value,
    initial: Option<&Value>,
) -> BuiltinResult<()> {
    coefficient(CoefficientRole::Numerator, numerator)?;
    coefficient(CoefficientRole::Denominator, denominator)?;
    signal_kind(signal)?;
    if let Some(initial) = initial {
        initial_kind(initial)?;
    }
    Ok(())
}

fn coefficient(role: CoefficientRole, value: &Value) -> BuiltinResult<()> {
    let shape = match value {
        Value::Tensor(tensor) => Some(tensor.shape.as_slice()),
        Value::ComplexTensor(tensor) => Some(tensor.shape.as_slice()),
        Value::LogicalArray(logical) => Some(logical.shape.as_slice()),
        Value::GpuTensor(handle) => Some(handle.shape()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => None,
        other => {
            return Err(filter_error_with_detail(
                &FILTER_ERROR_INVALID_COEFFICIENTS,
                format!("{} coefficients: unsupported {other:?}", role.label()),
            ))
        }
    };
    if let Some(shape) = shape {
        ensure_vector_shape(role, shape)?;
        if shape.contains(&0) {
            return Err(filter_error(role.empty_error()));
        }
    }
    if matches!(role, CoefficientRole::Denominator) && leading_coefficient_is_zero(value) {
        return Err(filter_error(&FILTER_ERROR_DENOMINATOR_A1_ZERO));
    }
    Ok(())
}

fn leading_coefficient_is_zero(value: &Value) -> bool {
    match value {
        Value::Num(value) => *value == 0.0,
        Value::Int(value) => value.is_zero(),
        Value::Bool(value) => !*value,
        Value::Complex(re, im) => *re == 0.0 && *im == 0.0,
        Value::Tensor(tensor) => tensor
            .numeric_value_at(0)
            .is_some_and(|value| value.is_zero()),
        Value::ComplexTensor(tensor) => tensor
            .numeric_value_at(0)
            .is_some_and(|(re, im)| re.is_zero() && im.is_zero()),
        Value::LogicalArray(logical) => logical.data.first() == Some(&false),
        Value::GpuTensor(_) => false,
        _ => false,
    }
}

fn signal_kind(value: &Value) -> BuiltinResult<()> {
    if matches!(
        value,
        Value::Tensor(_)
            | Value::ComplexTensor(_)
            | Value::LogicalArray(_)
            | Value::GpuTensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Complex(_, _)
    ) {
        Ok(())
    } else {
        Err(filter_error(&FILTER_ERROR_INVALID_SIGNAL))
    }
}

fn initial_kind(value: &Value) -> BuiltinResult<()> {
    if matches!(
        value,
        Value::Tensor(_)
            | Value::ComplexTensor(_)
            | Value::LogicalArray(_)
            | Value::GpuTensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Complex(_, _)
    ) {
        Ok(())
    } else {
        Err(filter_error(&FILTER_ERROR_INVALID_INITIAL_STATE))
    }
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
    fn known_nonzero_complex_imaginary_component_never_qualifies_as_real() {
        assert!(super::super::to_real_vec(&[Complex::new(1.0, 1e-300)]).is_none());
    }
}
