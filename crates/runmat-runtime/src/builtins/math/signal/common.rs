use num_complex::Complex;
use runmat_accelerate_api::{GpuTensorHandle, ProviderSpectralRange};
use runmat_builtins::WindowOption;
pub(crate) use runmat_builtins::{WindowOutputType, WindowSampling};
use runmat_value::{ComplexTensor, NumericDType, Tensor, Value};

use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(crate) mod convolution;
mod cooperation;
mod iir_boundary;
pub(super) mod output_precision;
mod waveform_boundary;
mod waveform_control;
mod waveform_map;
#[cfg(test)]
pub(crate) mod waveform_restore_fixture;
mod waveform_sample;
mod window_generation;
pub(crate) mod window_provider;
pub(crate) use cooperation::{SignalCooperation, SIGNAL_WORK_CHUNK};
pub(crate) use iir_boundary::preflight as preflight_iir_operands;
pub(crate) use iir_boundary::{
    IirAdmissionFault, IirAdmittedGeometry, IirFaultKind, IirForm, IirOperand,
};
pub(crate) use waveform_boundary::{
    admit_resident_waveform_operand, prepared_real_integer_tensor, waveform_input_tensor,
};
pub(crate) use waveform_control::{PreparedWaveformControl, WaveformControlError};
pub(crate) use waveform_map::{map_real_waveform, WaveformMapError, WaveformOutputMode};
pub(crate) use waveform_sample::{admit_real_waveform_sample, WaveformSampleError};
pub(crate) use window_generation::{
    hamming_coefficient, symmetric_window_values, window_allocation_error, window_tensor,
};

const EPS: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WindowOptions {
    pub len: usize,
    pub sampling: WindowSampling,
    pub output_type: WindowOutputType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WindowControls {
    pub sampling: WindowSampling,
    pub output_type: WindowOutputType,
}

#[derive(Debug)]
pub(crate) enum WindowArgError {
    TooManyOptions,
    InvalidLength,
    InvalidOptionType,
    Text(runmat_value::Utf16DecodeError),
    UnknownOption(String),
    Allocation(std::collections::TryReserveError),
    TensorBuild(String),
}

#[derive(Debug, Clone)]
pub(crate) struct ComplexVectorInput {
    pub data: Vec<Complex<f64>>,
    pub shape: Vec<usize>,
    pub is_complex: bool,
    pub gpu_handle: Option<GpuTensorHandle>,
}

pub(crate) fn signal_error(
    builtin: &'static str,
    identifier: Option<&'static str>,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(crate) async fn value_to_complex_vector(
    builtin: &'static str,
    label: &str,
    value: Value,
) -> BuiltinResult<ComplexVectorInput> {
    crate::builtins::common::validation::reject_typed_complex_integer(&value, builtin)?;
    let (value, gpu_handle) = match value {
        Value::GpuTensor(handle) => {
            let provider = super::resident_output::owner(&handle, || {
                signal_error(
                    builtin,
                    None,
                    format!("{builtin}: invalid resident {label}"),
                )
            })?;
            let value = gpu_helpers::download_value_preserving_residency_async(provider, &handle)
                .await
                .map_err(|source| {
                    let mut error = build_runtime_error(format!(
                        "{builtin}: failed to gather {label}: {}",
                        source.message()
                    ))
                    .with_builtin(builtin)
                    .with_source(source)
                    .build();
                    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    error
                })?;
            (value, Some(handle))
        }
        value => (value, None),
    };
    let mut input = host_value_to_complex_vector(builtin, label, value)?;
    input.gpu_handle = gpu_handle;
    Ok(input)
}

fn host_value_to_complex_vector(
    builtin: &'static str,
    label: &str,
    value: Value,
) -> BuiltinResult<ComplexVectorInput> {
    match value {
        Value::Tensor(tensor) => tensor_to_complex_vector(builtin, label, tensor),
        Value::ComplexTensor(tensor) => complex_tensor_to_complex_vector(builtin, label, tensor),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical).map_err(|e| {
                signal_error(builtin, None, format!("{builtin}: invalid {label}: {e}"))
            })?;
            tensor_to_complex_vector(builtin, label, tensor)
        }
        Value::Num(n) => Ok(ComplexVectorInput {
            data: vec![Complex::new(n, 0.0)],
            shape: vec![1, 1],
            is_complex: false,
            gpu_handle: None,
        }),
        Value::Int(i) => Ok(ComplexVectorInput {
            data: vec![Complex::new(i.to_f64(), 0.0)],
            shape: vec![1, 1],
            is_complex: false,
            gpu_handle: None,
        }),
        Value::Bool(b) => Ok(ComplexVectorInput {
            data: vec![Complex::new(if b { 1.0 } else { 0.0 }, 0.0)],
            shape: vec![1, 1],
            is_complex: false,
            gpu_handle: None,
        }),
        Value::Complex(re, im) => Ok(ComplexVectorInput {
            data: vec![Complex::new(re, im)],
            shape: vec![1, 1],
            is_complex: true,
            gpu_handle: None,
        }),
        other => Err(signal_error(
            builtin,
            None,
            format!("{builtin}: invalid {label}: received {other:?}"),
        )),
    }
}

fn tensor_to_complex_vector(
    builtin: &'static str,
    label: &str,
    tensor: Tensor,
) -> BuiltinResult<ComplexVectorInput> {
    ensure_vector_shape(builtin, label, tensor.shape())?;
    let shape = tensor.shape().to_vec();
    Ok(ComplexVectorInput {
        data: tensor::tensor_values_f64(&tensor)
            .into_iter()
            .map(|re| Complex::new(re, 0.0))
            .collect(),
        shape,
        is_complex: false,
        gpu_handle: None,
    })
}

fn complex_tensor_to_complex_vector(
    builtin: &'static str,
    label: &str,
    tensor: ComplexTensor,
) -> BuiltinResult<ComplexVectorInput> {
    ensure_vector_shape(builtin, label, tensor.shape())?;
    let shape = tensor.shape().to_vec();
    Ok(ComplexVectorInput {
        data: tensor::complex_tensor_into_values_complex64(tensor),
        shape,
        is_complex: true,
        gpu_handle: None,
    })
}

pub(crate) fn ensure_vector_shape(
    builtin: &'static str,
    label: &str,
    shape: &[usize],
) -> BuiltinResult<()> {
    let non_singleton = shape.iter().copied().filter(|&d| d > 1).count();
    if non_singleton > 1 {
        return Err(signal_error(
            builtin,
            None,
            format!("{builtin}: {label} must be a vector"),
        ));
    }
    Ok(())
}

pub(crate) fn vector_is_row(shape: &[usize]) -> bool {
    shape.first().copied().unwrap_or(1) == 1
}

pub(crate) fn gpu_vector_len(
    builtin: &'static str,
    label: &str,
    handle: &GpuTensorHandle,
) -> BuiltinResult<usize> {
    ensure_vector_shape(builtin, label, handle.shape())?;
    if handle.shape().contains(&0) {
        return Ok(0);
    }
    Ok(handle.shape().iter().copied().max().unwrap_or(1))
}

pub(crate) fn gpu_matrix_shape(
    builtin: &'static str,
    label: &str,
    handle: &GpuTensorHandle,
) -> BuiltinResult<(usize, usize)> {
    if handle.shape().len() > 2 {
        return Err(signal_error(
            builtin,
            None,
            format!("{builtin}: {label} must be a vector or 2-D matrix"),
        ));
    }
    let rows = handle.shape().first().copied().unwrap_or(1);
    let cols = handle.shape().get(1).copied().unwrap_or(1);
    if rows == 0 || cols == 0 {
        return Err(signal_error(
            builtin,
            None,
            format!("{builtin}: {label} must be nonempty"),
        ));
    }
    if rows == 1 || cols == 1 {
        Ok((rows.max(cols), 1))
    } else {
        Ok((rows, cols))
    }
}

pub(crate) fn selected_frequency_len(nfft: usize, range: ProviderSpectralRange) -> usize {
    match range {
        ProviderSpectralRange::Onesided => nfft / 2 + 1,
        ProviderSpectralRange::Twosided | ProviderSpectralRange::Centered => nfft,
    }
}

pub(crate) fn centered_shift(nfft: usize) -> usize {
    if nfft.is_multiple_of(2) {
        nfft / 2 + 1
    } else {
        nfft.div_ceil(2)
    }
}

pub(crate) fn centered_frequency_offset(nfft: usize) -> isize {
    if nfft.is_multiple_of(2) {
        nfft as isize / 2 - 1
    } else {
        nfft as isize / 2
    }
}

pub(crate) fn complex_vector_to_value(
    data: Vec<Complex<f64>>,
    shape: Vec<usize>,
    force_complex: bool,
) -> BuiltinResult<Value> {
    let has_imag = force_complex || data.iter().any(|z| z.im.abs() > EPS);
    if has_imag {
        let tensor = ComplexTensor::new(data.into_iter().map(|z| (z.re, z.im)).collect(), shape)
            .map_err(|e| signal_error("signal", None, format!("signal: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    } else {
        let tensor = Tensor::new(data.into_iter().map(|z| z.re).collect(), shape)
            .map_err(|e| signal_error("signal", None, format!("signal: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

pub(crate) fn complex_vector_to_value_with_dtype(
    data: Vec<Complex<f64>>,
    shape: Vec<usize>,
    force_complex: bool,
    dtype: NumericDType,
) -> BuiltinResult<Value> {
    if dtype != NumericDType::F32 {
        return complex_vector_to_value(data, shape, force_complex);
    }
    let has_imag = force_complex || data.iter().any(|z| z.im.abs() > EPS);
    if has_imag {
        let tensor = ComplexTensor::from_f32(
            data.into_iter()
                .map(|z| (z.re as f32, z.im as f32))
                .collect(),
            shape,
        )
        .map_err(|e| signal_error("signal", None, format!("signal: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    } else {
        let tensor = Tensor::from_f32(data.into_iter().map(|z| z.re as f32).collect(), shape)
            .map_err(|e| signal_error("signal", None, format!("signal: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    }
}

pub(crate) fn real_vector_to_row_value(data: Vec<f64>) -> BuiltinResult<Value> {
    let len = data.len();
    let tensor = Tensor::new(data, vec![1, len])
        .map_err(|e| signal_error("signal", None, format!("signal: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

pub(crate) fn parse_scalar_f64(
    builtin: &'static str,
    label: &str,
    value: &Value,
) -> BuiltinResult<f64> {
    let scalar = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Bool(b) => f64::from(u8::from(*b)),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => tensor::tensor_value_f64(t, 0),
        _ => {
            return Err(signal_error(
                builtin,
                None,
                format!("{builtin}: {label} must be a numeric scalar"),
            ))
        }
    };
    if !scalar.is_finite() {
        return Err(signal_error(
            builtin,
            None,
            format!("{builtin}: {label} must be finite"),
        ));
    }
    Ok(scalar)
}

pub(crate) fn parse_nonnegative_integer(
    builtin: &'static str,
    label: &str,
    value: &Value,
) -> BuiltinResult<usize> {
    if let Some(value) = tensor::scalar_integer_value(value) {
        return value.try_to_usize().ok_or_else(|| {
            signal_error(
                builtin,
                None,
                format!("{builtin}: {label} exceeds maximum supported size"),
            )
        });
    }
    let scalar = parse_scalar_f64(builtin, label, value)?;
    let rounded = scalar.round();
    if rounded < 0.0 || (rounded - scalar).abs() > 1e-9 {
        return Err(signal_error(
            builtin,
            None,
            format!("{builtin}: {label} must be a nonnegative integer"),
        ));
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(signal_error(
            builtin,
            None,
            format!("{builtin}: {label} exceeds maximum supported size"),
        ));
    }
    Ok(rounded as usize)
}

pub(crate) fn keyword(value: &Value) -> Result<Option<String>, runmat_value::Utf16DecodeError> {
    string_keyword(value)
}

pub(crate) fn scalar_length_arg(value: Value) -> Result<usize, WindowArgError> {
    if let Some(value) = tensor::scalar_integer_value(&value) {
        return value.try_to_usize().ok_or(WindowArgError::InvalidLength);
    }
    let scalar = match value {
        Value::Num(n) => n,
        Value::Int(i) => i.to_f64(),
        Value::Bool(b) => usize::from(b) as f64,
        Value::LogicalArray(array) if array.data.len() == 1 => {
            usize::from(array.data[0] != 0) as f64
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(&t) => tensor::tensor_value_f64(&t, 0),
        _ => return Err(WindowArgError::InvalidLength),
    };
    if !scalar.is_finite() || scalar < 0.0 {
        return Err(WindowArgError::InvalidLength);
    }
    let rounded = scalar.round();
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(WindowArgError::InvalidLength);
    }
    Ok(rounded as usize)
}

pub(crate) fn is_scalar_logical_window_length(value: &Value) -> bool {
    matches!(value, Value::Bool(_))
        || matches!(value, Value::LogicalArray(array) if array.data.len() == 1)
}

#[cfg(test)]
pub(crate) fn parse_window_controls(
    rest: &[Value],
    allow_type_name: bool,
) -> Result<WindowControls, WindowArgError> {
    parse_window_controls_view(
        crate::call::FunctionArgumentView::from_slice(rest),
        allow_type_name,
    )
}

pub(crate) fn parse_window_controls_view(
    rest: crate::call::FunctionArgumentView<'_>,
    allow_type_name: bool,
) -> Result<WindowControls, WindowArgError> {
    if rest.len() > 2 {
        return Err(WindowArgError::TooManyOptions);
    }
    let keywords = rest
        .iter()
        .map(|arg| {
            window_keyword(arg)
                .map_err(WindowArgError::Text)?
                .ok_or(WindowArgError::InvalidOptionType)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut sampling = WindowSampling::Symmetric;
    let mut output_type = WindowOutputType::Double;
    for (index, keyword) in keywords.into_iter().enumerate() {
        match (index, WindowOption::parse(&keyword)) {
            (0, Some(WindowOption::Sampling(value))) => sampling = value,
            (0, Some(WindowOption::OutputType(value))) if allow_type_name && rest.len() == 1 => {
                output_type = value
            }
            (1, Some(WindowOption::OutputType(value))) if allow_type_name => output_type = value,
            _ => return Err(WindowArgError::UnknownOption(keyword)),
        }
    }
    Ok(WindowControls {
        sampling,
        output_type,
    })
}

pub(crate) fn window_options_from_length(
    len_value: Value,
    controls: WindowControls,
) -> Result<WindowOptions, WindowArgError> {
    let len = scalar_length_arg(len_value)?;
    let WindowControls {
        sampling,
        output_type,
    } = controls;
    if matches!(sampling, WindowSampling::Periodic) && len.checked_add(1).is_none() {
        return Err(WindowArgError::InvalidLength);
    }
    Ok(WindowOptions {
        len,
        sampling,
        output_type,
    })
}

#[cfg(test)]
pub(crate) fn parse_window_options(
    len_value: Value,
    rest: &[Value],
    allow_type_name: bool,
) -> Result<WindowOptions, WindowArgError> {
    window_options_from_length(len_value, parse_window_controls(rest, allow_type_name)?)
}

fn string_keyword(value: &Value) -> Result<Option<String>, runmat_value::Utf16DecodeError> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(s)) => Some(s.to_ascii_lowercase()),
        Value::CharArray(chars) => {
            Some(runmat_value::decode_utf16_code_units(chars.data())?.to_ascii_lowercase())
        }
        _ => None,
    })
}

fn window_keyword(value: &Value) -> Result<Option<String>, runmat_value::Utf16DecodeError> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(text)) => Some(text.to_ascii_lowercase()),
        Value::StringArray(values)
            if values.data().len() == 1 && !values.data()[0].is_missing() =>
        {
            Some(
                values.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .to_ascii_lowercase(),
            )
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            Some(runmat_value::decode_utf16_code_units(chars.data())?.to_ascii_lowercase())
        }
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuGatherRetry;
    use runmat_value::{ComplexTensor, IntegerComplexStorage, IntegerStorage};

    fn first_unrepresentable_usize_double() -> f64 {
        if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        }
    }

    #[test]
    fn complex_vector_to_value_treats_near_zero_imaginary_as_real() {
        let value = complex_vector_to_value(
            vec![Complex::new(1.0, EPS / 2.0), Complex::new(2.0, -EPS / 2.0)],
            vec![1, 2],
            false,
        )
        .expect("value");

        let Value::Tensor(tensor) = value else {
            panic!("expected real tensor");
        };
        assert_eq!(tensor.shape(), [1, 2]);
        assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0]);
    }

    #[test]
    fn signal_complex_vector_parser_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![3, -2, 5]), vec![1, 3]).unwrap();

        let input = tensor_to_complex_vector("test", "x", tensor).expect("vector");

        assert_eq!(
            input.data,
            vec![
                Complex::new(3.0, 0.0),
                Complex::new(-2.0, 0.0),
                Complex::new(5.0, 0.0)
            ]
        );
    }

    #[test]
    fn signal_complex_vector_parser_reads_typed_complex_integer_storage_exactly() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::I16(vec![3, -2]),
            IntegerStorage::I16(vec![4, -5]),
        )
        .unwrap();
        let tensor = ComplexTensor::new_integer(storage, vec![1, 2]).unwrap();

        let input = complex_tensor_to_complex_vector("test", "x", tensor).expect("vector");

        assert_eq!(
            input.data,
            vec![Complex::new(3.0, 4.0), Complex::new(-2.0, -5.0)]
        );
    }

    #[test]
    fn signal_scalar_parser_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).unwrap();

        let scalar = parse_scalar_f64("test", "fs", &Value::Tensor(tensor)).expect("scalar");

        assert_eq!(scalar, 7.0);
    }

    #[test]
    fn parse_nonnegative_integer_rejects_values_outside_usize_range() {
        let err = parse_nonnegative_integer("test", "n", &Value::Num((usize::MAX as f64) * 2.0))
            .expect_err("out-of-range integer should fail");

        assert!(err.message().contains("exceeds maximum supported size"));
    }

    #[test]
    fn parse_nonnegative_integer_rejects_unrepresentable_double_boundary() {
        let err = parse_nonnegative_integer(
            "test",
            "n",
            &Value::Num(first_unrepresentable_usize_double()),
        )
        .expect_err("unrepresentable integer should fail");

        assert!(err.message().contains("exceeds maximum supported size"));
    }

    #[test]
    fn scalar_length_arg_rejects_values_outside_usize_range() {
        let err = scalar_length_arg(Value::Num((usize::MAX as f64) * 2.0))
            .expect_err("out-of-range length should fail");

        assert!(matches!(err, WindowArgError::InvalidLength));
    }

    #[test]
    fn scalar_length_arg_rejects_unrepresentable_double_boundary() {
        let err = scalar_length_arg(Value::Num(first_unrepresentable_usize_double()))
            .expect_err("unrepresentable length should fail");

        assert!(matches!(err, WindowArgError::InvalidLength));
    }

    #[test]
    fn scalar_length_arg_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::U16(vec![4]), vec![1, 1]).unwrap();

        let len = scalar_length_arg(Value::Tensor(tensor)).expect("valid length");

        assert_eq!(len, 4);
    }

    #[test]
    fn scalar_length_arg_accepts_valid_scalar_length() {
        let len = scalar_length_arg(Value::Num(4.0)).expect("valid length");

        assert_eq!(len, 4);
    }

    #[test]
    fn periodic_window_rejects_length_increment_overflow() {
        if usize::BITS != 64 {
            return;
        }
        let error = parse_window_options(
            Value::Int(runmat_value::IntValue::U64(u64::MAX)),
            &[Value::from("periodic")],
            true,
        )
        .expect_err("periodic effective length must not overflow");

        assert!(matches!(error, WindowArgError::InvalidLength));
    }

    #[test]
    fn window_controls_decode_scalar_text_once_and_reject_invalid_roles() {
        use runmat_value::{CharArray, StringArray};
        let single =
            Value::StringArray(StringArray::new(vec!["single".into()], vec![1, 1]).unwrap());
        let periodic = Value::CharArray(CharArray::new_row("periodic"));
        let controls = parse_window_controls(&[periodic, single], true).unwrap();
        assert_eq!(controls.sampling, WindowSampling::Periodic);
        assert_eq!(controls.output_type, WindowOutputType::Single);
        let misplaced =
            parse_window_controls(&[Value::from("single"), Value::from("periodic")], true);
        assert!(matches!(misplaced, Err(WindowArgError::UnknownOption(_))));
        let multirow = Value::CharArray(
            CharArray::from_unicode_scalars("periodic".chars().collect(), 2, 4).unwrap(),
        );
        assert!(matches!(
            parse_window_controls(&[multirow], true),
            Err(WindowArgError::InvalidOptionType)
        ));
    }

    #[test]
    fn window_host_capacity_overflow_preserves_original_resource_cause() {
        use runmat_builtins::{
            BLACKMAN_ERROR_INTERNAL, HAMMING_ERROR_INTERNAL, HANN_ERROR_INTERNAL,
        };
        let options = WindowOptions {
            len: usize::MAX,
            sampling: WindowSampling::Symmetric,
            output_type: WindowOutputType::Double,
        };
        let WindowArgError::Allocation(cause) = futures::executor::block_on(window_tensor(
            options,
            |_, _| panic!("capacity overflow must precede coefficients"),
            &mut SignalCooperation::new(None, "hann"),
        ))
        .expect("capacity validation is a domain failure")
        .expect_err("impossible host vector capacity") else {
            panic!("expected original allocation error");
        };
        for (builtin, descriptor) in [
            ("blackman", &BLACKMAN_ERROR_INTERNAL),
            ("hamming", &HAMMING_ERROR_INTERNAL),
            ("hann", &HANN_ERROR_INTERNAL),
        ] {
            let synthetic = Vec::<f64>::new()
                .try_reserve_exact(usize::MAX)
                .expect_err("impossible host vector capacity");
            let error = window_allocation_error(builtin, descriptor, synthetic);
            assert_eq!(error.identifier(), descriptor.identifier);
            assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
            let source = error.source.as_ref().expect("typed allocation cause");
            assert!(source
                .as_ref()
                .downcast_ref::<std::collections::TryReserveError>()
                .is_some());
        }
        assert!(!cause.to_string().is_empty());
    }

    #[test]
    fn gpu_vector_len_reports_zero_for_empty_vector_shape() {
        let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 0],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("admit owned fixture")
            .handle;
        let len = gpu_vector_len("test", "x", &handle).expect("vector length");
        assert_eq!(len, 0);
    }
}
