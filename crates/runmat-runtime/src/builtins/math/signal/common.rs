use num_complex::Complex;
use runmat_accelerate_api::{GpuTensorHandle, ProviderSpectralRange};
use runmat_builtins::{ComplexTensor, NumericDType, Tensor, Value};

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const EPS: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WindowSampling {
    Symmetric,
    Periodic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WindowOutputType {
    Double,
    Single,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WindowOptions {
    pub len: usize,
    pub sampling: WindowSampling,
    pub output_type: WindowOutputType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WindowArgError {
    InvalidLength,
    InvalidOptionType,
    UnknownOption(String),
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
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|flow| {
                    let source = map_control_flow_with_builtin(flow, builtin);
                    signal_error(
                        builtin,
                        None,
                        format!("{builtin}: failed to gather {label}: {}", source.message()),
                    )
                })?;
            let mut input = tensor_to_complex_vector(builtin, label, tensor)?;
            input.gpu_handle = Some(handle);
            Ok(input)
        }
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
    ensure_vector_shape(builtin, label, &tensor.shape)?;
    Ok(ComplexVectorInput {
        data: tensor
            .data
            .into_iter()
            .map(|re| Complex::new(re, 0.0))
            .collect(),
        shape: tensor.shape,
        is_complex: false,
        gpu_handle: None,
    })
}

fn complex_tensor_to_complex_vector(
    builtin: &'static str,
    label: &str,
    tensor: ComplexTensor,
) -> BuiltinResult<ComplexVectorInput> {
    ensure_vector_shape(builtin, label, &tensor.shape)?;
    Ok(ComplexVectorInput {
        data: tensor
            .data
            .into_iter()
            .map(|(re, im)| Complex::new(re, im))
            .collect(),
        shape: tensor.shape,
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
    ensure_vector_shape(builtin, label, &handle.shape)?;
    if handle.shape.contains(&0) {
        return Ok(0);
    }
    Ok(handle.shape.iter().copied().max().unwrap_or(1))
}

pub(crate) fn gpu_matrix_shape(
    builtin: &'static str,
    label: &str,
    handle: &GpuTensorHandle,
) -> BuiltinResult<(usize, usize)> {
    if handle.shape.len() > 2 {
        return Err(signal_error(
            builtin,
            None,
            format!("{builtin}: {label} must be a vector or 2-D matrix"),
        ));
    }
    let rows = handle.shape.first().copied().unwrap_or(1);
    let cols = handle.shape.get(1).copied().unwrap_or(1);
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
        Value::Tensor(t) if t.data.len() == 1 => t.data[0],
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
    let scalar = parse_scalar_f64(builtin, label, value)?;
    let rounded = scalar.round();
    if rounded < 0.0 || (rounded - scalar).abs() > 1e-9 {
        return Err(signal_error(
            builtin,
            None,
            format!("{builtin}: {label} must be a nonnegative integer"),
        ));
    }
    if rounded > usize::MAX as f64 {
        return Err(signal_error(
            builtin,
            None,
            format!("{builtin}: {label} exceeds maximum supported size"),
        ));
    }
    Ok(rounded as usize)
}

pub(crate) fn keyword(value: &Value) -> Option<String> {
    string_keyword(value)
}

pub(crate) fn scalar_length_arg(value: Value) -> Result<usize, WindowArgError> {
    let scalar = match value {
        Value::Num(n) => n,
        Value::Int(i) => i.to_f64(),
        Value::Bool(b) => usize::from(b) as f64,
        Value::Tensor(t) if t.data.len() == 1 => t.data[0],
        _ => return Err(WindowArgError::InvalidLength),
    };
    if !scalar.is_finite() || scalar < 0.0 {
        return Err(WindowArgError::InvalidLength);
    }
    let rounded = scalar.round();
    if rounded > usize::MAX as f64 {
        return Err(WindowArgError::InvalidLength);
    }
    Ok(rounded as usize)
}

pub(crate) fn window_tensor(
    options: WindowOptions,
    coeff: impl Fn(usize, usize) -> f64,
) -> Result<Value, WindowArgError> {
    let len = options.len;
    if len == 0 {
        return Tensor::new_with_dtype(Vec::new(), vec![0, 1], dtype_for(options.output_type))
            .map(Value::Tensor)
            .map_err(|e| WindowArgError::TensorBuild(e.to_string()));
    }
    if len == 1 {
        return Tensor::new_with_dtype(vec![1.0], vec![1, 1], dtype_for(options.output_type))
            .map(Value::Tensor)
            .map_err(|e| WindowArgError::TensorBuild(e.to_string()));
    }
    let effective_len = match options.sampling {
        WindowSampling::Symmetric => len,
        WindowSampling::Periodic => len + 1,
    };
    let mut data = (0..effective_len)
        .map(|idx| coeff(idx, effective_len))
        .collect::<Vec<_>>();
    if matches!(options.sampling, WindowSampling::Periodic) {
        data.pop();
    }
    Tensor::new_with_dtype(data, vec![len, 1], dtype_for(options.output_type))
        .map(Value::Tensor)
        .map_err(|e| WindowArgError::TensorBuild(e.to_string()))
}

pub(crate) fn parse_window_options(
    len_value: Value,
    rest: &[Value],
    allow_type_name: bool,
) -> Result<WindowOptions, WindowArgError> {
    let len = scalar_length_arg(len_value)?;
    let mut sampling = WindowSampling::Symmetric;
    let mut output_type = WindowOutputType::Double;
    for arg in rest {
        let Some(keyword) = string_keyword(arg) else {
            return Err(WindowArgError::InvalidOptionType);
        };
        match keyword.as_str() {
            "symmetric" => sampling = WindowSampling::Symmetric,
            "periodic" => sampling = WindowSampling::Periodic,
            "double" if allow_type_name => output_type = WindowOutputType::Double,
            "single" if allow_type_name => output_type = WindowOutputType::Single,
            _ => return Err(WindowArgError::UnknownOption(keyword)),
        }
    }
    Ok(WindowOptions {
        len,
        sampling,
        output_type,
    })
}

pub(crate) fn dtype_for(output_type: WindowOutputType) -> NumericDType {
    match output_type {
        WindowOutputType::Double => NumericDType::F64,
        WindowOutputType::Single => NumericDType::F32,
    }
}

pub(crate) fn provider_precision_matches(output_type: WindowOutputType) -> bool {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return false;
    };
    matches!(
        (provider.precision(), output_type),
        (
            runmat_accelerate_api::ProviderPrecision::F64,
            WindowOutputType::Double
        ) | (
            runmat_accelerate_api::ProviderPrecision::F32,
            WindowOutputType::Single
        )
    )
}

fn string_keyword(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_ascii_lowercase()),
        Value::CharArray(chars) => Some(chars.data.iter().collect::<String>().to_ascii_lowercase()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(tensor.shape, vec![1, 2]);
        assert_eq!(tensor.data, vec![1.0, 2.0]);
    }

    #[test]
    fn parse_nonnegative_integer_rejects_values_outside_usize_range() {
        let err = parse_nonnegative_integer("test", "n", &Value::Num((usize::MAX as f64) * 2.0))
            .expect_err("out-of-range integer should fail");

        assert!(err.message().contains("exceeds maximum supported size"));
    }

    #[test]
    fn scalar_length_arg_rejects_values_outside_usize_range() {
        let err = scalar_length_arg(Value::Num((usize::MAX as f64) * 2.0))
            .expect_err("out-of-range length should fail");

        assert_eq!(err, WindowArgError::InvalidLength);
    }

    #[test]
    fn scalar_length_arg_accepts_valid_scalar_length() {
        let len = scalar_length_arg(Value::Num(4.0)).expect("valid length");

        assert_eq!(len, 4);
    }

    #[test]
    fn gpu_vector_len_reports_zero_for_empty_vector_shape() {
        let handle = GpuTensorHandle {
            shape: vec![1, 0],
            device_id: 0,
            buffer_id: 0,
        };
        let len = gpu_vector_len("test", "x", &handle).expect("vector length");
        assert_eq!(len, 0);
    }
}
