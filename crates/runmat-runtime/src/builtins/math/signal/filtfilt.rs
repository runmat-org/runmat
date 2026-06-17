//! MATLAB-compatible zero-phase digital filtering.

use num_complex::Complex;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::signal::common::{
    complex_vector_to_value, signal_error, value_to_complex_vector, vector_is_row,
    ComplexVectorInput,
};
use crate::builtins::math::signal::filter;
use crate::builtins::math::signal::type_resolvers::filtfilt_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "filtfilt";
const EPS: f64 = 1.0e-12;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::filtfilt")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "filtfilt",
    op_kind: GpuOpKind::Custom("zero-phase-iir-filter"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("iir_filter")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses the provider `iir_filter` hook for the forward and reverse passes on real gpuArray inputs when available; complex inputs use the host reference path.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::filtfilt")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "filtfilt",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Zero-phase filtering is a two-pass stateful operation and is not fused.",
};

const FILTFILT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zero-phase filtered signal.",
}];

const FILTFILT_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numerator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Denominator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input signal vector.",
    },
];

const FILTFILT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "y = filtfilt(b, a, x)",
    inputs: &FILTFILT_INPUTS,
    outputs: &FILTFILT_OUTPUT,
}];

const FILTFILT_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTFILT.ARG_COUNT",
    identifier: Some("RunMat:filtfilt:ArgCount"),
    when: "The argument count is not exactly three.",
    message: "filtfilt: expected filtfilt(b, a, x)",
};

const FILTFILT_ERROR_INVALID_COEFFICIENTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTFILT.INVALID_COEFFICIENTS",
    identifier: Some("RunMat:filtfilt:InvalidCoefficients"),
    when: "Coefficient inputs are empty, non-vector, or have a zero leading denominator.",
    message: "filtfilt: invalid coefficient input",
};

const FILTFILT_ERROR_INVALID_SIGNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTFILT.INVALID_SIGNAL",
    identifier: Some("RunMat:filtfilt:InvalidSignal"),
    when: "The signal is not a numeric/logical vector or is too short for edge reflection.",
    message: "filtfilt: invalid signal input",
};

const FILTFILT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILTFILT.INTERNAL",
    identifier: Some("RunMat:filtfilt:Internal"),
    when: "Internal state solving, filtering, or tensor construction fails.",
    message: "filtfilt: internal error",
};

const FILTFILT_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FILTFILT_ERROR_ARG_COUNT,
    FILTFILT_ERROR_INVALID_COEFFICIENTS,
    FILTFILT_ERROR_INVALID_SIGNAL,
    FILTFILT_ERROR_INTERNAL,
];

pub const FILTFILT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FILTFILT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FILTFILT_ERRORS,
};

fn filtfilt_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    filtfilt_error_with_message(error.message, error)
}

fn filtfilt_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    filtfilt_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn filtfilt_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "filtfilt",
    category = "math/signal",
    summary = "Apply zero-phase forward-backward digital filtering.",
    keywords = "filtfilt,zero phase,filter,IIR,FIR,signal processing,gpu",
    accel = "custom",
    type_resolver(filtfilt_type),
    descriptor(crate::builtins::math::signal::filtfilt::FILTFILT_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::filtfilt"
)]
async fn filtfilt_builtin(b: Value, a: Value, x: Value) -> BuiltinResult<Value> {
    evaluate(b, a, x).await
}

pub async fn evaluate(b: Value, a: Value, x: Value) -> BuiltinResult<Value> {
    let b_input = value_to_complex_vector(BUILTIN_NAME, "numerator", b)
        .await
        .map_err(|err| {
            filtfilt_error_with_detail(&FILTFILT_ERROR_INVALID_COEFFICIENTS, err.message())
        })?;
    let a_input = value_to_complex_vector(BUILTIN_NAME, "denominator", a)
        .await
        .map_err(|err| {
            filtfilt_error_with_detail(&FILTFILT_ERROR_INVALID_COEFFICIENTS, err.message())
        })?;
    let x_input = value_to_complex_vector(BUILTIN_NAME, "signal", x)
        .await
        .map_err(|err| filtfilt_error_with_detail(&FILTFILT_ERROR_INVALID_SIGNAL, err.message()))?;

    let coeffs = NormalizedCoefficients::new(&b_input.data, &a_input.data)?;
    if x_input.data.is_empty() {
        return Err(filtfilt_error_with_detail(
            &FILTFILT_ERROR_INVALID_SIGNAL,
            "signal cannot be empty",
        ));
    }

    if let Some(value) = try_filtfilt_gpu(&coeffs, &x_input).await? {
        return Ok(value);
    }

    let filtered = filtfilt_host(&coeffs, &x_input.data)?;
    complex_vector_to_value(
        filtered,
        x_input.shape,
        b_input.is_complex || a_input.is_complex || x_input.is_complex,
    )
    .map_err(|err| filtfilt_error_with_detail(&FILTFILT_ERROR_INTERNAL, err.message()))
}

#[derive(Clone)]
struct NormalizedCoefficients {
    b: Vec<Complex<f64>>,
    a: Vec<Complex<f64>>,
    order: usize,
    state_len: usize,
}

impl NormalizedCoefficients {
    fn new(b: &[Complex<f64>], a: &[Complex<f64>]) -> BuiltinResult<Self> {
        if b.is_empty() || a.is_empty() {
            return Err(filtfilt_error_with_detail(
                &FILTFILT_ERROR_INVALID_COEFFICIENTS,
                "coefficient vectors cannot be empty",
            ));
        }
        let a0 = a[0];
        if a0.norm() <= EPS {
            return Err(filtfilt_error_with_detail(
                &FILTFILT_ERROR_INVALID_COEFFICIENTS,
                "denominator coefficient a(1) must be non-zero",
            ));
        }
        let order = b.len().max(a.len());
        let mut b_norm = vec![Complex::new(0.0, 0.0); order];
        let mut a_norm = vec![Complex::new(0.0, 0.0); order];
        for (idx, coeff) in b.iter().enumerate() {
            b_norm[idx] = *coeff / a0;
        }
        a_norm[0] = Complex::new(1.0, 0.0);
        for (idx, coeff) in a.iter().enumerate().skip(1) {
            a_norm[idx] = *coeff / a0;
        }
        Ok(Self {
            b: b_norm,
            a: a_norm,
            order,
            state_len: order.saturating_sub(1),
        })
    }

    fn is_real(&self) -> bool {
        self.b
            .iter()
            .chain(self.a.iter())
            .all(|z| z.im.abs() <= EPS)
    }
}

fn filtfilt_host(
    coeffs: &NormalizedCoefficients,
    signal: &[Complex<f64>],
) -> BuiltinResult<Vec<Complex<f64>>> {
    if coeffs.state_len == 0 {
        let gain = coeffs.b[0];
        return Ok(signal.iter().map(|x| gain * gain * *x).collect());
    }

    let nfact = 3usize
        .checked_mul(coeffs.state_len)
        .ok_or_else(|| filtfilt_error(&FILTFILT_ERROR_INTERNAL))?;
    if signal.len() <= nfact {
        return Err(filtfilt_error_with_detail(
            &FILTFILT_ERROR_INVALID_SIGNAL,
            format!(
                "signal length must be greater than {}, got {}",
                nfact,
                signal.len()
            ),
        ));
    }

    let zi = lfilter_zi(coeffs)?;
    let extended = odd_reflect(signal, nfact);
    let first = extended[0];
    let y = direct_form_filter(coeffs, &extended, &scale_state(&zi, first));
    let mut reversed = y;
    reversed.reverse();
    let second_first = reversed[0];
    let y2 = direct_form_filter(coeffs, &reversed, &scale_state(&zi, second_first));
    let mut restored = y2;
    restored.reverse();
    Ok(restored[nfact..nfact + signal.len()].to_vec())
}

fn odd_reflect(signal: &[Complex<f64>], nfact: usize) -> Vec<Complex<f64>> {
    if nfact == 0 {
        return signal.to_vec();
    }
    let mut out = Vec::with_capacity(signal.len() + 2 * nfact);
    let left_edge = signal[0] * Complex::new(2.0, 0.0);
    for idx in (1..=nfact).rev() {
        out.push(left_edge - signal[idx]);
    }
    out.extend_from_slice(signal);
    let right_edge = signal[signal.len() - 1] * Complex::new(2.0, 0.0);
    for offset in 1..=nfact {
        out.push(right_edge - signal[signal.len() - 1 - offset]);
    }
    out
}

fn scale_state(zi: &[Complex<f64>], scale: Complex<f64>) -> Vec<Complex<f64>> {
    zi.iter().map(|z| *z * scale).collect()
}

fn direct_form_filter(
    coeffs: &NormalizedCoefficients,
    signal: &[Complex<f64>],
    initial: &[Complex<f64>],
) -> Vec<Complex<f64>> {
    if coeffs.state_len == 0 {
        return signal.iter().map(|x| coeffs.b[0] * *x).collect();
    }
    let mut state = initial.to_vec();
    let mut output = Vec::with_capacity(signal.len());
    for &x_n in signal {
        let y = coeffs.b[0] * x_n + state[0];
        output.push(y);
        for idx in 1..coeffs.order {
            let next_state = if idx < coeffs.state_len {
                state[idx]
            } else {
                Complex::new(0.0, 0.0)
            };
            state[idx - 1] = coeffs.b[idx] * x_n + next_state - coeffs.a[idx] * y;
        }
    }
    output
}

fn lfilter_zi(coeffs: &NormalizedCoefficients) -> BuiltinResult<Vec<Complex<f64>>> {
    let n = coeffs.state_len;
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut matrix = vec![vec![Complex::new(0.0, 0.0); n]; n];
    for (row, row_values) in matrix.iter_mut().enumerate() {
        for (col, value) in row_values.iter_mut().enumerate() {
            if row == col {
                *value += Complex::new(1.0, 0.0);
            }
            let a_t = if col == 0 {
                -coeffs.a[row + 1]
            } else if row + 1 == col {
                Complex::new(1.0, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
            *value -= a_t;
        }
    }
    let rhs = (0..n)
        .map(|idx| coeffs.b[idx + 1] - coeffs.a[idx + 1] * coeffs.b[0])
        .collect::<Vec<_>>();
    solve_linear(matrix, rhs)
}

fn solve_linear(
    mut matrix: Vec<Vec<Complex<f64>>>,
    mut rhs: Vec<Complex<f64>>,
) -> BuiltinResult<Vec<Complex<f64>>> {
    let n = rhs.len();
    for pivot_idx in 0..n {
        let mut pivot_row = pivot_idx;
        let mut pivot_norm = matrix[pivot_idx][pivot_idx].norm();
        for (row, values) in matrix.iter().enumerate().skip(pivot_idx + 1) {
            let norm = values[pivot_idx].norm();
            if norm > pivot_norm {
                pivot_norm = norm;
                pivot_row = row;
            }
        }
        if pivot_norm <= EPS {
            return Err(filtfilt_error_with_detail(
                &FILTFILT_ERROR_INVALID_COEFFICIENTS,
                "filter initial-state system is singular",
            ));
        }
        if pivot_row != pivot_idx {
            matrix.swap(pivot_row, pivot_idx);
            rhs.swap(pivot_row, pivot_idx);
        }
        let pivot = matrix[pivot_idx][pivot_idx];
        let pivot_row_values = matrix[pivot_idx].clone();
        let pivot_rhs = rhs[pivot_idx];
        for row in pivot_idx + 1..n {
            let factor = matrix[row][pivot_idx] / pivot;
            for (col, pivot_value) in pivot_row_values.iter().enumerate().skip(pivot_idx) {
                matrix[row][col] -= factor * *pivot_value;
            }
            rhs[row] -= factor * pivot_rhs;
        }
    }

    let mut out = vec![Complex::new(0.0, 0.0); n];
    for row in (0..n).rev() {
        let mut acc = rhs[row];
        for (col, value) in matrix[row].iter().enumerate().skip(row + 1) {
            acc -= *value * out[col];
        }
        out[row] = acc / matrix[row][row];
    }
    Ok(out)
}

async fn try_filtfilt_gpu(
    coeffs: &NormalizedCoefficients,
    input: &ComplexVectorInput,
) -> BuiltinResult<Option<Value>> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        return Ok(None);
    };
    if input.gpu_handle.is_none() || input.is_complex || !coeffs.is_real() || coeffs.state_len == 0
    {
        return Ok(None);
    }
    let nfact = 3usize
        .checked_mul(coeffs.state_len)
        .ok_or_else(|| filtfilt_error(&FILTFILT_ERROR_INTERNAL))?;
    if input.data.len() <= nfact {
        return Ok(None);
    }

    let zi = lfilter_zi(coeffs)?;
    if zi.iter().any(|z| z.im.abs() > EPS) {
        return Ok(None);
    }
    let extended = odd_reflect(&input.data, nfact);
    let first_pass = match gpu_filter_pass(coeffs, &extended, &input.shape, &zi).await {
        Ok(value) => value,
        Err(err) => {
            log::debug!("filtfilt: first GPU pass failed ({err:?}), falling back to host");
            return Ok(None);
        }
    };
    let mut reversed = first_pass;
    reversed.reverse();
    let second_pass = match gpu_filter_pass(coeffs, &reversed, &input.shape, &zi).await {
        Ok(value) => value,
        Err(err) => {
            log::debug!("filtfilt: second GPU pass failed ({err:?}), falling back to host");
            return Ok(None);
        }
    };
    let mut restored = second_pass;
    restored.reverse();
    let trimmed = restored[nfact..nfact + input.data.len()]
        .iter()
        .map(|z| z.re)
        .collect::<Vec<_>>();
    let view = runmat_accelerate_api::HostTensorView {
        data: &trimmed,
        shape: &input.shape,
    };
    let handle = provider.upload(&view).map_err(|e| {
        filtfilt_error_with_detail(
            &FILTFILT_ERROR_INTERNAL,
            format!("provider upload failed: {e}"),
        )
    })?;
    Ok(Some(Value::GpuTensor(handle)))
}

struct OwnedGpuHandle<'a> {
    provider: &'a dyn runmat_accelerate_api::AccelProvider,
    handle: Option<runmat_accelerate_api::GpuTensorHandle>,
}

impl<'a> OwnedGpuHandle<'a> {
    fn new(
        provider: &'a dyn runmat_accelerate_api::AccelProvider,
        handle: runmat_accelerate_api::GpuTensorHandle,
    ) -> Self {
        Self {
            provider,
            handle: Some(handle),
        }
    }

    fn handle(&self) -> &runmat_accelerate_api::GpuTensorHandle {
        self.handle.as_ref().expect("owned GPU handle")
    }
}

impl Drop for OwnedGpuHandle<'_> {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = self.provider.free(&handle);
        }
    }
}

async fn gpu_filter_pass(
    coeffs: &NormalizedCoefficients,
    signal: &[Complex<f64>],
    original_shape: &[usize],
    zi: &[Complex<f64>],
) -> BuiltinResult<Vec<Complex<f64>>> {
    let provider = runmat_accelerate_api::provider().ok_or_else(|| {
        filtfilt_error_with_detail(&FILTFILT_ERROR_INTERNAL, "no acceleration provider")
    })?;
    let is_row = vector_is_row(original_shape);
    let shape = if is_row {
        vec![1, signal.len()]
    } else {
        vec![signal.len(), 1]
    };
    let dim = if is_row { 2.0 } else { 1.0 };
    let real_signal = signal.iter().map(|z| z.re).collect::<Vec<_>>();
    let signal_handle = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &real_signal,
            shape: &shape,
        })
        .map_err(|e| {
            filtfilt_error_with_detail(
                &FILTFILT_ERROR_INTERNAL,
                format!("provider signal upload failed: {e}"),
            )
        })?;
    let signal_handle = OwnedGpuHandle::new(provider, signal_handle);
    let b = Tensor::new(
        coeffs.b.iter().map(|z| z.re).collect(),
        vec![1, coeffs.b.len()],
    )
    .map_err(|e| filtfilt_error_with_detail(&FILTFILT_ERROR_INTERNAL, e))?;
    let a = Tensor::new(
        coeffs.a.iter().map(|z| z.re).collect(),
        vec![1, coeffs.a.len()],
    )
    .map_err(|e| filtfilt_error_with_detail(&FILTFILT_ERROR_INTERNAL, e))?;
    let scaled_zi = scale_state(zi, signal[0])
        .into_iter()
        .map(|z| z.re)
        .collect::<Vec<_>>();
    let zi_shape = if is_row {
        vec![1, coeffs.state_len]
    } else {
        vec![coeffs.state_len, 1]
    };
    let zi_value = Tensor::new(scaled_zi, zi_shape)
        .map(Value::Tensor)
        .map_err(|e| filtfilt_error_with_detail(&FILTFILT_ERROR_INTERNAL, e))?;
    let eval = filter::evaluate(
        Value::Tensor(b),
        Value::Tensor(a),
        Value::GpuTensor(signal_handle.handle().clone()),
        &[zi_value, Value::Num(dim)],
    )
    .await?;
    let (output, _) = eval.into_pair();
    let tensor = match output {
        Value::GpuTensor(handle) => {
            let output_handle = OwnedGpuHandle::new(provider, handle);
            gpu_helpers::gather_tensor_async(output_handle.handle())
                .await
                .map_err(|flow| {
                    signal_error(
                        BUILTIN_NAME,
                        FILTFILT_ERROR_INTERNAL.identifier,
                        format!("filtfilt: provider result gather failed: {flow:?}"),
                    )
                })?
        }
        other => tensor::value_to_tensor(&other)
            .map_err(|e| filtfilt_error_with_detail(&FILTFILT_ERROR_INTERNAL, e))?,
    };
    Ok(tensor
        .data
        .into_iter()
        .map(|re| Complex::new(re, 0.0))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{
        handle_precision, provider_for_handle, AccelDownloadFuture, AccelProvider, GpuTensorHandle,
        HostTensorView, ProviderPrecision,
    };
    use runmat_builtins::{builtin_function_by_name, ComplexTensor};
    use std::sync::atomic::{AtomicU64, Ordering};

    fn call(b: Value, a: Value, x: Value) -> BuiltinResult<Value> {
        block_on(evaluate(b, a, x))
    }

    fn tensor_data(value: Value) -> (Vec<f64>, Vec<usize>) {
        let tensor = test_support::gather(value).expect("gather tensor");
        (tensor.data, tensor.shape)
    }

    fn approx(lhs: &[f64], rhs: &[f64], tol: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            assert!((a - b).abs() <= tol, "mismatch at {idx}: got {a}, want {b}");
        }
    }

    fn value_tolerance(value: &Value) -> f64 {
        match value {
            Value::GpuTensor(handle) => match handle_precision(handle)
                .or_else(|| provider_for_handle(handle).map(|provider| provider.precision()))
                .unwrap_or(ProviderPrecision::F64)
            {
                ProviderPrecision::F64 => 1e-8,
                ProviderPrecision::F32 => 1e-6,
            },
            _ => 1e-8,
        }
    }

    struct DownloadFailProvider {
        next_buffer_id: AtomicU64,
        frees: AtomicU64,
    }

    impl DownloadFailProvider {
        const fn new() -> Self {
            Self {
                next_buffer_id: AtomicU64::new(1),
                frees: AtomicU64::new(0),
            }
        }

        fn reset(&self) {
            self.next_buffer_id.store(1, Ordering::SeqCst);
            self.frees.store(0, Ordering::SeqCst);
        }

        fn free_count(&self) -> u64 {
            self.frees.load(Ordering::SeqCst)
        }
    }

    impl AccelProvider for DownloadFailProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Ok(GpuTensorHandle {
                shape: host.shape.to_vec(),
                device_id: self.device_id(),
                buffer_id: self.next_buffer_id.fetch_add(1, Ordering::SeqCst),
            })
        }

        fn download<'a>(&'a self, _h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async { Err(anyhow::anyhow!("forced download failure")) })
        }

        fn free(&self, _h: &GpuTensorHandle) -> anyhow::Result<()> {
            self.frees.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn device_info(&self) -> String {
            "download-fail-provider".to_string()
        }

        fn device_id(&self) -> u32 {
            10_001
        }
    }

    static DOWNLOAD_FAIL_PROVIDER: DownloadFailProvider = DownloadFailProvider::new();

    #[test]
    fn descriptor_is_registered() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("filtfilt builtin");
        let descriptor = builtin.descriptor.expect("descriptor");
        assert_eq!(descriptor.signatures[0].label, "y = filtfilt(b, a, x)");
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.FILTFILT.INVALID_SIGNAL"));
    }

    #[test]
    fn fir_moving_average_preserves_row_shape() {
        let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let x = Tensor::new((1..=20).map(|v| v as f64).collect(), vec![1, 20]).unwrap();
        let (data, shape) =
            tensor_data(call(Value::Tensor(b), Value::Tensor(a), Value::Tensor(x)).unwrap());
        assert_eq!(shape, vec![1, 20]);
        approx(
            &data[3..17],
            &(4..=17).map(|v| v as f64).collect::<Vec<_>>(),
            1e-9,
        );
    }

    #[test]
    fn iir_coefficients_are_supported() {
        let b = Tensor::new(vec![0.2], vec![1, 1]).unwrap();
        let a = Tensor::new(vec![1.0, -0.8], vec![1, 2]).unwrap();
        let x = Tensor::new(vec![1.0; 16], vec![16, 1]).unwrap();
        let (data, shape) =
            tensor_data(call(Value::Tensor(b), Value::Tensor(a), Value::Tensor(x)).unwrap());
        assert_eq!(shape, vec![16, 1]);
        approx(&data, &[1.0; 16], 1e-8);
    }

    #[test]
    fn rejects_too_short_signal() {
        let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let err = call(Value::Tensor(b), Value::Tensor(a), Value::Tensor(x)).unwrap_err();
        assert!(err.message().contains("greater than"));
    }

    #[test]
    fn scalar_gain_filters_forward_and_backward() {
        let y = call(Value::Num(2.0), Value::Num(1.0), Value::Num(3.0)).unwrap();
        let (data, shape) = tensor_data(y);
        assert_eq!(shape, vec![1, 1]);
        approx(&data, &[12.0], 1e-12);
    }

    #[test]
    fn complex_host_path() {
        let b = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let x = ComplexTensor::new(vec![(1.0, 1.0), (2.0, -1.0)], vec![1, 2]).unwrap();
        let value = call(Value::Tensor(b), Value::Tensor(a), Value::ComplexTensor(x)).unwrap();
        let Value::ComplexTensor(out) = value else {
            panic!("expected complex output");
        };
        assert_eq!(out.shape, vec![1, 2]);
        assert_eq!(out.data, vec![(1.0, 1.0), (2.0, -1.0)]);
    }

    #[test]
    fn gpu_filter_pass_frees_signal_handle_when_filter_evaluation_fails() {
        let _guard = test_support::accel_test_lock();
        DOWNLOAD_FAIL_PROVIDER.reset();
        let _provider_guard =
            runmat_accelerate_api::ThreadProviderGuard::set(Some(&DOWNLOAD_FAIL_PROVIDER));

        let b = vec![Complex::new(0.5, 0.0), Complex::new(0.5, 0.0)];
        let a = vec![Complex::new(1.0, 0.0)];
        let coeffs = NormalizedCoefficients::new(&b, &a).expect("coefficients");
        let signal = (1..=8)
            .map(|value| Complex::new(value as f64, 0.0))
            .collect::<Vec<_>>();
        let zi = vec![Complex::new(0.0, 0.0)];

        let result = block_on(gpu_filter_pass(&coeffs, &signal, &[1, 8], &zi));

        assert!(result.is_err());
        assert_eq!(DOWNLOAD_FAIL_PROVIDER.free_count(), 1);
    }

    #[test]
    fn gpu_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
            let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let x = Tensor::new((1..=20).map(|v| v as f64).collect(), vec![1, 20]).unwrap();
            let cpu = call(
                Value::Tensor(b.clone()),
                Value::Tensor(a.clone()),
                Value::Tensor(x.clone()),
            )
            .unwrap();
            let (cpu_data, _) = tensor_data(cpu);
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &x.data,
                    shape: &x.shape,
                })
                .expect("upload");
            let gpu = call(Value::Tensor(b), Value::Tensor(a), Value::GpuTensor(handle)).unwrap();
            let tol = value_tolerance(&gpu);
            let (gpu_data, shape) = tensor_data(gpu);
            assert_eq!(shape, vec![1, 20]);
            approx(&gpu_data, &cpu_data, tol);
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn wgpu_matches_cpu() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let x = Tensor::new((1..=20).map(|v| v as f64).collect(), vec![1, 20]).unwrap();
        let cpu = call(
            Value::Tensor(b.clone()),
            Value::Tensor(a.clone()),
            Value::Tensor(x.clone()),
        )
        .unwrap();
        let (cpu_data, _) = tensor_data(cpu);
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &x.data,
                shape: &x.shape,
            })
            .expect("upload");
        let gpu = call(Value::Tensor(b), Value::Tensor(a), Value::GpuTensor(handle)).unwrap();
        let (gpu_data, _) = tensor_data(gpu);
        approx(&gpu_data, &cpu_data, 1e-6);
    }
}
