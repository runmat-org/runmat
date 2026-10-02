//! MATLAB-compatible zero-phase digital filtering.

use num_complex::Complex;
use runmat_builtins::{
    BuiltinErrorDescriptor, FILTFILT_ERROR_ARG_COUNT, FILTFILT_ERROR_INTERNAL,
    FILTFILT_ERROR_INVALID_COEFFICIENTS, FILTFILT_ERROR_INVALID_SIGNAL, FILTFILT_ERROR_PASS_FAILED,
    FILTFILT_INTEGER_INPUT_EXTENSION, FILTFILT_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Tensor, Value};

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::fft::common::ensure_wide_integer_data_exact;
use crate::builtins::math::signal::common::{
    complex_vector_to_value_with_dtype, preflight_iir_operands, value_to_complex_vector,
    vector_is_row, ComplexVectorInput, IirAdmissionFault, IirFaultKind, IirForm, IirOperand,
};
use crate::builtins::math::signal::filter;
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

fn filtfilt_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    filtfilt_error_with_message(error.message, error)
}

fn filtfilt_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    filtfilt_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn filtfilt_error_with_source(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: &'static str,
    source: RuntimeError,
    after_provider_effect: bool,
) -> RuntimeError {
    let span = source.span;
    let context = source.context.clone();
    let retry = source.gpu_gather_retry;
    let mut builder = build_runtime_error(format!("{}: {detail}", descriptor.message))
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    let mut error = builder.build();
    error.span = span;
    error.context = context;
    error.gpu_gather_retry = if after_provider_effect {
        crate::GpuGatherRetry::Never
    } else {
        retry
    };
    error
}

fn provider_pass_error(pass: FilterPass, source: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "{}: {} pass failed",
        FILTFILT_ERROR_PASS_FAILED.message,
        pass.label()
    ))
    .with_builtin(BUILTIN_NAME)
    .with_source(source);
    if let Some(identifier) = FILTFILT_ERROR_PASS_FAILED.identifier {
        builder = builder.with_identifier(identifier);
    }
    let mut error = builder.build();
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

#[derive(Clone, Copy)]
enum FilterPass {
    Forward,
    Reverse,
}

impl FilterPass {
    fn label(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Reverse => "reverse",
        }
    }
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
    binding_variant = "default",
    descriptor(runmat_builtins::FILTFILT_DESCRIPTOR),
    extensions(runmat_builtins::FILTFILT_EXTENSIONS),
    integer_capabilities(runmat_builtins::FILTFILT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::signal::filtfilt"
)]
async fn filtfilt_builtin(b: Value, a: Value, x: Value) -> BuiltinResult<Value> {
    evaluate(b, a, x).await
}

pub async fn evaluate(b: Value, a: Value, x: Value) -> BuiltinResult<Value> {
    ensure_filtfilt_extensions([&b, &a, &x])?;
    preflight_iir_operands(IirForm::ZeroPhase, &b, &a, &x, None)
        .map_err(filtfilt_preflight_error)?;
    let b = gather_wide_integer_for_exactness(b).await?;
    let a = gather_wide_integer_for_exactness(a).await?;
    let x = gather_wide_integer_for_exactness(x).await?;
    for value in [&b, &a, &x] {
        ensure_wide_integer_data_exact(value, BUILTIN_NAME)?;
    }
    let output_single = [&b, &a, &x].into_iter().any(value_is_single);
    crate::builtins::common::validation::reject_typed_complex_integer(&b, BUILTIN_NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&a, BUILTIN_NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&x, BUILTIN_NAME)?;
    let b_input = value_to_complex_vector(BUILTIN_NAME, "numerator", b)
        .await
        .map_err(|source| {
            filtfilt_error_with_source(
                &FILTFILT_ERROR_INVALID_COEFFICIENTS,
                "numerator conversion failed",
                source,
                false,
            )
        })?;
    let a_input = value_to_complex_vector(BUILTIN_NAME, "denominator", a)
        .await
        .map_err(|source| {
            filtfilt_error_with_source(
                &FILTFILT_ERROR_INVALID_COEFFICIENTS,
                "denominator conversion failed",
                source,
                false,
            )
        })?;
    let x_input = value_to_complex_vector(BUILTIN_NAME, "signal", x)
        .await
        .map_err(|source| {
            filtfilt_error_with_source(
                &FILTFILT_ERROR_INVALID_SIGNAL,
                "signal conversion failed",
                source,
                false,
            )
        })?;

    let coeffs = NormalizedCoefficients::new(&b_input.data, &a_input.data)?;
    if x_input.data.is_empty() {
        return Err(filtfilt_error_with_detail(
            &FILTFILT_ERROR_INVALID_SIGNAL,
            "signal cannot be empty",
        ));
    }
    if x_input
        .data
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(filtfilt_error_with_detail(
            &FILTFILT_ERROR_INVALID_SIGNAL,
            "signal values must be finite",
        ));
    }

    let output_dtype = if output_single {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    if let Some(value) = try_filtfilt_gpu(&coeffs, &x_input, output_dtype).await? {
        return Ok(value);
    }

    let filtered = filtfilt_host(&coeffs, &x_input.data)?;
    complex_vector_to_value_with_dtype(
        filtered,
        x_input.shape,
        b_input.is_complex || a_input.is_complex || x_input.is_complex,
        output_dtype,
    )
    .map_err(|source| {
        filtfilt_error_with_source(
            &FILTFILT_ERROR_INTERNAL,
            "result construction failed",
            source,
            false,
        )
    })
}

fn filtfilt_preflight_error(fault: IirAdmissionFault) -> RuntimeError {
    let descriptor = match fault.operand {
        IirOperand::Numerator | IirOperand::Denominator => &FILTFILT_ERROR_INVALID_COEFFICIENTS,
        IirOperand::Signal | IirOperand::InitialState => &FILTFILT_ERROR_INVALID_SIGNAL,
    };
    let detail = match fault.kind {
        IirFaultKind::Empty => "input vector cannot be empty",
        IirFaultKind::NonVector => "input must be a vector",
        IirFaultKind::LeadingDenominatorZero => "denominator coefficient a(1) must be non-zero",
        IirFaultKind::SignalTooShort => "signal length must be greater than reflection length",
        IirFaultKind::GeometryOverflow => "input geometry exceeds the supported range",
        IirFaultKind::UnsupportedClass => "input must be numeric or logical",
    };
    filtfilt_error_with_detail(descriptor, detail)
}

async fn gather_wide_integer_for_exactness(value: Value) -> BuiltinResult<Value> {
    if matches!(&value, Value::GpuTensor(handle) if matches!(
        runmat_accelerate_api::handle_integer_type(handle),
        Some(runmat_accelerate_api::IntegerElementType::I64 | runmat_accelerate_api::IntegerElementType::U64)
    )) {
        return gpu_helpers::gather_value_async(&value)
            .await
            .map_err(|source| {
                filtfilt_error_with_source(
                    &FILTFILT_ERROR_INVALID_SIGNAL,
                    "wide integer gather failed",
                    source,
                    true,
                )
            });
    }
    Ok(value)
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn value_is_single(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::ComplexTensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_none()
            && !runmat_accelerate_api::handle_is_logical(handle)
            && runmat_accelerate_api::handle_precision(handle) == Some(runmat_accelerate_api::ProviderPrecision::F32))
}

fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

fn ensure_filtfilt_extensions(values: [&Value; 3]) -> BuiltinResult<()> {
    if values.iter().copied().any(is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FILTFILT_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if values.iter().copied().any(is_logical_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FILTFILT_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
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
        if a0 == Complex::new(0.0, 0.0) {
            return Err(filtfilt_error_with_detail(
                &FILTFILT_ERROR_INVALID_COEFFICIENTS,
                "denominator coefficient a(1) must be non-zero",
            ));
        }
        let order = b.len().max(a.len());
        let mut b_norm = vec![Complex::new(0.0, 0.0); order];
        let mut a_norm = vec![Complex::new(0.0, 0.0); order];
        for (idx, coeff) in b.iter().enumerate() {
            b_norm[idx] = divide_by_leading_coefficient(*coeff, a0);
        }
        a_norm[0] = Complex::new(1.0, 0.0);
        for (idx, coeff) in a.iter().enumerate().skip(1) {
            a_norm[idx] = divide_by_leading_coefficient(*coeff, a0);
        }
        Ok(Self {
            b: b_norm,
            a: a_norm,
            order,
            state_len: order.saturating_sub(1),
        })
    }

    fn is_real(&self) -> bool {
        self.b.iter().chain(self.a.iter()).all(|z| z.im == 0.0)
    }
}

fn divide_by_leading_coefficient(value: Complex<f64>, leading: Complex<f64>) -> Complex<f64> {
    if leading.im == 0.0 {
        Complex::new(value.re / leading.re, value.im / leading.re)
    } else {
        value / leading
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
    let extended = odd_reflect(signal, nfact)?;
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

fn odd_reflect(signal: &[Complex<f64>], nfact: usize) -> BuiltinResult<Vec<Complex<f64>>> {
    if nfact == 0 {
        return Ok(signal.to_vec());
    }
    let total = nfact
        .checked_mul(2)
        .and_then(|padding| signal.len().checked_add(padding))
        .ok_or_else(|| {
            filtfilt_error_with_detail(&FILTFILT_ERROR_INTERNAL, "reflection length overflow")
        })?;
    let mut out = Vec::new();
    out.try_reserve_exact(total).map_err(|source| {
        let mut builder = build_runtime_error(format!(
            "{}: reflection allocation failed",
            FILTFILT_ERROR_INTERNAL.message
        ))
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
        if let Some(identifier) = FILTFILT_ERROR_INTERNAL.identifier {
            builder = builder.with_identifier(identifier);
        }
        let mut wrapped = builder.build();
        wrapped.gpu_gather_retry = crate::GpuGatherRetry::Never;
        wrapped
    })?;
    let left_edge = signal[0];
    for idx in (1..=nfact).rev() {
        out.push(reflected_sample(left_edge, signal[idx])?);
    }
    out.extend_from_slice(signal);
    let right_edge = signal[signal.len() - 1];
    for offset in 1..=nfact {
        out.push(reflected_sample(
            right_edge,
            signal[signal.len() - 1 - offset],
        )?);
    }
    Ok(out)
}

fn reflected_sample(edge: Complex<f64>, neighbor: Complex<f64>) -> BuiltinResult<Complex<f64>> {
    let real = edge.re + (edge.re - neighbor.re);
    let imaginary = edge.im + (edge.im - neighbor.im);
    if real.is_finite() && imaginary.is_finite() {
        Ok(Complex::new(real, imaginary))
    } else {
        Err(filtfilt_error_with_detail(
            &FILTFILT_ERROR_INVALID_SIGNAL,
            "endpoint reflection is outside the floating-point range",
        ))
    }
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
    output_dtype: NumericDType,
) -> BuiltinResult<Option<Value>> {
    let Some(input_handle) = input.gpu_handle.as_ref() else {
        return Ok(None);
    };
    let Some(provider) = runmat_accelerate_api::provider_for_handle(input_handle) else {
        return Ok(None);
    };
    if matches!(output_dtype, NumericDType::F32)
        != matches!(
            provider.precision(),
            runmat_accelerate_api::ProviderPrecision::F32
        )
    {
        return Ok(None);
    }
    if input.is_complex || !coeffs.is_real() || coeffs.state_len == 0 {
        return Ok(None);
    }
    let nfact = 3usize
        .checked_mul(coeffs.state_len)
        .ok_or_else(|| filtfilt_error(&FILTFILT_ERROR_INTERNAL))?;
    if input.data.len() <= nfact {
        return Ok(None);
    }

    let zi = lfilter_zi(coeffs)?;
    if zi.iter().any(|z| z.im != 0.0) {
        return Ok(None);
    }
    let extended = odd_reflect(&input.data, nfact)?;
    let first_pass = gpu_filter_pass(provider, coeffs, &extended, &input.shape, &zi)
        .await
        .map_err(|source| provider_pass_error(FilterPass::Forward, source))?;
    let mut reversed = first_pass;
    reversed.reverse();
    let second_pass = gpu_filter_pass(provider, coeffs, &reversed, &input.shape, &zi)
        .await
        .map_err(|source| provider_pass_error(FilterPass::Reverse, source))?;
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
    let handle = provider.upload(&view).map_err(|source| {
        crate::runtime_error::with_provider_execution_source(
            filtfilt_error_with_detail(&FILTFILT_ERROR_PASS_FAILED, "result upload failed"),
            source.into(),
        )
    })?;
    Ok(Some(Value::GpuTensor(handle)))
}

async fn gpu_filter_pass(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    coeffs: &NormalizedCoefficients,
    signal: &[Complex<f64>],
    original_shape: &[usize],
    zi: &[Complex<f64>],
) -> BuiltinResult<Vec<Complex<f64>>> {
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
        .map_err(|source| {
            crate::runtime_error::with_provider_execution_source(
                filtfilt_error_with_detail(&FILTFILT_ERROR_PASS_FAILED, "signal upload failed"),
                source.into(),
            )
        })?;
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
    let zi_shape = vec![coeffs.state_len, 1];
    let zi_value = Tensor::new(scaled_zi, zi_shape)
        .map(Value::Tensor)
        .map_err(|e| filtfilt_error_with_detail(&FILTFILT_ERROR_INTERNAL, e))?;
    let eval = filter::evaluate(
        Value::Tensor(b),
        Value::Tensor(a),
        Value::GpuTensor(signal_handle.clone()),
        &[zi_value, Value::Num(dim)],
    )
    .await?;
    let (output, _) = eval.into_pair();
    let tensor = match output {
        Value::GpuTensor(handle) => {
            gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|source| {
                    filtfilt_error_with_source(
                        &FILTFILT_ERROR_PASS_FAILED,
                        "provider result gather failed",
                        source,
                        true,
                    )
                })?
        }
        other => tensor::value_to_tensor(&other)
            .map_err(|e| filtfilt_error_with_detail(&FILTFILT_ERROR_INTERNAL, e))?,
    };
    Ok(tensor::tensor_into_values_f64(tensor)
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
        handle_precision, provider_for_handle, AccelDownloadFuture, AccelNumericDownloadFuture,
        AccelProvider, AccelProviderFuture, GpuTensorHandle, HostTensorView,
        ProviderIirFilterOptions, ProviderIirFilterResult, ProviderPrecision,
    };
    use runmat_builtins::builtin_function_by_name;
    use runmat_value::ComplexTensor;

    fn call(b: Value, a: Value, x: Value) -> BuiltinResult<Value> {
        block_on(evaluate(b, a, x))
    }

    fn tensor_data(value: Value) -> (Vec<f64>, Vec<usize>) {
        let tensor = test_support::gather(value).expect("gather tensor");
        (tensor.materialize_f64(), tensor.shape)
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
        owner: runmat_accelerate_api::test_support::TestAllocationOwner,
        probes: std::sync::Mutex<Vec<runmat_accelerate_api::test_support::RetirementProbe>>,
    }
    impl DownloadFailProvider {
        fn new() -> Self {
            Self {
                owner: Default::default(),
                probes: Default::default(),
            }
        }
        fn free_count(&self) -> usize {
            self.probes
                .lock()
                .unwrap()
                .iter()
                .map(|probe| probe.retirements())
                .sum()
        }
    }
    impl AccelProvider for DownloadFailProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            let allocation =
                self.owner
                    .allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
                        data: runmat_accelerate_api::HostNumericDataOwned::F64(host.data.to_vec()),
                        shape: host.shape.to_vec(),
                        storage: runmat_accelerate_api::GpuTensorStorage::Real,
                    })?;
            self.probes.lock().unwrap().push(allocation.probe);
            Ok(allocation.handle)
        }
        fn download<'a>(&'a self, _h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async { Err(anyhow::anyhow!("forced download failure")) })
        }
        fn device_info(&self) -> String {
            "download-fail-provider".to_string()
        }
        fn device_id(&self) -> u32 {
            self.owner.device_id()
        }
    }

    #[test]
    fn descriptor_is_registered() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("filtfilt builtin");
        let descriptor = builtin.descriptor.expect("descriptor");
        assert_eq!(descriptor.signatures[0].label, "y = filtfilt(b, a, x)");
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.FILTFILT.INVALID_SIGNAL"));
        assert_eq!(builtin.integer_capabilities.len(), 1);
        assert_eq!(builtin.extensions.len(), 2);
    }

    #[test]
    fn compatibility_mode_rejects_integer_and_logical_extensions() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer = call(
            Value::Int(runmat_value::IntValue::I16(1)),
            Value::Num(1.0),
            Value::Num(1.0),
        )
        .expect_err("integer extension must reject");
        assert_eq!(
            integer.identifier(),
            FILTFILT_INTEGER_INPUT_EXTENSION.error_identifier
        );
        let logical = call(Value::Bool(true), Value::Num(1.0), Value::Num(1.0))
            .expect_err("logical extension must reject");
        assert_eq!(
            logical.identifier(),
            FILTFILT_LOGICAL_INPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn nonzero_subnormal_denominator_is_not_classified_as_zero() {
        let signal = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let output = call(
            Value::Num(1.0e-300),
            Value::Num(1.0e-300),
            Value::Tensor(signal),
        )
        .expect("finite nonzero denominator");
        let (values, shape) = tensor_data(output);
        assert_eq!(shape, vec![1, 2]);
        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn tiny_nonzero_imaginary_coefficient_remains_complex() {
        let signal = Tensor::new((1..=8).map(f64::from).collect(), vec![1, 8]).unwrap();
        let numerator = ComplexTensor::new(vec![(1.0, 0.0), (0.0, 1.0e-300)], vec![1, 2]).unwrap();
        let output = call(
            Value::ComplexTensor(numerator),
            Value::Num(1.0),
            Value::Tensor(signal),
        )
        .expect("small imaginary coefficient");
        let Value::ComplexTensor(output) = output else {
            panic!("complex coefficient must produce complex output");
        };
        assert_eq!(output.shape, vec![1, 8]);
        assert!(output.materialize_f64().iter().any(|&(_, im)| im != 0.0));
    }

    #[test]
    fn finite_large_constant_reflection_does_not_overflow_before_identity_filter() {
        let signal = Tensor::new(vec![1.0e308; 8], vec![1, 8]).unwrap();
        let identity_with_state = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let output = call(
            Value::Tensor(identity_with_state),
            Value::Num(1.0),
            Value::Tensor(signal),
        )
        .expect("representable large constant reflection");
        let (values, shape) = tensor_data(output);
        assert_eq!(shape, vec![1, 8]);
        assert_eq!(values, vec![1.0e308; 8]);

        let signal = Tensor::new(
            vec![f64::MAX, -f64::MAX, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![1, 8],
        )
        .unwrap();
        let b = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let error = call(Value::Tensor(b), Value::Num(1.0), Value::Tensor(signal))
            .expect_err("unrepresentable endpoint reflection");
        assert_eq!(error.identifier(), FILTFILT_ERROR_INVALID_SIGNAL.identifier);
    }

    #[test]
    fn resident_exact_wide_integer_uses_authoritative_gather_fallback() {
        test_support::with_test_provider(|provider| {
            let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
            let signal = Tensor::new_integer(
                runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_994; 4]),
                vec![1, 4],
            )
            .expect("wide signal");
            let handle = gpu_helpers::upload_tensor(provider, &signal).expect("wide upload");
            let result = block_on(evaluate(
                Value::Num(1.0),
                Value::Num(1.0),
                Value::GpuTensor(handle),
            ))
            .expect("exact resident wide integer");
            let gathered = test_support::gather(result).expect("gather result");
            assert_eq!(gathered.numeric_dtype(), NumericDType::F64);
            assert_eq!(gathered.shape, vec![1, 4]);
        });
    }

    #[test]
    fn single_output_is_native_and_nonfinite_signal_rejects() {
        let single = Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let value = call(Value::Num(1.0), Value::Num(1.0), Value::Tensor(single)).unwrap();
        let Value::Tensor(value) = value else {
            panic!("expected tensor")
        };
        assert_eq!(value.numeric_dtype(), NumericDType::F32);

        let nonfinite = Tensor::new(vec![1.0, f64::NAN], vec![1, 2]).unwrap();
        call(Value::Num(1.0), Value::Num(1.0), Value::Tensor(nonfinite))
            .expect_err("nonfinite signal must reject");
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
        assert_eq!(out.materialize_f64(), vec![(1.0, 1.0), (2.0, -1.0)]);
    }

    #[test]
    fn gpu_filter_pass_frees_signal_handle_when_filter_evaluation_fails() {
        let _guard = test_support::accel_test_lock();
        let provider = Box::leak(Box::new(DownloadFailProvider::new()));
        let _provider_guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));

        let b = vec![Complex::new(0.5, 0.0), Complex::new(0.5, 0.0)];
        let a = vec![Complex::new(1.0, 0.0)];
        let coeffs = NormalizedCoefficients::new(&b, &a).expect("coefficients");
        let signal = (1..=8)
            .map(|value| Complex::new(value as f64, 0.0))
            .collect::<Vec<_>>();
        let zi = vec![Complex::new(0.0, 0.0)];

        let result = block_on(gpu_filter_pass(provider, &coeffs, &signal, &[1, 8], &zi));

        assert!(result.is_err());
        assert_eq!(provider.free_count(), 1);
    }

    struct OperationalFailureProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider,
        calls: std::sync::atomic::AtomicUsize,
        downloads: std::sync::atomic::AtomicUsize,
    }

    impl AccelProvider for OperationalFailureProvider {
        fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            self.inner.upload(host)
        }
        fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            self.downloads
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.download(handle)
        }
        fn download_numeric<'a>(
            &'a self,
            handle: &'a GpuTensorHandle,
        ) -> AccelNumericDownloadFuture<'a> {
            self.downloads
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.download_numeric(handle)
        }
        fn device_info(&self) -> String {
            "filtfilt-operational-failure".into()
        }
        fn device_id(&self) -> u32 {
            self.inner.device_id()
        }
        fn iir_filter<'a>(
            &'a self,
            _b: &'a GpuTensorHandle,
            _a: &'a GpuTensorHandle,
            _x: &'a GpuTensorHandle,
            _options: ProviderIirFilterOptions,
        ) -> AccelProviderFuture<'a, ProviderIirFilterResult> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async { Err(anyhow::anyhow!("device failed during filter pass")) })
        }
    }

    #[test]
    fn resident_pass_operational_failure_keeps_cause_and_does_not_replay() {
        let _lock = test_support::accel_test_lock();
        let provider: &'static OperationalFailureProvider =
            Box::leak(Box::new(OperationalFailureProvider {
                inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
                calls: std::sync::atomic::AtomicUsize::new(0),
                downloads: std::sync::atomic::AtomicUsize::new(0),
            }));
        let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let x = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                shape: &[1, 8],
            })
            .unwrap();
        let b = Tensor::new(vec![0.5, 0.5], vec![1, 2]).unwrap();
        let error = call(Value::Tensor(b), Value::Num(1.0), Value::GpuTensor(x))
            .expect_err("operational pass failure must terminate");
        assert_eq!(error.identifier(), FILTFILT_ERROR_PASS_FAILED.identifier);
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(std::error::Error::source(&error).is_some());
        assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn host_invalid_denominator_precedes_resident_numerator_readback() {
        let _lock = test_support::accel_test_lock();
        let provider: &'static OperationalFailureProvider =
            Box::leak(Box::new(OperationalFailureProvider {
                inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
                calls: std::sync::atomic::AtomicUsize::new(0),
                downloads: std::sync::atomic::AtomicUsize::new(0),
            }));
        let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let b = provider
            .upload(&HostTensorView {
                data: &[0.5, 0.5],
                shape: &[1, 2],
            })
            .expect("resident coefficients");
        let empty_a = Tensor::new(Vec::new(), vec![0, 1]).expect("empty denominator");
        let error = call(
            Value::GpuTensor(b.clone()),
            Value::Tensor(empty_a),
            Value::Num(1.0),
        )
        .expect_err("known-invalid denominator");
        assert_eq!(
            error.identifier(),
            FILTFILT_ERROR_INVALID_COEFFICIENTS.identifier
        );
        assert_eq!(
            provider.downloads.load(std::sync::atomic::Ordering::SeqCst),
            0
        );
        assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        b.release().expect("release original resident coefficients");
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
                    data: &x.materialize_f64(),
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
                data: &x.materialize_f64(),
                shape: &x.shape,
            })
            .expect("upload");
        let gpu = call(Value::Tensor(b), Value::Tensor(a), Value::GpuTensor(handle)).unwrap();
        let (gpu_data, _) = tensor_data(gpu);
        approx(&gpu_data, &cpu_data, 1e-6);
    }
}
