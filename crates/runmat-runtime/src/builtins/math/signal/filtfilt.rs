//! MATLAB-compatible zero-phase digital filtering.

use num_complex::Complex;
use runmat_accelerate_api::{
    GpuTensorHandle, ProviderIirCoefficientSource, ProviderIirFilterOptions,
    ProviderPreparedIirCoefficients,
};
use runmat_builtins::{
    BuiltinErrorDescriptor, FILTFILT_ERROR_INTERNAL, FILTFILT_ERROR_INVALID_COEFFICIENTS,
    FILTFILT_ERROR_INVALID_SIGNAL, FILTFILT_ERROR_PASS_FAILED, FILTFILT_ERROR_PROVIDER_OUTPUT,
    FILTFILT_INTEGER_INPUT_EXTENSION, FILTFILT_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Value};

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
    SignalCooperation,
};
use crate::builtins::math::signal::resident_output::{self, OutputContext};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[path = "filtfilt/initial_state.rs"]
mod initial_state;
mod resident_zero_phase;
use initial_state::lfilter_zi;

const BUILTIN_NAME: &str = "filtfilt";

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
    let mut context = source.context.clone();
    context.builtin = Some(BUILTIN_NAME.to_owned());
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
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::filtfilt"
)]
async fn filtfilt_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    b: Value,
    a: Value,
    x: Value,
) -> BuiltinResult<Value> {
    evaluate_with_runtime(runtime.as_ref(), b, a, x).await
}

pub async fn evaluate(b: Value, a: Value, x: Value) -> BuiltinResult<Value> {
    evaluate_with_runtime(None, b, a, x).await
}

async fn evaluate_with_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    b: Value,
    a: Value,
    x: Value,
) -> BuiltinResult<Value> {
    let mut cooperation = SignalCooperation::new(runtime, BUILTIN_NAME);
    cooperation.completed(0).await?;
    ensure_filtfilt_extensions([&b, &a, &x])?;
    let admitted_geometry = preflight_iir_operands(IirForm::ZeroPhase, &b, &a, &x, None)
        .map_err(filtfilt_preflight_error)?;
    for (value, descriptor) in [
        (&b, &FILTFILT_ERROR_INVALID_COEFFICIENTS),
        (&a, &FILTFILT_ERROR_INVALID_COEFFICIENTS),
        (&x, &FILTFILT_ERROR_INVALID_SIGNAL),
    ] {
        if let Value::GpuTensor(handle) = value {
            resident_output::owner(handle, || filtfilt_error(descriptor))?;
        }
    }
    let output_dtype = if [&b, &a, &x].into_iter().any(value_is_single) {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    let resident_preflight = if let Value::GpuTensor(handle) = &x {
        resident_zero_phase::preflight_route(
            handle,
            admitted_geometry.coefficient_len,
            output_dtype,
        )?
    } else {
        false
    };
    let b = gather_wide_integer_for_exactness(b).await?;
    let a = gather_wide_integer_for_exactness(a).await?;
    let x = gather_wide_integer_for_exactness(x).await?;
    for value in [&b, &a, &x] {
        ensure_wide_integer_data_exact(value, BUILTIN_NAME)?;
    }
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
    let coeffs = NormalizedCoefficients::new(&b_input.data, &a_input.data)?;
    let borrowed = [b_input.gpu_handle.as_ref(), a_input.gpu_handle.as_ref()]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    if let Value::GpuTensor(input) = &x {
        if let Some(value) = resident_zero_phase::try_execute(
            &coeffs,
            input,
            output_dtype,
            &borrowed,
            runtime,
            &mut cooperation,
            resident_preflight,
        )
        .await?
        {
            return Ok(value);
        }
    }
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

    if let Some(value) =
        try_filtfilt_gpu(&coeffs, &x_input, output_dtype, &borrowed, &mut cooperation).await?
    {
        return Ok(value);
    }

    let filtered = filtfilt_host(&coeffs, &x_input.data, &mut cooperation).await?;
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
        if b.iter()
            .chain(a)
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(filtfilt_error_with_detail(
                &FILTFILT_ERROR_INVALID_COEFFICIENTS,
                "coefficient values must be finite",
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
        if b_norm
            .iter()
            .chain(&a_norm)
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(filtfilt_error_with_detail(
                &FILTFILT_ERROR_INVALID_COEFFICIENTS,
                "normalized coefficients exceed the floating-point range",
            ));
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

async fn filtfilt_host(
    coeffs: &NormalizedCoefficients,
    signal: &[Complex<f64>],
    cooperation: &mut SignalCooperation,
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

    let zi = lfilter_zi(coeffs, cooperation).await?;
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

fn prepared_state_shape(
    signal_shape: &[usize],
    axis: usize,
    order: usize,
) -> BuiltinResult<Vec<usize>> {
    runmat_accelerate_api::provider_iir_state_shape(signal_shape, axis, order).map_err(|source| {
        crate::runtime_error::with_provider_execution_source(
            filtfilt_error(&FILTFILT_ERROR_INTERNAL),
            source.into(),
        )
    })
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

async fn try_filtfilt_gpu(
    coeffs: &NormalizedCoefficients,
    input: &ComplexVectorInput,
    output_dtype: NumericDType,
    borrowed: &[&GpuTensorHandle],
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Option<Value>> {
    let Some(input_handle) = input.gpu_handle.as_ref() else {
        return Ok(None);
    };
    let provider = resident_output::owner(input_handle, || {
        filtfilt_error_with_detail(&FILTFILT_ERROR_INVALID_SIGNAL, "invalid resident signal")
    })?;
    let Some(input_precision) = runmat_accelerate_api::handle_precision(input_handle) else {
        return Ok(None);
    };
    if matches!(output_dtype, NumericDType::F32)
        != matches!(
            provider.precision(),
            runmat_accelerate_api::ProviderPrecision::F32
        )
        || input_precision != provider.precision()
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

    let zi = lfilter_zi(coeffs, cooperation).await?;
    if zi.iter().any(|z| z.im != 0.0) {
        return Ok(None);
    }
    let extended = odd_reflect(&input.data, nfact)?;
    let b_real = coeffs.b.iter().map(|z| z.re).collect::<Vec<_>>();
    let a_real = coeffs.a.iter().map(|z| z.re).collect::<Vec<_>>();
    let coefficient_len =
        u64::try_from(b_real.len()).map_err(|_| filtfilt_error(&FILTFILT_ERROR_INTERNAL))?;
    let extended_len =
        u64::try_from(extended.len()).map_err(|_| filtfilt_error(&FILTFILT_ERROR_INTERNAL))?;
    let extended_shape = if vector_is_row(&input.shape) {
        vec![1, extended_len]
    } else {
        vec![extended_len, 1]
    };
    let state_len =
        u64::try_from(coeffs.state_len).map_err(|_| filtfilt_error(&FILTFILT_ERROR_INTERNAL))?;
    let state_shape = prepared_state_shape(
        &input.shape,
        usize::from(vector_is_row(&input.shape)),
        coeffs.state_len,
    )?
    .into_iter()
    .map(|extent| u64::try_from(extent).map_err(|_| filtfilt_error(&FILTFILT_ERROR_INTERNAL)))
    .collect::<BuiltinResult<Vec<_>>>()?;
    let element_type = runmat_accelerate_api::ProviderElementType::from(provider.precision());
    let representation = |shape| runmat_accelerate_api::ProviderRepresentation {
        element_type,
        storage: runmat_accelerate_api::ProviderStorage::DenseReal,
        layout: runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous,
        shape,
        residency: runmat_accelerate_api::ProviderResidency::Device,
    };
    let coefficients_shape = vec![coefficient_len, 1];
    let query = runmat_accelerate_api::ProviderFeasibilityQuery {
        operation: runmat_accelerate_api::ProviderOperationIdentity::new(
            runmat_accelerate_api::PROVIDER_IIR_PREPARED_PASS_OPERATION,
        ),
        family: runmat_accelerate_api::ProviderOperationFamily::Library,
        inputs: vec![
            representation(coefficients_shape.clone()),
            representation(coefficients_shape),
            representation(extended_shape.clone()),
            representation(state_shape.clone()),
        ],
        outputs: vec![representation(extended_shape), representation(state_shape)],
        workload: runmat_accelerate_api::ProviderWorkload {
            iir: Some(runmat_accelerate_api::ProviderIirWorkload {
                route: runmat_accelerate_api::ProviderIirRoute::PreparedPass,
                dim: u32::from(vector_is_row(&input.shape)),
                state_len,
                reflection_len: 0,
            }),
            ..Default::default()
        },
    };
    if !provider.query_feasibility(&query).is_supported() {
        return Ok(None);
    }
    let pass_borrowed = std::iter::once(input_handle)
        .chain(borrowed.iter().copied())
        .collect::<Vec<_>>();
    let prepared = ProviderPreparedIirCoefficients::upload(
        provider,
        &b_real,
        &a_real,
        input_precision,
        &pass_borrowed,
    )
    .map_err(|source| {
        let mut error = filtfilt_error_with_detail(
            &FILTFILT_ERROR_PASS_FAILED,
            "coefficient preparation failed",
        );
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error.source = Some(Box::new(source));
        error
    })?;
    let first_pass = gpu_filter_pass(
        provider,
        &prepared,
        coeffs,
        &extended,
        &input.shape,
        &zi,
        &pass_borrowed,
    )
    .await
    .map_err(|source| provider_pass_error(FilterPass::Forward, source))?;
    let mut reversed = first_pass;
    reversed.reverse();
    let second_pass = gpu_filter_pass(
        provider,
        &prepared,
        coeffs,
        &reversed,
        &input.shape,
        &zi,
        &pass_borrowed,
    )
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
    let context = OutputContext {
        input: input_handle,
        provider,
        caller_inputs: borrowed,
    };
    let contract = resident_output::floating_contract(
        input_handle,
        input.shape.clone(),
        runmat_accelerate_api::GpuTensorStorage::Real,
    );
    let output = context.admit(handle, &contract, &[], || {
        filtfilt_error(&FILTFILT_ERROR_PROVIDER_OUTPUT)
    })?;
    Ok(Some(Value::GpuTensor(output)))
}

async fn gpu_filter_pass(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    prepared: &ProviderPreparedIirCoefficients,
    coeffs: &NormalizedCoefficients,
    signal: &[Complex<f64>],
    original_shape: &[usize],
    zi: &[Complex<f64>],
    borrowed: &[&GpuTensorHandle],
) -> BuiltinResult<Vec<Complex<f64>>> {
    let is_row = vector_is_row(original_shape);
    let shape = if is_row {
        vec![1, signal.len()]
    } else {
        vec![signal.len(), 1]
    };
    let dim = if is_row { 1 } else { 0 };
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
    let scaled_zi = scale_state(zi, signal[0])
        .into_iter()
        .map(|z| z.re)
        .collect::<Vec<_>>();
    let zi_shape = prepared_state_shape(&shape, dim, coeffs.state_len)?;
    let zi_handle = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &scaled_zi,
            shape: &zi_shape,
        })
        .map_err(|source| {
            let mut error = filtfilt_error_with_detail(
                &FILTFILT_ERROR_PASS_FAILED,
                "initial-state upload failed",
            );
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            crate::runtime_error::with_provider_execution_source(error, source.into())
        })?;
    let result = provider
        .iir_filter(
            ProviderIirCoefficientSource::Prepared(prepared),
            &signal_handle,
            ProviderIirFilterOptions {
                dim,
                zi: Some(zi_handle.clone()),
            },
        )
        .await
        .map_err(|source| {
            let mut error = filtfilt_error(&FILTFILT_ERROR_PASS_FAILED);
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            crate::runtime_error::with_provider_execution_source(error, source.into())
        })?;
    let runmat_accelerate_api::ProviderIirFilterResult {
        output,
        final_state,
    } = result;
    let protected = borrowed
        .iter()
        .copied()
        .chain([prepared.numerator(), prepared.denominator(), &zi_handle])
        .collect::<Vec<_>>();
    let context = OutputContext {
        input: &signal_handle,
        provider,
        caller_inputs: &protected,
    };
    let Some(final_state) = final_state else {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            filtfilt_error(&FILTFILT_ERROR_PROVIDER_OUTPUT),
        ));
    };
    let output_contract = resident_output::floating_contract(
        &signal_handle,
        shape,
        runmat_accelerate_api::GpuTensorStorage::Real,
    );
    let output = match context.admit(output, &output_contract, &[&final_state], || {
        filtfilt_error(&FILTFILT_ERROR_PROVIDER_OUTPUT)
    }) {
        Ok(output) => output,
        Err(error) => return Err(gpu_helpers::reject_gpu_output(final_state, error)),
    };
    let state_contract = resident_output::floating_contract(
        &signal_handle,
        zi_shape,
        runmat_accelerate_api::GpuTensorStorage::Real,
    );
    if let Err(error) = context.admit(final_state, &state_contract, &[&output], || {
        filtfilt_error(&FILTFILT_ERROR_PROVIDER_OUTPUT)
    }) {
        return Err(gpu_helpers::reject_gpu_output(output, error));
    }
    let tensor = gpu_helpers::gather_tensor_async(&output)
        .await
        .map_err(|source| {
            filtfilt_error_with_source(
                &FILTFILT_ERROR_PASS_FAILED,
                "provider result gather failed",
                source,
                true,
            )
        })?;
    Ok(tensor::tensor_into_values_f64(tensor)
        .into_iter()
        .map(|re| Complex::new(re, 0.0))
        .collect())
}

#[cfg(test)]
#[path = "filtfilt/tests/mod.rs"]
mod tests;
