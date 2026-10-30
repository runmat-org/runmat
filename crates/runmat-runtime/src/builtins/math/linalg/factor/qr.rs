//! MATLAB-compatible `qr` builtin with pivoted and economy forms.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg(test)]
use crate::builtins::math::linalg::type_resolvers::{matrix_dims, numeric_tensor_from_shape};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use num_complex::Complex64;
use runmat_accelerate_api::{
    GpuTensorHandle, ProviderQrColumnOrder, ProviderQrOutputs, ProviderQrPivot, ProviderQrResult,
};
#[cfg(test)]
use runmat_builtins::shape_rules::{element_count_if_known, unknown_shape};
use runmat_builtins::{BuiltinErrorDescriptor, FactorOption};
#[cfg(test)]
use runmat_builtins::{ResolveContext, Type};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, NumericDType, Tensor, Value};

use super::lu::PivotMode;
mod request;
pub use request::QrOutputRequest;
use request::{QrOptions, QrPermutationFormat};
mod provider_result;

const BUILTIN_NAME: &str = "qr";

#[cfg(test)]
use runmat_builtins::QR_DESCRIPTOR;
use runmat_builtins::{
    QR_ERROR_INTERNAL, QR_ERROR_INVALID_ARGUMENT, QR_ERROR_INVALID_INPUT,
    QR_INTEGER_INPUT_EXTENSION, QR_INTEGER_OPTION_EXTENSION, QR_LOGICAL_INPUT_EXTENSION,
    QR_LOGICAL_OPTION_EXTENSION, QR_RESIDENT_OPTION_EXTENSION,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::factor::qr")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "qr",
    op_kind: GpuOpKind::Custom("qr-factor"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("qr")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may download to host and re-upload results; the bundled WGPU backend currently uses the runtime QR implementation.",
};

fn qr_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    qr_error_with_message(error.message, error)
}

fn qr_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn qr_invalid_argument(message: impl Into<String>) -> RuntimeError {
    qr_error_with_message(message, &QR_ERROR_INVALID_ARGUMENT)
}

fn qr_invalid_input(message: impl Into<String>) -> RuntimeError {
    qr_error_with_message(message, &QR_ERROR_INVALID_INPUT)
}

fn qr_provider_error(error: runmat_accelerate_api::ProviderExecutionError) -> RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        qr_internal_error(error.to_string()),
        error,
    )
}

fn qr_internal_error(message: impl Into<String>) -> RuntimeError {
    qr_error_with_message(message, &QR_ERROR_INTERNAL)
}

#[cfg(test)]
fn qr_type(args: &[Type], _context: &ResolveContext) -> Type {
    let Some(input) = args.first() else {
        return Type::Unknown;
    };
    match input {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            if element_count_if_known(shape.as_slice()) == Some(1) {
                return Type::Num;
            }
            if args.len() == 1 {
                let (rows, cols) = matrix_dims(shape);
                numeric_tensor_from_shape(vec![rows, cols])
            } else {
                Type::Tensor {
                    shape: Some(unknown_shape(shape.len().max(2))),
                }
            }
        }
        Type::Tensor { shape: None } | Type::Logical { shape: None } => Type::tensor(),
        Type::Num | Type::Int | Type::Bool => Type::Num,
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

fn with_qr_context(mut error: RuntimeError) -> RuntimeError {
    if error.message() == "interaction pending..." {
        return build_runtime_error("interaction pending...")
            .with_builtin(BUILTIN_NAME)
            .build();
    }
    if error.context.builtin.is_none() {
        error.context = error.context.with_builtin(BUILTIN_NAME);
    }
    error
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::factor::qr")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "qr",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "QR factorisation executes eagerly and does not participate in fusion.",
};

#[runtime_builtin(
    name = "qr",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::linalg::factor::qr"
)]
async fn qr_builtin_context(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let count = crate::output_count::current_output_count();
    let request = match count {
        None | Some(0 | 1) => QrOutputRequest::R,
        Some(2) => QrOutputRequest::Qr,
        Some(3) => QrOutputRequest::Qrp,
        Some(_) => return Err(qr_error(&QR_ERROR_INVALID_ARGUMENT)),
    };
    if let Value::SparseTensor(source) = &value {
        admit_qr_extensions(&value, rest.view())?;
        return sparse::evaluate(runtime.as_ref(), source, rest.view(), count).await;
    }
    let eval = evaluate_requested(value, rest.view(), request).await?;
    match count {
        None => Ok(eval.r()),
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![eval.r()])),
        Some(2) => Ok(Value::output_list(vec![eval.q(), eval.r()])),
        Some(3) => Ok(Value::output_list(vec![
            eval.q(),
            eval.r(),
            eval.permutation(),
        ])),
        Some(_) => unreachable!("output count was validated before QR execution"),
    }
}

mod sparse;

#[cfg(test)]
async fn qr_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    qr_builtin_context(None, value, rest).await
}

/// Output envelope containing only the values selected by its typed request.
#[derive(Clone)]
pub struct QrEval {
    q: Option<Value>,
    r: Value,
    perm_matrix: Option<Value>,
    perm_vector: Option<Value>,
    mode: QrMode,
    pivot_mode: PivotMode,
}

impl QrEval {
    fn from_provider(
        result: ProviderQrResult,
        options: &QrOptions,
        request: QrOutputRequest,
    ) -> BuiltinResult<Self> {
        let pivot_mode = options.format.pivot_mode();
        let (q, r, perm_matrix, perm_vector) = match (request, result) {
            (QrOutputRequest::R, ProviderQrResult::R { r }) => (None, r, None, None),
            (QrOutputRequest::Qr, ProviderQrResult::Qr { q, r }) => (Some(q), r, None, None),
            (QrOutputRequest::Qrp, ProviderQrResult::Qrp { q, r, permutation }) => match pivot_mode
            {
                PivotMode::Matrix => (Some(q), r, Some(permutation), None),
                PivotMode::Vector => (Some(q), r, None, Some(permutation)),
            },
            (
                QrOutputRequest::FullEnvelope | QrOutputRequest::FullEnvelopeUnpivoted,
                ProviderQrResult::FullEnvelope {
                    q,
                    r,
                    perm_matrix,
                    perm_vector,
                },
            ) => (Some(q), r, Some(perm_matrix), Some(perm_vector)),
            _ => {
                return Err(qr_internal_error(
                    "qr: provider returned a different output form",
                ))
            }
        };
        Ok(Self {
            q: q.map(Value::GpuTensor),
            r: Value::GpuTensor(r),
            perm_matrix: perm_matrix.map(Value::GpuTensor),
            perm_vector: perm_vector.map(Value::GpuTensor),
            mode: options.mode,
            pivot_mode,
        })
    }

    /// Orthogonal/unitary factor `Q`; available for QR/QRP/full-envelope requests.
    pub fn q(&self) -> Value {
        self.q.clone().expect("QR request includes Q")
    }

    /// Upper-triangular (or trapezoidal) factor `R`.
    pub fn r(&self) -> Value {
        self.r.clone()
    }

    /// Permutation output respecting the requested pivot mode.
    pub fn permutation(&self) -> Value {
        match self.pivot_mode {
            PivotMode::Matrix => self.permutation_matrix(),
            PivotMode::Vector => self.permutation_vector(),
        }
    }

    /// Matrix permutation; available for matrix-form QRP and full-envelope requests.
    pub fn permutation_matrix(&self) -> Value {
        self.perm_matrix
            .clone()
            .expect("QR request includes matrix permutation")
    }

    /// Vector permutation; available for vector-form QRP and full-envelope requests.
    pub fn permutation_vector(&self) -> Value {
        self.perm_vector
            .clone()
            .expect("QR request includes vector permutation")
    }

    /// Selected pivot mode.
    pub fn pivot_mode(&self) -> PivotMode {
        self.pivot_mode
    }

    /// Selected size mode (full or economy).
    pub fn mode(&self) -> QrMode {
        self.mode
    }
}

/// Size mode for QR outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum QrMode {
    #[default]
    Full,
    Economy,
}

/// Evaluate the builtin with full access to multiple outputs.
pub async fn evaluate(
    value: Value,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<QrEval> {
    evaluate_requested(value, args, QrOutputRequest::FullEnvelope).await
}

/// Evaluate exactly the requested public QR outputs.
pub async fn evaluate_requested(
    value: Value,
    args: crate::call::FunctionArgumentView<'_>,
    request: QrOutputRequest,
) -> BuiltinResult<QrEval> {
    admit_qr_extensions(&value, args)?;
    let options = parse_options(args).await?;
    if matches!(request, QrOutputRequest::R | QrOutputRequest::Qr)
        && options.format.requires_permutation_output()
    {
        return Err(qr_invalid_argument(
            "qr: matrix and vector permutation options require three outputs",
        ));
    }
    evaluate_with_options(value, options, request).await
}

/// Provider host fallback uses the same factorization with an explicit typed request.
pub async fn evaluate_provider(
    value: Value,
    options: runmat_accelerate_api::ProviderQrOptions,
) -> BuiltinResult<QrEval> {
    let request = match options.outputs {
        ProviderQrOutputs::R => QrOutputRequest::R,
        ProviderQrOutputs::Qr => QrOutputRequest::Qr,
        ProviderQrOutputs::Qrp(_) => QrOutputRequest::Qrp,
        ProviderQrOutputs::FullEnvelope => match options.column_order {
            ProviderQrColumnOrder::Pivoted => QrOutputRequest::FullEnvelope,
            ProviderQrColumnOrder::Unpivoted => QrOutputRequest::FullEnvelopeUnpivoted,
        },
    };
    if request.column_order() != options.column_order {
        return Err(qr_invalid_argument(
            "qr: provider QR request has inconsistent column order",
        ));
    }
    let format = match options.outputs {
        ProviderQrOutputs::Qrp(ProviderQrPivot::Vector) => QrPermutationFormat::VectorExplicit,
        _ => QrPermutationFormat::MatrixDefault,
    };
    let mode = if options.economy {
        QrMode::Economy
    } else {
        QrMode::Full
    };
    admit_qr_extensions(&value, crate::call::FunctionArgumentView::from_slice(&[]))?;
    evaluate_with_options(value, QrOptions { mode, format }, request).await
}

fn admit_qr_extensions(
    value: &Value,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    if crate::builtins::common::validation::value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &QR_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if crate::builtins::common::validation::value_has_logical_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &QR_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    for option in args {
        if crate::builtins::common::validation::value_contains_explicit_gpu(option) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &QR_RESIDENT_OPTION_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if crate::builtins::common::validation::value_has_logical_class(option) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &QR_LOGICAL_OPTION_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if crate::builtins::common::validation::value_has_native_integer_class(option) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &QR_INTEGER_OPTION_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    Ok(())
}

async fn evaluate_with_options(
    value: Value,
    options: QrOptions,
    request: QrOutputRequest,
) -> BuiltinResult<QrEval> {
    crate::builtins::common::validation::reject_typed_complex_integer(&value, BUILTIN_NAME)?;
    if let Value::GpuTensor(handle) = &value {
        gpu_helpers::require_resident_compute_owner(handle, || {
            qr_internal_error("qr: no exact owner for resident input")
        })?;
    }
    if crate::builtins::common::validation::value_has_native_integer_class(&value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(&value)
            .await?
    {
        return Err(qr_invalid_input(
            "qr: integer input must be exactly representable as double",
        ));
    }
    match value {
        Value::GpuTensor(handle) => {
            if handle.shape().len() > 2 {
                return Err(qr_invalid_input("qr: input must be 2-D"));
            }
            gpu_helpers::require_resident_compute_owner(&handle, || {
                qr_internal_error("qr: no exact owner for resident input")
            })?;
            let integer_input = runmat_accelerate_api::handle_integer_type(&handle).is_some();
            if !integer_input {
                if let Some(eval) = evaluate_gpu(&handle, &options, request).await? {
                    return Ok(eval);
                }
            }
            let owner = gpu_helpers::exact_provider_for_handle(&handle)
                .ok_or_else(|| qr_internal_error("qr: no exact owner for resident input"))?;
            let gathered = gpu_helpers::download_value_preserving_residency_async(owner, &handle)
                .await
                .map_err(with_qr_context)?;
            let dtype = if integer_input || runmat_accelerate_api::handle_is_logical(&handle) {
                Some(NumericDType::F64)
            } else {
                match runmat_accelerate_api::handle_precision(&handle) {
                    Some(runmat_accelerate_api::ProviderPrecision::F32) => Some(NumericDType::F32),
                    Some(runmat_accelerate_api::ProviderPrecision::F64) => Some(NumericDType::F64),
                    None => None,
                }
            };
            evaluate_host_value(gathered, options, request, dtype, Some(&handle)).await
        }
        other => evaluate_host_value(other, options, request, None, None).await,
    }
}

async fn evaluate_gpu(
    handle: &GpuTensorHandle,
    options: &QrOptions,
    request: QrOutputRequest,
) -> BuiltinResult<Option<QrEval>> {
    let provider = match gpu_helpers::exact_provider_for_handle(handle) {
        Some(p) => p,
        None => return Ok(None),
    };
    if runmat_accelerate_api::handle_storage(handle)
        != runmat_accelerate_api::GpuTensorStorage::Real
        || runmat_accelerate_api::handle_is_logical(handle)
        || runmat_accelerate_api::handle_precision(handle) != Some(provider.precision())
    {
        return Ok(None);
    }
    let _residency = gpu_helpers::ResidencyRestoreGuard::new(handle);
    let pivot_mode = options.format.pivot_mode();
    let outputs = match request {
        QrOutputRequest::R => ProviderQrOutputs::R,
        QrOutputRequest::Qr => ProviderQrOutputs::Qr,
        QrOutputRequest::Qrp => ProviderQrOutputs::Qrp(match pivot_mode {
            PivotMode::Matrix => ProviderQrPivot::Matrix,
            PivotMode::Vector => ProviderQrPivot::Vector,
        }),
        QrOutputRequest::FullEnvelope | QrOutputRequest::FullEnvelopeUnpivoted => {
            ProviderQrOutputs::FullEnvelope
        }
    };
    let provider_options = runmat_accelerate_api::ProviderQrOptions {
        economy: matches!(options.mode, QrMode::Economy),
        column_order: request.column_order(),
        outputs,
    };
    if std::env::var("RUNMAT_DEBUG_QR").is_ok() {
        log::debug!(
            "qr evaluate_gpu: handle={} mode={:?} outputs={:?}",
            handle.buffer_id(),
            options.mode,
            outputs
        );
    }
    if matches!(request, QrOutputRequest::Qr) && matches!(options.mode, QrMode::Economy) {
        if let Some((lhs, rhs)) = provider.take_matmul_sources(handle) {
            match provider
                .qr_power_iter(handle, Some(&lhs), &rhs, &provider_options)
                .await
            {
                Ok(Some(result)) => {
                    return Ok(Some(provider_result::admit(
                        result, handle, provider, options, request,
                    )?));
                }
                Ok(None) => {
                    // fall through to standard qr
                }
                Err(err) if err.is_unsupported() => {}
                Err(err) => return Err(qr_provider_error(err)),
            }
        }
    }
    match provider.qr(handle, provider_options).await {
        Ok(result) => Ok(Some(provider_result::admit(
            result, handle, provider, options, request,
        )?)),
        Err(error) if error.is_unsupported() => Ok(None),
        Err(error) => Err(qr_provider_error(error)),
    }
}

async fn evaluate_host_value(
    value: Value,
    options: QrOptions,
    request: QrOutputRequest,
    dtype_override: Option<NumericDType>,
    source_handle: Option<&GpuTensorHandle>,
) -> BuiltinResult<QrEval> {
    let output_dtype = dtype_override.unwrap_or_else(|| qr_output_dtype(&value));
    let matrix = extract_matrix(value).await?;
    let components = qr_factor(matrix, request.column_order(), request.is_full_envelope())?;
    assemble_eval(components, options, request, output_dtype, source_handle)
}

fn qr_output_dtype(value: &Value) -> NumericDType {
    match value {
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => NumericDType::F32,
        Value::ComplexTensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            NumericDType::F32
        }
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32) =>
        {
            NumericDType::F32
        }
        _ => NumericDType::F64,
    }
}

async fn parse_options(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<QrOptions> {
    if args.len() > 2 {
        return Err(qr_invalid_argument("qr: too many option arguments"));
    }
    let mut opts = QrOptions::default();
    for arg in args {
        if is_zero_scalar(arg).await? {
            opts.mode = QrMode::Economy;
            opts.format = QrPermutationFormat::LegacyZeroVector;
            continue;
        }
        if let Some(text) = tensor::value_to_string(arg) {
            match FactorOption::parse(&text) {
                Some(FactorOption::Economy) => opts.mode = QrMode::Economy,
                Some(FactorOption::Vector) => opts.format = QrPermutationFormat::VectorExplicit,
                Some(FactorOption::Matrix) => opts.format = QrPermutationFormat::MatrixExplicit,
                _ => return Err(qr_invalid_argument(format!("qr: unknown option '{text}'"))),
            }
            continue;
        }
        return Err(qr_invalid_argument(
            "qr: option must be numeric zero or a string ('econ', 'matrix', 'vector')",
        ));
    }
    Ok(opts)
}

async fn is_zero_scalar(value: &Value) -> BuiltinResult<bool> {
    let zero = match value {
        Value::Num(n) => *n == 0.0,
        Value::Int(i) => i.is_zero(),
        Value::Bool(b) => !b,
        Value::Tensor(t) => {
            if tensor::is_scalar_tensor(t) {
                t.integer_storage()
                    .and_then(|storage| storage.value_at(0))
                    .map_or_else(
                        || tensor::tensor_value_f64(t, 0) == 0.0,
                        |value| value.is_zero(),
                    )
            } else {
                false
            }
        }
        Value::GpuTensor(handle) => {
            if handle.shape().iter().product::<usize>() == 1 {
                let host = gpu_helpers::gather_tensor_async(handle)
                    .await
                    .map_err(with_qr_context)?;
                return Ok(host
                    .integer_storage()
                    .and_then(|storage| storage.value_at(0))
                    .map_or_else(
                        || tensor::tensor_value_f64(&host, 0) == 0.0,
                        |value| value.is_zero(),
                    ));
            }
            false
        }
        _ => false,
    };
    Ok(zero)
}

async fn extract_matrix(value: Value) -> BuiltinResult<ColMajorMatrix> {
    match value {
        Value::Tensor(t) => ColMajorMatrix::from_tensor(&t),
        Value::ComplexTensor(ct) => ColMajorMatrix::from_complex_tensor(&ct),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|err| qr_invalid_input(format!("qr: {err}")))?;
            ColMajorMatrix::from_tensor(&tensor)
        }
        Value::Num(n) => Ok(ColMajorMatrix::from_scalar(Complex64::new(n, 0.0))),
        Value::Int(i) => Ok(ColMajorMatrix::from_scalar(Complex64::new(i.to_f64(), 0.0))),
        Value::Bool(b) => Ok(ColMajorMatrix::from_scalar(Complex64::new(
            if b { 1.0 } else { 0.0 },
            0.0,
        ))),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(with_qr_context)?;
            ColMajorMatrix::from_tensor(&tensor)
        }
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => {
            Err(qr_invalid_input("qr: expected a numeric matrix"))
        }
        other => Err(qr_invalid_input(format!(
            "qr: unsupported input type {other:?}"
        ))),
    }
}

fn assemble_eval(
    components: QrComponents,
    options: QrOptions,
    request: QrOutputRequest,
    output_dtype: NumericDType,
    source_handle: Option<&GpuTensorHandle>,
) -> BuiltinResult<QrEval> {
    let rows = components.reflectors.rows;
    let cols = components.reflectors.cols;
    let factor_rows = if matches!(options.mode, QrMode::Economy) && rows >= cols {
        cols
    } else {
        rows
    };
    let q_full = (!matches!(request, QrOutputRequest::R))
        .then(|| build_q(&components.reflectors, &components.taus, factor_rows));
    let r_full = build_r(&components.reflectors, factor_rows);
    let q_value = q_full
        .as_ref()
        .map(|q| matrix_to_value(q, "qr", output_dtype))
        .transpose()?;
    let r_value = matrix_to_value(&r_full, "qr", output_dtype)?;

    let permutation = components.permutation.as_deref();
    let perm_matrix_value = if request.is_full_envelope()
        || matches!(request, QrOutputRequest::Qrp)
            && matches!(options.format.pivot_mode(), PivotMode::Matrix)
    {
        let indices = permutation.expect("pivoted request has permutation indices");
        Some(matrix_to_value(
            &perm_matrix(cols, indices),
            "qr",
            NumericDType::F64,
        )?)
    } else {
        None
    };
    let perm_vector_value = if request.is_full_envelope()
        || matches!(request, QrOutputRequest::Qrp)
            && matches!(options.format.pivot_mode(), PivotMode::Vector)
    {
        let indices = permutation.expect("pivoted request has permutation indices");
        Some(pivot_vector_to_value(indices)?)
    } else {
        None
    };

    publish_eval(
        QrEval {
            q: q_value,
            r: r_value,
            perm_matrix: perm_matrix_value,
            perm_vector: perm_vector_value,
            mode: options.mode,
            pivot_mode: options.format.pivot_mode(),
        },
        source_handle,
    )
}

fn pivot_vector_to_value(pivot: &[usize]) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(pivot.len());
    for &idx in pivot {
        data.push((idx + 1) as f64);
    }
    let tensor = Tensor::new(data, vec![pivot.len(), 1])
        .map_err(|e| qr_internal_error(format!("qr: {e}")))?;
    Ok(Value::Tensor(tensor))
}

fn perm_matrix(size: usize, perm: &[usize]) -> ColMajorMatrix {
    let mut matrix = ColMajorMatrix::zeros(size, size);
    for (col, &row_idx) in perm.iter().enumerate() {
        if row_idx < size {
            matrix.set(row_idx, col, Complex64::new(1.0, 0.0));
        }
    }
    matrix
}

fn matrix_to_value(
    matrix: &ColMajorMatrix,
    label: &str,
    dtype: NumericDType,
) -> BuiltinResult<Value> {
    matrix.to_value(label, dtype)
}

fn publish_eval(eval: QrEval, source_handle: Option<&GpuTensorHandle>) -> BuiltinResult<QrEval> {
    let Some(input) = source_handle else {
        return Ok(eval);
    };
    let provider = gpu_helpers::exact_provider_for_handle(input)
        .ok_or_else(|| qr_internal_error("qr: no exact owner for resident output"))?;
    let mut uploaded: Vec<GpuTensorHandle> = Vec::with_capacity(4);
    for value in [
        eval.q.as_ref(),
        Some(&eval.r),
        eval.perm_matrix.as_ref(),
        eval.perm_vector.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        let protected = std::iter::once(input)
            .chain(uploaded.iter())
            .collect::<Vec<_>>();
        let handle = match value {
            Value::Tensor(tensor) => {
                gpu_helpers::upload_tensor_protecting(provider, tensor, &protected)
            }
            Value::ComplexTensor(tensor) => {
                gpu_helpers::upload_complex_tensor_protecting(provider, tensor, &protected)
            }
            _ => return Err(qr_internal_error("qr: unexpected host factor value")),
        }
        .map_err(with_qr_context)?;
        uploaded.push(handle);
    }
    let provenance_error = uploaded
        .iter_mut()
        .find_map(|output| gpu_helpers::propagate_output_provenance(output, [input]).err());
    if let Some(error) = provenance_error {
        return Err(gpu_helpers::reject_gpu_outputs(
            uploaded,
            with_qr_context(error),
        ));
    }
    let mut outputs = uploaded.into_iter();
    let adopt = |handle: GpuTensorHandle| -> BuiltinResult<Value> {
        gpu_helpers::resident_gpu_value(handle).map_err(with_qr_context)
    };
    let q = eval
        .q
        .map(|_| adopt(outputs.next().expect("requested Q")))
        .transpose()?;
    let r = adopt(outputs.next().expect("requested R"))?;
    let perm_matrix = eval
        .perm_matrix
        .map(|_| adopt(outputs.next().expect("requested matrix P")))
        .transpose()?;
    let perm_vector = eval
        .perm_vector
        .map(|_| adopt(outputs.next().expect("requested vector P")))
        .transpose()?;
    Ok(QrEval {
        q,
        r,
        perm_matrix,
        perm_vector,
        mode: eval.mode,
        pivot_mode: eval.pivot_mode,
    })
}

struct QrComponents {
    reflectors: ColMajorMatrix,
    taus: Vec<Complex64>,
    permutation: Option<Vec<usize>>,
}

fn qr_factor(
    mut matrix: ColMajorMatrix,
    order: ProviderQrColumnOrder,
    full_envelope: bool,
) -> BuiltinResult<QrComponents> {
    let rows = matrix.rows;
    let cols = matrix.cols;
    let min_dim = rows.min(cols);
    let mut taus = vec![Complex64::new(0.0, 0.0); min_dim];
    let mut permutation = (matches!(order, ProviderQrColumnOrder::Pivoted) || full_envelope)
        .then(|| (0..cols).collect::<Vec<_>>());
    let mut col_norms = matches!(order, ProviderQrColumnOrder::Pivoted).then(|| {
        (0..cols)
            .map(|j| column_norm_from(&matrix, j, 0))
            .collect::<Vec<_>>()
    });

    for (k, tau_slot) in taus.iter_mut().enumerate().take(min_dim) {
        if let (Some(norms), Some(indices)) = (&mut col_norms, &mut permutation) {
            let pivot = select_pivot(norms, k);
            if pivot != k {
                matrix.swap_columns(k, pivot);
                norms.swap(k, pivot);
                indices.swap(k, pivot);
            }
        }

        let tau = {
            let column_slice = matrix.column_segment_mut(k, k);
            householder(column_slice)
        };
        *tau_slot = tau;
        if tau != Complex64::new(0.0, 0.0) {
            let v = householder_vector(&matrix, k);
            for j in (k + 1)..cols {
                apply_householder(&mut matrix, &v, tau, k, j);
            }
        }

        if let Some(norms) = &mut col_norms {
            for (j, norm) in norms.iter_mut().enumerate().skip(k + 1) {
                *norm = column_norm_from(&matrix, j, k + 1);
            }
        }
    }

    Ok(QrComponents {
        reflectors: matrix,
        taus,
        permutation,
    })
}

fn select_pivot(col_norms: &[f64], start: usize) -> usize {
    use std::cmp::Ordering;

    col_norms
        .iter()
        .enumerate()
        .skip(start)
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(start)
}

fn column_norm_from(matrix: &ColMajorMatrix, col: usize, start_row: usize) -> f64 {
    let mut norm: f64 = 0.0;
    for row in start_row..matrix.rows {
        norm = norm.hypot(matrix.get(row, col).norm());
    }
    norm
}

fn householder(column: &mut [Complex64]) -> Complex64 {
    if column.is_empty() {
        return Complex64::new(0.0, 0.0);
    }
    let alpha = column[0];
    let mut tail_norm: f64 = 0.0;
    for z in column.iter().skip(1) {
        tail_norm = tail_norm.hypot(z.norm());
    }

    if tail_norm == 0.0 && alpha == Complex64::new(0.0, 0.0) {
        column[0] = Complex64::new(0.0, 0.0);
        for z in column.iter_mut().skip(1) {
            *z = Complex64::new(0.0, 0.0);
        }
        return Complex64::new(0.0, 0.0);
    }

    if tail_norm == 0.0 && alpha.im == 0.0 && alpha.re >= 0.0 {
        for z in column.iter_mut().skip(1) {
            *z = Complex64::new(0.0, 0.0);
        }
        return Complex64::new(0.0, 0.0);
    }

    let alpha_abs = alpha.norm();
    let total_norm = alpha_abs.hypot(tail_norm);
    let sign = if alpha_abs == 0.0 {
        Complex64::new(1.0, 0.0)
    } else {
        Complex64::new(alpha.re / alpha_abs, alpha.im / alpha_abs)
    };
    let beta = -sign * Complex64::new(total_norm, 0.0);
    let tau = if beta == Complex64::new(0.0, 0.0) {
        Complex64::new(0.0, 0.0)
    } else {
        Complex64::new(1.0, 0.0)
            - crate::builtins::common::complex_arithmetic::divide_complex64(alpha, beta)
    };
    // The unscaled difference can overflow even when the reflector and every
    // stored factor are finite. Form the equivalent tail quotient in units of
    // the (finite) column norm.
    let denominator = Complex64::new(
        alpha.re / total_norm - beta.re / total_norm,
        alpha.im / total_norm - beta.im / total_norm,
    );
    if denominator == Complex64::new(0.0, 0.0) {
        for z in column.iter_mut().skip(1) {
            *z = Complex64::new(0.0, 0.0);
        }
    } else {
        for z in column.iter_mut().skip(1) {
            let scaled = Complex64::new(z.re / total_norm, z.im / total_norm);
            *z = crate::builtins::common::complex_arithmetic::divide_complex64(scaled, denominator);
        }
    }
    column[0] = beta;
    tau
}

fn householder_vector(matrix: &ColMajorMatrix, k: usize) -> Vec<Complex64> {
    let mut v = Vec::with_capacity(matrix.rows - k);
    v.push(Complex64::new(1.0, 0.0));
    for row in (k + 1)..matrix.rows {
        v.push(matrix.get(row, k));
    }
    v
}

fn apply_householder(
    matrix: &mut ColMajorMatrix,
    v: &[Complex64],
    tau: Complex64,
    k: usize,
    target_col: usize,
) {
    let mut dot = Complex64::new(0.0, 0.0);
    for (offset, vi) in v.iter().enumerate() {
        let row = k + offset;
        dot += vi.conj() * matrix.get(row, target_col);
    }
    dot *= tau;
    for (offset, vi) in v.iter().enumerate() {
        let row = k + offset;
        let updated = matrix.get(row, target_col) - vi * dot;
        matrix.set(row, target_col, updated);
    }
}

fn build_q(reflectors: &ColMajorMatrix, taus: &[Complex64], columns: usize) -> ColMajorMatrix {
    let rows = reflectors.rows;
    let mut q = ColMajorMatrix::zeros(rows, columns);
    for diagonal in 0..columns {
        q.set(diagonal, diagonal, Complex64::new(1.0, 0.0));
    }
    for (k, &tau) in taus.iter().enumerate().rev() {
        if tau == Complex64::new(0.0, 0.0) {
            continue;
        }
        let v = householder_vector(reflectors, k);
        for col in 0..columns {
            let mut dot = Complex64::new(0.0, 0.0);
            for (offset, vi) in v.iter().enumerate() {
                let row = k + offset;
                dot += vi.conj() * q.get(row, col);
            }
            dot *= tau;
            for (offset, vi) in v.iter().enumerate() {
                let row = k + offset;
                let updated = q.get(row, col) - vi * dot;
                q.set(row, col, updated);
            }
        }
    }
    q
}

fn build_r(reflectors: &ColMajorMatrix, rows: usize) -> ColMajorMatrix {
    let cols = reflectors.cols;
    let mut r = ColMajorMatrix::zeros(rows, cols);
    for col in 0..cols {
        for row in 0..rows {
            if row <= col {
                r.set(row, col, reflectors.get(row, col));
            }
        }
    }
    r
}

struct ColMajorMatrix {
    rows: usize,
    cols: usize,
    data: Vec<Complex64>,
}

impl ColMajorMatrix {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Complex64::new(0.0, 0.0); rows.saturating_mul(cols)],
        }
    }

    fn from_scalar(value: Complex64) -> Self {
        Self {
            rows: 1,
            cols: 1,
            data: vec![value],
        }
    }

    fn from_tensor(tensor: &Tensor) -> BuiltinResult<Self> {
        if tensor.shape().len() > 2 {
            return Err(qr_invalid_input("qr: input must be 2-D"));
        }
        let rows = tensor.rows();
        let cols = tensor.cols();
        let values = tensor::tensor_values_f64_cow(tensor);
        let mut data = vec![Complex64::new(0.0, 0.0); rows.saturating_mul(cols)];
        for col in 0..cols {
            for row in 0..rows {
                let idx = row + col * rows;
                data[idx] = Complex64::new(values[idx], 0.0);
            }
        }
        Ok(Self { rows, cols, data })
    }

    fn from_complex_tensor(tensor: &ComplexTensor) -> BuiltinResult<Self> {
        if tensor.shape().len() > 2 {
            return Err(qr_invalid_input("qr: input must be 2-D"));
        }
        let rows = tensor.rows();
        let cols = tensor.cols();
        let values = tensor.materialize_f64();
        let mut data = vec![Complex64::new(0.0, 0.0); rows.saturating_mul(cols)];
        for col in 0..cols {
            for row in 0..rows {
                let idx = row + col * rows;
                let (re, im) = values[idx];
                data[idx] = Complex64::new(re, im);
            }
        }
        Ok(Self { rows, cols, data })
    }

    fn to_value(&self, label: &str, dtype: NumericDType) -> BuiltinResult<Value> {
        if self.data.iter().all(|z| z.im == 0.0) {
            let mut real_data = Vec::with_capacity(self.rows * self.cols);
            for col in 0..self.cols {
                for row in 0..self.rows {
                    real_data.push(self.get(row, col).re);
                }
            }
            let tensor = Tensor::new_with_dtype(real_data, vec![self.rows, self.cols], dtype)
                .map_err(|e| qr_internal_error(format!("{label}: {e}")))?;
            Ok(Value::Tensor(tensor))
        } else {
            let mut complex_data = Vec::with_capacity(self.rows * self.cols);
            for col in 0..self.cols {
                for row in 0..self.rows {
                    let val = self.get(row, col);
                    complex_data.push((val.re, val.im));
                }
            }
            let tensor = ComplexTensor::from_f64_values_with_dtype(
                complex_data,
                vec![self.rows, self.cols],
                dtype,
            )
            .map_err(|e| qr_internal_error(format!("{label}: {e}")))?;
            Ok(Value::ComplexTensor(tensor))
        }
    }

    fn get(&self, row: usize, col: usize) -> Complex64 {
        self.data[row + col * self.rows]
    }

    fn set(&mut self, row: usize, col: usize, value: Complex64) {
        let idx = row + col * self.rows;
        self.data[idx] = value;
    }

    fn swap_columns(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        for row in 0..self.rows {
            let idx_a = row + a * self.rows;
            let idx_b = row + b * self.rows;
            self.data.swap(idx_a, idx_b);
        }
    }

    fn column_segment_mut(&mut self, col: usize, start_row: usize) -> &mut [Complex64] {
        let offset = start_row + col * self.rows;
        let len = self.rows.saturating_sub(start_row);
        &mut self.data[offset..offset.saturating_add(len)]
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{IntegerStorage, Tensor as Matrix};

    fn tensor_from_value(value: Value) -> Matrix {
        match value {
            Value::Tensor(t) => t,
            other => panic!("expected dense tensor, got {other:?}"),
        }
    }

    fn tensor_close(a: &Matrix, b: &Matrix, tol: f64) {
        assert_eq!(a.shape(), b.shape());
        for (lhs, rhs) in a.materialize_f64().iter().zip(&b.materialize_f64()) {
            if (lhs - rhs).abs() > tol {
                panic!("tensor mismatch: lhs={lhs}, rhs={rhs}, tol={tol}");
            }
        }
    }

    #[test]
    fn qr_type_returns_tensor_for_arrays() {
        let out = qr_type(
            &[Type::Tensor {
                shape: Some(vec![Some(3), Some(2)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(2)])
            }
        );
    }

    #[test]
    fn qr_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = QR_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"R = qr(A)"));
        assert!(labels.contains(&"R = qr(A, option)"));
        assert!(labels.contains(&"R = qr(A, option1, option2)"));
        assert!(labels.contains(&"[Q, R] = qr(A)"));
        assert!(labels.contains(&"[Q, R] = qr(A, option)"));
        assert!(labels.contains(&"[Q, R] = qr(A, option1, option2)"));
        assert!(labels.contains(&"[Q, R, P] = qr(A)"));
        assert!(labels.contains(&"[Q, R, P] = qr(A, option)"));
        assert!(labels.contains(&"[Q, R, P] = qr(A, option1, option2)"));
    }

    #[test]
    fn qr_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = QR_DESCRIPTOR.errors.iter().map(|err| err.code).collect();
        assert!(codes.contains(&"RM.QR.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.QR.INVALID_INPUT"));
        assert!(codes.contains(&"RM.QR.INTERNAL"));
    }

    #[test]
    fn qr_matrix_conversion_reads_typed_integer_storage_exactly() {
        let tensor = Matrix::new_integer(IntegerStorage::I16(vec![1, 2, 3, 4, 5, 6]), vec![3, 2])
            .expect("typed integer tensor");

        let matrix = ColMajorMatrix::from_tensor(&tensor).expect("matrix");
        assert_eq!(matrix.rows, 3);
        assert_eq!(matrix.cols, 2);
        assert_eq!(
            matrix.data,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
            ]
        );
    }

    #[test]
    fn qr_resident_integer_gates_then_uses_checked_owner_fallback() {
        test_support::with_test_provider(|provider| {
            let input =
                Matrix::new_integer(IntegerStorage::U16(vec![2, 0, 0, 4]), vec![2, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &input)
                .expect("integer upload")
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("explicit input view");
            {
                let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
                let err = match evaluate(Value::GpuTensor(handle.clone()), &[]) {
                    Err(err) => err,
                    Ok(_) => panic!("strict mode must gate resident integer input"),
                };
                assert_eq!(
                    err.identifier(),
                    QR_INTEGER_INPUT_EXTENSION.error_identifier
                );
            }
            let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
            let eval = evaluate(Value::GpuTensor(handle), &[]).expect("resident qr");
            for output in [eval.q(), eval.r(), eval.permutation_matrix()] {
                let Value::GpuTensor(output) = output else {
                    panic!("resident fallback must restore every matrix output");
                };
                assert_eq!(runmat_accelerate_api::handle_integer_type(&output), None);
                assert!(runmat_accelerate_api::handle_is_explicit(&output));
            }
        });
    }

    #[test]
    fn qr_typed_and_resident_integer_zero_select_vector_economy_form() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let matrix = || Matrix::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let typed = evaluate(
            Value::Tensor(matrix()),
            &[Value::Int(runmat_value::IntValue::U16(0))],
        )
        .expect("typed zero option");
        assert_eq!(typed.mode(), QrMode::Economy);
        assert_eq!(typed.pivot_mode(), PivotMode::Vector);

        test_support::with_test_provider(|provider| {
            let zero = Matrix::new_integer(IntegerStorage::U16(vec![0]), vec![1, 1]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &zero).expect("integer upload");
            let handle = handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("validated fixture provenance");
            {
                let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
                let err =
                    match evaluate(Value::Tensor(matrix()), &[Value::GpuTensor(handle.clone())]) {
                        Err(err) => err,
                        Ok(_) => panic!("resident option must be gated in strict mode"),
                    };
                assert_eq!(
                    err.identifier(),
                    QR_RESIDENT_OPTION_EXTENSION.error_identifier
                );
            }
            let resident = evaluate(Value::Tensor(matrix()), &[Value::GpuTensor(handle)])
                .expect("resident zero option");
            assert_eq!(resident.mode(), QrMode::Economy);
            assert_eq!(resident.pivot_mode(), PivotMode::Vector);
        });
    }

    #[test]
    fn qr_invalid_argument_identifier_is_stable() {
        match evaluate(Value::Num(1.0), &[Value::from("nope")]) {
            Err(err) => assert_eq!(err.identifier(), QR_ERROR_INVALID_ARGUMENT.identifier),
            Ok(_) => panic!("expected invalid option error"),
        }
        match evaluate_requested(
            Value::Num(1.0),
            &[Value::from("vector")],
            QrOutputRequest::Qr,
        ) {
            Err(err) => assert_eq!(err.identifier(), QR_ERROR_INVALID_ARGUMENT.identifier),
            Ok(_) => panic!("permutation form requires three outputs"),
        }
    }

    #[test]
    fn qr_invalid_input_identifier_is_stable() {
        match evaluate(Value::from("x"), &[]) {
            Err(err) => assert_eq!(err.identifier(), QR_ERROR_INVALID_INPUT.identifier),
            Ok(_) => panic!("expected invalid input error"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn qr_output_count_selects_unpivoted_or_pivoted_column_order() {
        let data = vec![1.0, 0.0, 0.0, 100.0];
        let a = Matrix::new(data.clone(), vec![2, 2]).unwrap();
        let r_value = qr_builtin(Value::Tensor(a.clone()), Vec::new()).expect("qr");
        let unpivoted = evaluate_requested(Value::Tensor(a.clone()), &[], QrOutputRequest::Qr)
            .expect("two-output QR");
        let pivoted = evaluate_requested(Value::Tensor(a.clone()), &[], QrOutputRequest::Qrp)
            .expect("three-output QR");
        let r_eval = tensor_from_value(unpivoted.r());
        let r_builtin = tensor_from_value(r_value);
        tensor_close(&r_eval, &r_builtin, 1e-10);
        let qr =
            crate::builtins::common::matrix::matrix_mul(&tensor_from_value(unpivoted.q()), &r_eval)
                .expect("Q*R");
        tensor_close(&qr, &a, 1e-10);
        let pivot_r = tensor_from_value(pivoted.r());
        assert_ne!(r_eval.materialize_f64(), pivot_r.materialize_f64());
        let ap = crate::builtins::common::matrix::matrix_mul(
            &a,
            &tensor_from_value(pivoted.permutation_matrix()),
        )
        .expect("A*P");
        let pivot_qr =
            crate::builtins::common::matrix::matrix_mul(&tensor_from_value(pivoted.q()), &pivot_r)
                .expect("pivoted Q*R");
        tensor_close(&pivot_qr, &ap, 1e-10);
    }

    #[test]
    fn qr_single_input_keeps_factor_class_and_invalid_arity_precedes_factorization() {
        let a = Matrix::from_f32(vec![1.0, 0.0, 0.0, 100.0], vec![2, 2]).unwrap();
        let eval =
            evaluate_requested(Value::Tensor(a), &[], QrOutputRequest::Qr).expect("single QR");
        assert_eq!(
            tensor_from_value(eval.q()).numeric_dtype(),
            NumericDType::F32
        );
        assert_eq!(
            tensor_from_value(eval.r()).numeric_dtype(),
            NumericDType::F32
        );
        let complex = ComplexTensor::from_f32(
            vec![(1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (100.0, -1.0)],
            vec![2, 2],
        )
        .unwrap();
        let complex_eval =
            evaluate_requested(Value::ComplexTensor(complex), &[], QrOutputRequest::Qr)
                .expect("complex single QR");
        let Value::ComplexTensor(complex_r) = complex_eval.r() else {
            panic!("complex single R retains its complex domain");
        };
        assert_eq!(complex_r.numeric_dtype(), NumericDType::F32);
        let _outputs = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
        let error = qr_builtin(Value::from("invalid input"), Vec::new())
            .expect_err("output arity must reject before input inspection");
        assert_eq!(error.identifier(), QR_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn qr_scaled_nonzero_diagonal_is_not_cleaned_to_zero() {
        let a = Matrix::new(vec![1.0e-13, 0.0, 0.0, 2.0e-13], vec![2, 2]).unwrap();
        let eval =
            evaluate_requested(Value::Tensor(a), &[], QrOutputRequest::R).expect("scaled QR");
        let r = tensor_from_value(eval.r()).materialize_f64();
        assert_eq!(r[0], 1.0e-13);
        assert_eq!(r[3], 2.0e-13);

        let original = Complex64::new(1.0e200, 1.0e200);
        let mut column = [original, Complex64::new(1.0e200, 0.0)];
        let tau = householder(&mut column);
        assert!(tau.re.is_finite() && tau.im.is_finite());
        assert!(column
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite()));
        let reconstructed_tail = (original - column[0]) * column[1];
        assert!((reconstructed_tail.re / 1.0e200 - 1.0).abs() < 1.0e-12);
        assert!((reconstructed_tail.im / 1.0e200).abs() < 1.0e-12);

        let mut near_limit = [
            Complex64::new(1.0e308, 1.0e308),
            Complex64::new(1.0e308, 0.0),
        ];
        let tau = householder(&mut near_limit);
        assert!(tau.re.is_finite() && tau.im.is_finite());
        assert!(near_limit
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn qr_three_outputs_reconstructs_input() {
        let data = vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let a = Matrix::new(data.clone(), vec![3, 2]).unwrap();
        let eval = evaluate(Value::Tensor(a.clone()), &[]).expect("evaluate");
        let q = tensor_from_value(eval.q());
        let r = tensor_from_value(eval.r());
        let e = tensor_from_value(eval.permutation_matrix());
        let mut qtq_data = vec![0.0; q.cols() * q.cols()];
        for i in 0..q.cols() {
            for j in 0..q.cols() {
                let mut sum = 0.0;
                for k in 0..q.rows() {
                    sum += q.materialize_f64()[k + i * q.rows()]
                        * q.materialize_f64()[k + j * q.rows()];
                }
                qtq_data[i + j * q.cols()] = sum;
            }
        }
        let qtq = Matrix::new(qtq_data, vec![q.cols(), q.cols()]).unwrap();
        let identity = Matrix::new(
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            vec![3, 3],
        )
        .unwrap();
        tensor_close(&qtq, &identity, 1e-10);

        let qr = crate::builtins::common::matrix::matrix_mul(&q, &r).expect("Q*R");
        let ae = crate::builtins::common::matrix::matrix_mul(&a, &e).expect("A*E");
        tensor_close(&qr, &ae, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn qr_vector_option_returns_pivot_vector() {
        let data = vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let a = Matrix::new(data, vec![3, 2]).unwrap();
        let eval = evaluate(Value::Tensor(a), &[Value::from("vector")]).expect("evaluate");
        assert_eq!(eval.pivot_mode(), PivotMode::Vector);
        let vector = tensor_from_value(eval.permutation());
        assert_eq!(vector.shape(), [2, 1]);
        assert!(vector
            .materialize_f64()
            .iter()
            .all(|v| *v == 1.0 || *v == 2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn qr_economy_shapes_for_tall_matrix() {
        let data: Vec<f64> = (0..12).map(|i| (i + 1) as f64).collect();
        let a = Matrix::new(data, vec![4, 3]).unwrap();
        let eval = evaluate(Value::Tensor(a), &[Value::from(0.0)]).expect("evaluate econ");
        assert_eq!(eval.mode(), QrMode::Economy);
        let q = tensor_from_value(eval.q());
        let r = tensor_from_value(eval.r());
        assert_eq!(q.shape(), [4, 3]);
        assert_eq!(r.shape(), [3, 3]);
    }

    #[test]
    fn qr_economy_option_reads_typed_integer_scalar_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let data: Vec<f64> = (0..12).map(|i| (i + 1) as f64).collect();
        let a = Matrix::new(data, vec![4, 3]).unwrap();
        let option = Matrix::new_integer(IntegerStorage::U8(vec![0]), vec![1, 1]).expect("option");

        let eval = evaluate(Value::Tensor(a), &[Value::Tensor(option)]).expect("evaluate econ");
        assert_eq!(eval.mode(), QrMode::Economy);
        assert_eq!(tensor_from_value(eval.q()).shape(), [4, 3]);
        assert_eq!(tensor_from_value(eval.r()).shape(), [3, 3]);
    }

    #[test]
    fn qr_economy_option_zero_classifier_reads_all_integer_storage_variants() {
        let options = [
            IntegerStorage::I8(vec![0]),
            IntegerStorage::I16(vec![0]),
            IntegerStorage::I32(vec![0]),
            IntegerStorage::I64(vec![0]),
            IntegerStorage::U8(vec![0]),
            IntegerStorage::U16(vec![0]),
            IntegerStorage::U32(vec![0]),
            IntegerStorage::U64(vec![0]),
        ];

        for storage in options {
            let option = Matrix::new_integer(storage, vec![1, 1]).expect("option");
            assert!(block_on(is_zero_scalar(&Value::Tensor(option))).expect("zero option"));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn qr_economy_wide_matrix_matches_full() {
        let data: Vec<f64> = (0..12).map(|i| (i + 1) as f64).collect();
        let a = Matrix::new(data, vec![3, 4]).unwrap();
        let full = evaluate(Value::Tensor(a.clone()), &[]).expect("full");
        let econ = evaluate(Value::Tensor(a), &[Value::from("econ")]).expect("econ");
        let q_full = tensor_from_value(full.q());
        let r_full = tensor_from_value(full.r());
        let q_econ = tensor_from_value(econ.q());
        let r_econ = tensor_from_value(econ.r());
        tensor_close(&q_full, &q_econ, 1e-10);
        tensor_close(&r_full, &r_econ, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn qr_gpu_provider_returns_gpu_results() {
        test_support::with_test_provider(|provider| {
            let data: Vec<f64> = vec![1.0, 4.0, 2.0, 5.0];
            let a = Matrix::new(data.clone(), vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &a.materialize_f64(),
                shape: a.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu_eval = evaluate(Value::GpuTensor(handle), &[]).expect("evaluate gpu");
            assert!(matches!(gpu_eval.q(), Value::GpuTensor(_)));
            assert!(matches!(gpu_eval.r(), Value::GpuTensor(_)));
            assert!(matches!(gpu_eval.permutation_matrix(), Value::GpuTensor(_)));
            assert!(matches!(gpu_eval.permutation_vector(), Value::GpuTensor(_)));

            let gathered_q = test_support::gather(gpu_eval.q()).expect("gather Q");
            let gathered_r = test_support::gather(gpu_eval.r()).expect("gather R");
            let gathered_p = test_support::gather(gpu_eval.permutation_matrix())
                .expect("gather permutation matrix");
            let gathered_vec = test_support::gather(gpu_eval.permutation_vector())
                .expect("gather permutation vector");

            let host_eval = evaluate(Value::Tensor(a), &[]).expect("host eval");
            let host_q = tensor_from_value(host_eval.q());
            let host_r = tensor_from_value(host_eval.r());
            let host_p = tensor_from_value(host_eval.permutation_matrix());
            let host_vec = tensor_from_value(host_eval.permutation_vector());

            tensor_close(&gathered_q, &host_q, 1e-10);
            tensor_close(&gathered_r, &host_r, 1e-10);
            tensor_close(&gathered_p, &host_p, 1e-10);
            tensor_close(&gathered_vec, &host_vec, 1e-10);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn qr_wgpu_matches_cpu() {
        let _accel_guard = test_support::accel_test_lock();
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }

        let tol = match runmat_accelerate_api::provider()
            .expect("provider")
            .precision()
        {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };

        let tensor = Matrix::new(vec![3.0, 0.0, 4.0, 4.0, 0.0, 5.0], vec![3, 2]).unwrap();
        let host_eval = evaluate(Value::Tensor(tensor.clone()), &[]).expect("host eval");
        let host_q = tensor_from_value(host_eval.q());
        let host_r = tensor_from_value(host_eval.r());
        let host_p = tensor_from_value(host_eval.permutation_matrix());

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let provider = runmat_accelerate_api::provider().expect("provider");
        let handle = provider.upload(&view).expect("upload");
        let gpu_eval = evaluate(Value::GpuTensor(handle), &[]).expect("gpu eval");

        let gpu_q = test_support::gather(gpu_eval.q()).expect("gather Q");
        let gpu_r = test_support::gather(gpu_eval.r()).expect("gather R");
        let gpu_p = test_support::gather(gpu_eval.permutation_matrix()).expect("gather P");
        let gpu_vec =
            test_support::gather(gpu_eval.permutation_vector()).expect("gather pivot vector");

        tensor_close(&gpu_q, &host_q, tol);
        tensor_close(&gpu_r, &host_r, tol);
        tensor_close(&gpu_p, &host_p, tol);
        let host_vec = tensor_from_value(host_eval.permutation_vector());
        tensor_close(&gpu_vec, &host_vec, tol);

        let single = [4.0_f32, 0.0, 0.0, 9.0];
        let single_handle = provider
            .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                data: runmat_accelerate_api::HostNumericDataView::F32(&single),
                shape: &[2, 2],
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            })
            .expect("single upload");
        let single_eval = evaluate(Value::GpuTensor(single_handle), &[]).expect("single gpu QR");
        let precision = |value: Value| match value {
            Value::GpuTensor(handle) => runmat_accelerate_api::handle_precision(&handle),
            other => panic!("QR output lost residency: {other:?}"),
        };
        assert_eq!(
            precision(single_eval.q()),
            Some(runmat_accelerate_api::ProviderPrecision::F32)
        );
        assert_eq!(
            precision(single_eval.r()),
            Some(runmat_accelerate_api::ProviderPrecision::F32)
        );
        assert_eq!(
            precision(single_eval.permutation_matrix()),
            Some(runmat_accelerate_api::ProviderPrecision::F64)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn qr_wgpu_economy_fallback_preserves_residency_and_values() {
        let _accel_guard = test_support::accel_test_lock();
        std::env::set_var("RUNMAT_WGPU_FORCE_PRECISION", "f32");
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            std::env::remove_var("RUNMAT_WGPU_FORCE_PRECISION");
            return;
        }

        let tensor = Matrix::new(
            vec![
                1.0, 4.0, 2.0, 5.0, 3.0, 6.0, //
                7.0, 8.0, 1.5, 2.5, 3.5, 4.5,
            ],
            vec![6, 2],
        )
        .unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let provider = runmat_accelerate_api::provider().expect("provider");
        let handle = provider.upload(&view).expect("upload");
        let gpu_eval =
            evaluate(Value::GpuTensor(handle), &[Value::from("econ")]).expect("gpu economy eval");

        match gpu_eval.q() {
            Value::GpuTensor(_) => {}
            other => panic!("expected gpuArray Q, got {:?}", other),
        }
        match gpu_eval.r() {
            Value::GpuTensor(_) => {}
            other => panic!("expected gpuArray R, got {:?}", other),
        }

        let gpu_q = test_support::gather(gpu_eval.q()).expect("gather Q");
        let gpu_r = test_support::gather(gpu_eval.r()).expect("gather R");

        // Q'*Q ≈ I
        let mut qtq_data = vec![0.0; gpu_q.cols() * gpu_q.cols()];
        for i in 0..gpu_q.cols() {
            for j in 0..gpu_q.cols() {
                let mut sum = 0.0;
                for k in 0..gpu_q.rows() {
                    sum += gpu_q.materialize_f64()[k + i * gpu_q.rows()]
                        * gpu_q.materialize_f64()[k + j * gpu_q.rows()];
                }
                qtq_data[i + j * gpu_q.cols()] = sum;
            }
        }
        let qtq = Matrix::new(qtq_data, vec![gpu_q.cols(), gpu_q.cols()]).unwrap();
        let identity = Matrix::new(
            (0..(gpu_q.cols() * gpu_q.cols()))
                .map(|idx| {
                    let row = idx % gpu_q.cols();
                    let col = idx / gpu_q.cols();
                    if row == col {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>(),
            vec![gpu_q.cols(), gpu_q.cols()],
        )
        .unwrap();
        tensor_close(&qtq, &identity, 1e-3);

        // The evaluation envelope always exposes pivoted factors and their
        // permutation, so Q*R reconstructs A*P.
        let qr_product = crate::builtins::common::matrix::matrix_mul(&gpu_q, &gpu_r).expect("Q*R");
        let a_matrix =
            Matrix::new(tensor.materialize_f64().clone(), tensor.shape().to_vec()).unwrap();
        let gpu_p = test_support::gather(gpu_eval.permutation_matrix()).expect("gather P");
        let permuted = crate::builtins::common::matrix::matrix_mul(&a_matrix, &gpu_p).expect("A*P");
        tensor_close(&qr_product, &permuted, 1e-3);
    }

    fn qr_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        let rest = crate::execution::RootedValues::new(rest)
            .map_err(|error| error.into_runtime_error())?;
        block_on(super::qr_builtin(value, rest.into()))
    }

    fn evaluate(value: Value, args: &[Value]) -> BuiltinResult<QrEval> {
        block_on(super::evaluate(
            value,
            crate::call::FunctionArgumentView::from_slice(args),
        ))
    }

    fn evaluate_requested(
        value: Value,
        args: &[Value],
        request: QrOutputRequest,
    ) -> BuiltinResult<QrEval> {
        block_on(super::evaluate_requested(
            value,
            crate::call::FunctionArgumentView::from_slice(args),
            request,
        ))
    }
}
