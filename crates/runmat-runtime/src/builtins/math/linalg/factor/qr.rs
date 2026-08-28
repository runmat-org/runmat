//! MATLAB-compatible `qr` builtin with pivoted and economy forms.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::linalg::type_resolvers::{matrix_dims, numeric_tensor_from_shape};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use num_complex::Complex64;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::shape_rules::{element_count_if_known, unknown_shape};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, Tensor, Value};

use super::lu::PivotMode;

const BUILTIN_NAME: &str = "qr";

pub const QR_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "qr-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "qr with a typed-integer matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QrIntegerInputExtension"),
};

pub const QR_INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "qr-integer-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "qr with a typed-integer economy selector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QrIntegerOptionExtension"),
};
pub const QR_RESIDENT_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "qr-resident-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "qr with a gpuArray option value is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QrResidentOptionExtension"),
};
pub const QR_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "qr-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "qr with a logical matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QrLogicalInputExtension"),
};
pub const QR_LOGICAL_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "qr-logical-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "qr with a logical economy selector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:QrLogicalOptionExtension"),
};

pub const QR_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    QR_INTEGER_INPUT_EXTENSION,
    QR_INTEGER_OPTION_EXTENSION,
    QR_RESIDENT_OPTION_EXTENSION,
    QR_LOGICAL_INPUT_EXTENSION,
    QR_LOGICAL_OPTION_EXTENSION,
];

const QR_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The public full-matrix classes are single and double; RunMat mode admits real integer matrices only after an exact binary64-boundary check.",
}];

const QR_INTEGER_OPTION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "economy_selector",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented legacy economy selector is the numeric literal 0; RunMat mode additionally accepts typed integer scalar zero without a floating conversion.",
    }];

pub const QR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[Q, R, P] = qr(integer_A, options...)",
        inputs: &QR_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The gated matrix crosses one checked binary64 QR boundary. Q, R, and matrix-form P are double; vector-form P is a one-based double index vector, and automatic residency may restore outputs to the owner.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[Q, R, P] = qr(A, integer_zero)",
        inputs: &QR_INTEGER_OPTION_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed integer scalar zero selects the legacy economy/vector form exactly in RunMat mode; every nonzero or nonscalar integer option rejects as an invalid option, and an explicit resident option is independently gated before exact-owner download.",
    },
];

const QR_OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Upper triangular (or trapezoidal) factor.",
}];

const QR_OUTPUT_QR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Q",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Orthogonal/unitary factor.",
    },
    BuiltinParamDescriptor {
        name: "R",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper triangular (or trapezoidal) factor.",
    },
];

const QR_OUTPUT_QRP: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "Q",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Orthogonal/unitary factor.",
    },
    BuiltinParamDescriptor {
        name: "R",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper triangular (or trapezoidal) factor.",
    },
    BuiltinParamDescriptor {
        name: "P",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Permutation matrix or vector based on pivot mode.",
    },
];

const QR_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix to factorize.",
}];

const QR_INPUTS_A_OPTION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix to factorize.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option token (`0`, `econ`, `matrix`, or `vector`).",
    },
];

const QR_INPUTS_A_OPTION_OPTION: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix to factorize.",
    },
    BuiltinParamDescriptor {
        name: "option1",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First option token (`0`, `econ`, `matrix`, or `vector`).",
    },
    BuiltinParamDescriptor {
        name: "option2",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second option token (`0`, `econ`, `matrix`, or `vector`).",
    },
];

const QR_SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "R = qr(A)",
        inputs: &QR_INPUTS_A,
        outputs: &QR_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = qr(A, option)",
        inputs: &QR_INPUTS_A_OPTION,
        outputs: &QR_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = qr(A, option1, option2)",
        inputs: &QR_INPUTS_A_OPTION_OPTION,
        outputs: &QR_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "[Q, R] = qr(A)",
        inputs: &QR_INPUTS_A,
        outputs: &QR_OUTPUT_QR,
    },
    BuiltinSignatureDescriptor {
        label: "[Q, R] = qr(A, option)",
        inputs: &QR_INPUTS_A_OPTION,
        outputs: &QR_OUTPUT_QR,
    },
    BuiltinSignatureDescriptor {
        label: "[Q, R] = qr(A, option1, option2)",
        inputs: &QR_INPUTS_A_OPTION_OPTION,
        outputs: &QR_OUTPUT_QR,
    },
    BuiltinSignatureDescriptor {
        label: "[Q, R, P] = qr(A)",
        inputs: &QR_INPUTS_A,
        outputs: &QR_OUTPUT_QRP,
    },
    BuiltinSignatureDescriptor {
        label: "[Q, R, P] = qr(A, option)",
        inputs: &QR_INPUTS_A_OPTION,
        outputs: &QR_OUTPUT_QRP,
    },
    BuiltinSignatureDescriptor {
        label: "[Q, R, P] = qr(A, option1, option2)",
        inputs: &QR_INPUTS_A_OPTION_OPTION,
        outputs: &QR_OUTPUT_QRP,
    },
];

const QR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QR.INVALID_ARGUMENT",
    identifier: Some("RunMat:qr:InvalidArgument"),
    when: "Options are invalid or output count exceeds supported forms.",
    message: "qr currently supports at most three outputs",
};

const QR_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QR.INVALID_INPUT",
    identifier: Some("RunMat:qr:InvalidInput"),
    when: "Input is non-numeric or has rank greater than two.",
    message: "qr: expected a numeric 2-D input matrix",
};

const QR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QR.INTERNAL",
    identifier: Some("RunMat:qr:Internal"),
    when: "QR runtime fails to materialize output values.",
    message: "qr: internal runtime failure",
};

const QR_ERRORS: [BuiltinErrorDescriptor; 3] = [
    QR_ERROR_INVALID_ARGUMENT,
    QR_ERROR_INVALID_INPUT,
    QR_ERROR_INTERNAL,
];

pub const QR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &QR_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &QR_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::factor::qr")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "qr",
    op_kind: GpuOpKind::Custom("qr-factor"),
    supported_precisions: &[ScalarType::F64],
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

fn qr_internal_error(message: impl Into<String>) -> RuntimeError {
    qr_error_with_message(message, &QR_ERROR_INTERNAL)
}

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
    category = "math/linalg/factor",
    summary = "Compute QR factorizations with optional pivoting.",
    keywords = "qr,factorization,decomposition,householder",
    accel = "sink",
    sink = true,
    type_resolver(qr_type),
    descriptor(crate::builtins::math::linalg::factor::qr::QR_DESCRIPTOR),
    extensions(crate::builtins::math::linalg::factor::qr::QR_EXTENSIONS),
    integer_capabilities(crate::builtins::math::linalg::factor::qr::QR_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::linalg::factor::qr"
)]
async fn qr_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(value, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![eval.r()]));
        }
        if out_count == 2 {
            return Ok(Value::OutputList(vec![eval.q(), eval.r()]));
        }
        if out_count == 3 {
            return Ok(Value::OutputList(vec![
                eval.q(),
                eval.r(),
                eval.permutation(),
            ]));
        }
        return Err(qr_error(&QR_ERROR_INVALID_ARGUMENT));
    }
    Ok(eval.r())
}

/// Output envelope for QR factorisation, used by the builtin and the VM multi-output path.
#[derive(Clone)]
pub struct QrEval {
    q: Value,
    r: Value,
    perm_matrix: Value,
    perm_vector: Value,
    mode: QrMode,
    pivot_mode: PivotMode,
}

impl QrEval {
    /// Orthogonal/unitary factor `Q`.
    pub fn q(&self) -> Value {
        self.q.clone()
    }

    /// Upper-triangular (or trapezoidal) factor `R`.
    pub fn r(&self) -> Value {
        self.r.clone()
    }

    /// Permutation output respecting the requested pivot mode.
    pub fn permutation(&self) -> Value {
        match self.pivot_mode {
            PivotMode::Matrix => self.perm_matrix.clone(),
            PivotMode::Vector => self.perm_vector.clone(),
        }
    }

    /// Always-available permutation matrix.
    pub fn permutation_matrix(&self) -> Value {
        self.perm_matrix.clone()
    }

    /// Always-available permutation vector (1-based indices).
    pub fn permutation_vector(&self) -> Value {
        self.perm_vector.clone()
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

#[derive(Clone, Copy, Debug, Default)]
struct QrOptions {
    mode: QrMode,
    pivot: PivotMode,
}

/// Evaluate the builtin with full access to multiple outputs.
pub async fn evaluate(value: Value, args: &[Value]) -> BuiltinResult<QrEval> {
    if crate::builtins::common::validation::value_has_native_integer_class(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &QR_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if crate::builtins::common::validation::value_has_logical_class(&value) {
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
    let options = parse_options(args).await?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, BUILTIN_NAME)?;
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
            let integer_input = runmat_accelerate_api::handle_integer_type(&handle).is_some();
            if !integer_input {
                if let Some(eval) = evaluate_gpu(&handle, &options).await? {
                    return Ok(eval);
                }
            }
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(with_qr_context)?;
            let owner = gpu_helpers::exact_provider_for_handle(&handle);
            evaluate_host_value(Value::Tensor(tensor), options, owner).await
        }
        other => evaluate_host_value(other, options, None).await,
    }
}

async fn evaluate_gpu(
    handle: &GpuTensorHandle,
    options: &QrOptions,
) -> BuiltinResult<Option<QrEval>> {
    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };
    let provider_options = runmat_accelerate_api::ProviderQrOptions {
        economy: matches!(options.mode, QrMode::Economy),
        pivot: match options.pivot {
            PivotMode::Matrix => runmat_accelerate_api::ProviderQrPivot::Matrix,
            PivotMode::Vector => runmat_accelerate_api::ProviderQrPivot::Vector,
        },
    };
    if std::env::var("RUNMAT_DEBUG_QR").is_ok() {
        log::debug!(
            "qr evaluate_gpu: handle={} mode={:?} pivot={:?}",
            handle.buffer_id,
            options.mode,
            options.pivot
        );
    }
    if let Some((lhs, rhs)) = provider.take_matmul_sources(handle) {
        match provider
            .qr_power_iter(handle, Some(&lhs), &rhs, &provider_options)
            .await
        {
            Ok(Some(result)) => {
                return Ok(Some(QrEval {
                    q: Value::GpuTensor(result.q),
                    r: Value::GpuTensor(result.r),
                    perm_matrix: Value::GpuTensor(result.perm_matrix),
                    perm_vector: Value::GpuTensor(result.perm_vector),
                    mode: options.mode,
                    pivot_mode: options.pivot,
                }));
            }
            Ok(None) => {
                // fall through to standard qr
            }
            Err(err) => {
                log::debug!("qr_power_iter fallback: {}", err);
            }
        }
    }
    match provider.qr(handle, provider_options).await {
        Ok(result) => Ok(Some(QrEval {
            q: Value::GpuTensor(result.q),
            r: Value::GpuTensor(result.r),
            perm_matrix: Value::GpuTensor(result.perm_matrix),
            perm_vector: Value::GpuTensor(result.perm_vector),
            mode: options.mode,
            pivot_mode: options.pivot,
        })),
        Err(_) => Ok(None),
    }
}

async fn evaluate_host_value(
    value: Value,
    options: QrOptions,
    output_provider: Option<&'static dyn AccelProvider>,
) -> BuiltinResult<QrEval> {
    let matrix = extract_matrix(value).await?;
    let components = qr_factor(matrix)?;
    assemble_eval(components, options, output_provider)
}

async fn parse_options(args: &[Value]) -> BuiltinResult<QrOptions> {
    if args.len() > 2 {
        return Err(qr_invalid_argument("qr: too many option arguments"));
    }
    let mut opts = QrOptions::default();
    for arg in args {
        if is_zero_scalar(arg).await {
            opts.mode = QrMode::Economy;
            opts.pivot = PivotMode::Vector;
            continue;
        }
        if let Some(text) = tensor::value_to_string(arg) {
            let normalized = text.trim().to_ascii_lowercase();
            if normalized == "0" || normalized == "econ" || normalized == "economy" {
                opts.mode = QrMode::Economy;
                continue;
            }
            if normalized == "vector" {
                opts.pivot = PivotMode::Vector;
                continue;
            }
            if normalized == "matrix" {
                opts.pivot = PivotMode::Matrix;
                continue;
            }
            return Err(qr_invalid_argument(format!("qr: unknown option '{text}'")));
        }
        return Err(qr_invalid_argument(
            "qr: option must be numeric zero or a string ('econ', 'matrix', 'vector')",
        ));
    }
    Ok(opts)
}

async fn is_zero_scalar(value: &Value) -> bool {
    match value {
        Value::Num(n) => n.abs() <= EPS_SCALAR,
        Value::Int(i) => i.is_zero(),
        Value::Bool(b) => !b,
        Value::Tensor(t) => {
            if tensor::is_scalar_tensor(t) {
                t.integer_storage()
                    .and_then(|storage| storage.value_at(0))
                    .map_or_else(
                        || tensor::tensor_value_f64(t, 0).abs() <= EPS_SCALAR,
                        |value| value.is_zero(),
                    )
            } else {
                false
            }
        }
        Value::GpuTensor(handle) => {
            if handle.shape.iter().product::<usize>() == 1 {
                if let Ok(host) = gpu_helpers::gather_tensor_async(handle).await {
                    return host
                        .integer_storage()
                        .and_then(|storage| storage.value_at(0))
                        .map_or_else(
                            || tensor::tensor_value_f64(&host, 0).abs() <= EPS_SCALAR,
                            |value| value.is_zero(),
                        );
                }
            }
            false
        }
        _ => false,
    }
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
    output_provider: Option<&'static dyn AccelProvider>,
) -> BuiltinResult<QrEval> {
    let rows = components.reflectors.rows;
    let cols = components.reflectors.cols;
    let mut q_full = build_q(&components.reflectors, &components.taus);
    let mut r_full = build_full_r(&components.reflectors);
    q_full.clean(EPS_CLEAN);
    r_full.clean(EPS_CLEAN);
    let perm_matrix = perm_matrix(cols, &components.permutation);
    let perm_vector = components.permutation.clone();

    let (q_value, r_value) = match options.mode {
        QrMode::Full => (
            matrix_to_value(&q_full, "qr", output_provider)?,
            matrix_to_value(&r_full, "qr", output_provider)?,
        ),
        QrMode::Economy => {
            let mut q_econ = if rows >= cols {
                q_full.take_columns(cols)
            } else {
                q_full
            };
            let mut r_econ = if rows >= cols {
                r_full.take_rows(cols)
            } else {
                r_full
            };
            q_econ.clean(EPS_CLEAN);
            r_econ.clean(EPS_CLEAN);
            (
                matrix_to_value(&q_econ, "qr", output_provider)?,
                matrix_to_value(&r_econ, "qr", output_provider)?,
            )
        }
    };

    let perm_matrix_value = matrix_to_value(&perm_matrix, "qr", output_provider)?;
    let perm_vector_value =
        maybe_upload_value(pivot_vector_to_value(&perm_vector)?, output_provider);

    Ok(QrEval {
        q: q_value,
        r: r_value,
        perm_matrix: perm_matrix_value,
        perm_vector: perm_vector_value,
        mode: options.mode,
        pivot_mode: options.pivot,
    })
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
    output_provider: Option<&'static dyn AccelProvider>,
) -> BuiltinResult<Value> {
    let value = matrix.to_value(label)?;
    Ok(maybe_upload_value(value, output_provider))
}

fn maybe_upload_value(value: Value, output_provider: Option<&'static dyn AccelProvider>) -> Value {
    match value {
        Value::Tensor(tensor) => {
            if let Some(provider) = output_provider {
                if let Ok(handle) = gpu_helpers::upload_tensor(provider, &tensor) {
                    return Value::GpuTensor(handle);
                }
            }
            Value::Tensor(tensor)
        }
        other => other,
    }
}

struct QrComponents {
    reflectors: ColMajorMatrix,
    taus: Vec<Complex64>,
    permutation: Vec<usize>,
}

fn qr_factor(mut matrix: ColMajorMatrix) -> BuiltinResult<QrComponents> {
    let rows = matrix.rows;
    let cols = matrix.cols;
    let min_dim = rows.min(cols);
    let mut taus = vec![Complex64::new(0.0, 0.0); min_dim];
    let mut permutation: Vec<usize> = (0..cols).collect();
    let mut col_norms: Vec<f64> = (0..cols)
        .map(|j| column_norm_sq_from(&matrix, j, 0))
        .collect();

    for (k, tau_slot) in taus.iter_mut().enumerate().take(min_dim) {
        let pivot = select_pivot(&col_norms, k);
        if pivot != k {
            matrix.swap_columns(k, pivot);
            col_norms.swap(k, pivot);
            permutation.swap(k, pivot);
        }

        let tau = {
            let column_slice = matrix.column_segment_mut(k, k);
            householder(column_slice)
        };
        *tau_slot = tau;
        if tau.norm() != 0.0 {
            let v = householder_vector(&matrix, k);
            for j in (k + 1)..cols {
                apply_householder(&mut matrix, &v, tau, k, j);
            }
        }

        for (j, norm) in col_norms.iter_mut().enumerate().skip(k + 1) {
            *norm = column_norm_sq_from(&matrix, j, k + 1);
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

fn column_norm_sq_from(matrix: &ColMajorMatrix, col: usize, start_row: usize) -> f64 {
    let mut sum = 0.0;
    for row in start_row..matrix.rows {
        sum += matrix.get(row, col).norm_sqr();
    }
    sum
}

fn householder(column: &mut [Complex64]) -> Complex64 {
    if column.is_empty() {
        return Complex64::new(0.0, 0.0);
    }
    let alpha = column[0];
    let mut tail_norm_sq = 0.0;
    for z in column.iter().skip(1) {
        tail_norm_sq += z.norm_sqr();
    }

    if tail_norm_sq <= EPS_CLEAN && alpha.norm() <= EPS_CLEAN {
        column[0] = Complex64::new(0.0, 0.0);
        for z in column.iter_mut().skip(1) {
            *z = Complex64::new(0.0, 0.0);
        }
        return Complex64::new(0.0, 0.0);
    }

    if tail_norm_sq <= EPS_CLEAN && alpha.im.abs() <= EPS_CLEAN && alpha.re >= 0.0 {
        for z in column.iter_mut().skip(1) {
            *z = Complex64::new(0.0, 0.0);
        }
        return Complex64::new(0.0, 0.0);
    }

    let alpha_abs = alpha.norm();
    let total_norm = (alpha_abs * alpha_abs + tail_norm_sq).sqrt();
    let sign = if alpha_abs <= EPS_CLEAN {
        Complex64::new(1.0, 0.0)
    } else {
        alpha / Complex64::new(alpha_abs, 0.0)
    };
    let beta = -sign * Complex64::new(total_norm, 0.0);
    let tau = if beta.norm() <= EPS_CLEAN {
        Complex64::new(0.0, 0.0)
    } else {
        (beta - alpha) / beta
    };
    let denom = alpha - beta;
    if denom.norm() <= EPS_CLEAN {
        for z in column.iter_mut().skip(1) {
            *z = Complex64::new(0.0, 0.0);
        }
    } else {
        for z in column.iter_mut().skip(1) {
            *z /= denom;
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

fn build_q(reflectors: &ColMajorMatrix, taus: &[Complex64]) -> ColMajorMatrix {
    let rows = reflectors.rows;
    let mut q = ColMajorMatrix::identity(rows);
    for (k, &tau) in taus.iter().enumerate().rev() {
        if tau.norm() == 0.0 {
            continue;
        }
        let v = householder_vector(reflectors, k);
        for col in 0..rows {
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

fn build_full_r(reflectors: &ColMajorMatrix) -> ColMajorMatrix {
    let rows = reflectors.rows;
    let cols = reflectors.cols;
    let mut r = ColMajorMatrix::zeros(rows, cols);
    for col in 0..cols {
        for row in 0..rows {
            if row <= col {
                let mut val = reflectors.get(row, col);
                if val.norm() <= EPS_CLEAN {
                    val = Complex64::new(0.0, 0.0);
                } else if val.im.abs() <= EPS_CLEAN {
                    val = Complex64::new(val.re, 0.0);
                }
                r.set(row, col, val);
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

    fn identity(size: usize) -> Self {
        let mut matrix = Self::zeros(size, size);
        for i in 0..size {
            matrix.set(i, i, Complex64::new(1.0, 0.0));
        }
        matrix
    }

    fn from_scalar(value: Complex64) -> Self {
        Self {
            rows: 1,
            cols: 1,
            data: vec![value],
        }
    }

    fn from_tensor(tensor: &Tensor) -> BuiltinResult<Self> {
        if tensor.shape.len() > 2 {
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
        if tensor.shape.len() > 2 {
            return Err(qr_invalid_input("qr: input must be 2-D"));
        }
        let rows = tensor.rows;
        let cols = tensor.cols;
        let mut data = vec![Complex64::new(0.0, 0.0); rows.saturating_mul(cols)];
        for col in 0..cols {
            for row in 0..rows {
                let idx = row + col * rows;
                let (re, im) = tensor.materialize_f64()[idx];
                data[idx] = Complex64::new(re, im);
            }
        }
        Ok(Self { rows, cols, data })
    }

    fn to_value(&self, label: &str) -> BuiltinResult<Value> {
        if self.data.iter().all(|z| z.im.abs() <= EPS_CLEAN) {
            let mut real_data = Vec::with_capacity(self.rows * self.cols);
            for col in 0..self.cols {
                for row in 0..self.rows {
                    real_data.push(self.get(row, col).re);
                }
            }
            let tensor = Tensor::new(real_data, vec![self.rows, self.cols])
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
            let tensor = ComplexTensor::new(complex_data, vec![self.rows, self.cols])
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

    fn take_columns(&self, count: usize) -> ColMajorMatrix {
        let cols = count.min(self.cols);
        let mut data = vec![Complex64::new(0.0, 0.0); self.rows * cols];
        for col in 0..cols {
            for row in 0..self.rows {
                data[row + col * self.rows] = self.get(row, col);
            }
        }
        ColMajorMatrix {
            rows: self.rows,
            cols,
            data,
        }
    }

    fn take_rows(&self, count: usize) -> ColMajorMatrix {
        let rows = count.min(self.rows);
        let mut data = vec![Complex64::new(0.0, 0.0); rows * self.cols];
        for col in 0..self.cols {
            for row in 0..rows {
                data[row + col * rows] = self.get(row, col);
            }
        }
        ColMajorMatrix {
            rows,
            cols: self.cols,
            data,
        }
    }

    fn clean(&mut self, tol: f64) {
        for val in &mut self.data {
            if val.norm() <= tol {
                *val = Complex64::new(0.0, 0.0);
            } else if val.im.abs() <= tol {
                *val = Complex64::new(val.re, 0.0);
            }
        }
    }
}

const EPS_SCALAR: f64 = 1.0e-12;
const EPS_CLEAN: f64 = 1.0e-12;

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
        assert_eq!(a.shape, b.shape);
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
            let handle = gpu_helpers::upload_tensor(provider, &input).expect("integer upload");
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
            let handle =
                handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
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
    fn qr_single_output_returns_upper_triangular() {
        let data = vec![1.0, 4.0, 2.0, 5.0];
        let a = Matrix::new(data.clone(), vec![2, 2]).unwrap();
        let r_value = qr_builtin(Value::Tensor(a.clone()), Vec::new()).expect("qr");
        let eval = evaluate(Value::Tensor(a), &[]).expect("evaluate");
        let r_eval = tensor_from_value(eval.r());
        let r_builtin = tensor_from_value(r_value);
        tensor_close(&r_eval, &r_builtin, 1e-10);
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
        assert_eq!(vector.shape, vec![2, 1]);
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
        assert_eq!(q.shape, vec![4, 3]);
        assert_eq!(r.shape, vec![3, 3]);
    }

    #[test]
    fn qr_economy_option_reads_typed_integer_scalar_storage_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let data: Vec<f64> = (0..12).map(|i| (i + 1) as f64).collect();
        let a = Matrix::new(data, vec![4, 3]).unwrap();
        let option = Matrix::new_integer(IntegerStorage::U8(vec![0]), vec![1, 1]).expect("option");

        let eval = evaluate(Value::Tensor(a), &[Value::Tensor(option)]).expect("evaluate econ");
        assert_eq!(eval.mode(), QrMode::Economy);
        assert_eq!(tensor_from_value(eval.q()).shape, vec![4, 3]);
        assert_eq!(tensor_from_value(eval.r()).shape, vec![3, 3]);
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
            assert!(block_on(is_zero_scalar(&Value::Tensor(option))));
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
                shape: &a.shape,
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
            shape: &tensor.shape,
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
            shape: &tensor.shape,
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
        let a_matrix = Matrix::new(tensor.materialize_f64().clone(), tensor.shape.clone()).unwrap();
        let gpu_p = test_support::gather(gpu_eval.permutation_matrix()).expect("gather P");
        let permuted = crate::builtins::common::matrix::matrix_mul(&a_matrix, &gpu_p).expect("A*P");
        tensor_close(&qr_product, &permuted, 1e-3);
    }

    fn qr_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::qr_builtin(value, rest))
    }

    fn evaluate(value: Value, args: &[Value]) -> BuiltinResult<QrEval> {
        block_on(super::evaluate(value, args))
    }
}
