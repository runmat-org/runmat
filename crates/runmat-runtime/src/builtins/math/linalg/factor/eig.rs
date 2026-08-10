//! MATLAB-compatible `eig` builtin with host and GPU-aware semantics.
//!
//! Implements the dense eigenvalue decomposition for real and complex matrices,
//! including the vector-only form, generalized `eig(A,B)` for nonsingular `B`,
//! the `[V,D]` factorisation, and the three-output `[V,D,W]` variant that
//! returns left eigenvectors. GPU inputs are currently gathered back to the host
//! unless their owning provider implements the reserved `eig` hook; host
//! fallback outputs are restored to that owner when possible.

use std::collections::HashSet;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::linalg::type_resolvers::eig_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use nalgebra::linalg::Schur;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, IntValue, NumericDType, NumericScalar, Tensor, Value,
};
use runmat_macros::runtime_builtin;

const BUILTIN_NAME: &str = "eig";

const REAL_EPS: f64 = 1e-12;

const EIG_NONFLOATING_COEFFICIENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "eig-nonfloating-coefficient",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "eig with an integer or logical coefficient matrix is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EigNonfloatingCoefficientExtension"),
    };

pub const EIG_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [EIG_NONFLOATING_COEFFICIENT_EXTENSION];

const EIG_INTEGER_A_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Admitted integer coefficients must be exactly representable at the explicit binary64 eigensolver boundary.",
}];
const EIG_INTEGER_AB_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "First generalized coefficient matrix.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Second generalized coefficient matrix.",
    },
];

pub const EIG_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "eig(A, ...) with typed-integer A",
        inputs: &EIG_INTEGER_A_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only coefficient extension with double outputs.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "eig(A, B, ...) with a typed-integer coefficient",
        inputs: &EIG_INTEGER_AB_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each integer coefficient is checked independently before binary64 materialization.",
    },
];

const EIG_OUTPUT_D: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "d",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Eigenvalues as a column vector.",
}];

const EIG_OUTPUT_VD: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right eigenvectors.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Diagonal matrix (or vector when `vector` option is used).",
    },
];

const EIG_OUTPUT_VDW: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right eigenvectors.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Diagonal matrix (or vector when `vector` option is used).",
    },
    BuiltinParamDescriptor {
        name: "W",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left eigenvectors.",
    },
];

const EIG_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input square matrix.",
}];

const EIG_INPUTS_A_OPTIONS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input square matrix.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional selectors (`balance`, `nobalance`, `vector`, `matrix`).",
    },
];

const EIG_INPUTS_AB: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input square matrix.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second square matrix in the generalized problem A*V = B*V*D.",
    },
];

const EIG_INPUTS_AB_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input square matrix.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second square matrix in the generalized problem A*V = B*V*D.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional selectors (`balance`, `nobalance`, `vector`, `matrix`).",
    },
];

const EIG_SIGNATURES: [BuiltinSignatureDescriptor; 12] = [
    BuiltinSignatureDescriptor {
        label: "d = eig(A)",
        inputs: &EIG_INPUTS_A,
        outputs: &EIG_OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "d = eig(A, options...)",
        inputs: &EIG_INPUTS_A_OPTIONS,
        outputs: &EIG_OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "[V, D] = eig(A)",
        inputs: &EIG_INPUTS_A,
        outputs: &EIG_OUTPUT_VD,
    },
    BuiltinSignatureDescriptor {
        label: "[V, D] = eig(A, options...)",
        inputs: &EIG_INPUTS_A_OPTIONS,
        outputs: &EIG_OUTPUT_VD,
    },
    BuiltinSignatureDescriptor {
        label: "[V, D, W] = eig(A)",
        inputs: &EIG_INPUTS_A,
        outputs: &EIG_OUTPUT_VDW,
    },
    BuiltinSignatureDescriptor {
        label: "[V, D, W] = eig(A, options...)",
        inputs: &EIG_INPUTS_A_OPTIONS,
        outputs: &EIG_OUTPUT_VDW,
    },
    BuiltinSignatureDescriptor {
        label: "d = eig(A, B)",
        inputs: &EIG_INPUTS_AB,
        outputs: &EIG_OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "d = eig(A, B, options...)",
        inputs: &EIG_INPUTS_AB_OPTIONS,
        outputs: &EIG_OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "[V, D] = eig(A, B)",
        inputs: &EIG_INPUTS_AB,
        outputs: &EIG_OUTPUT_VD,
    },
    BuiltinSignatureDescriptor {
        label: "[V, D] = eig(A, B, options...)",
        inputs: &EIG_INPUTS_AB_OPTIONS,
        outputs: &EIG_OUTPUT_VD,
    },
    BuiltinSignatureDescriptor {
        label: "[V, D, W] = eig(A, B)",
        inputs: &EIG_INPUTS_AB,
        outputs: &EIG_OUTPUT_VDW,
    },
    BuiltinSignatureDescriptor {
        label: "[V, D, W] = eig(A, B, options...)",
        inputs: &EIG_INPUTS_AB_OPTIONS,
        outputs: &EIG_OUTPUT_VDW,
    },
];

const EIG_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EIG.INVALID_ARGUMENT",
    identifier: Some("RunMat:eig:InvalidArgument"),
    when: "Option arguments or requested output count are invalid.",
    message: "eig currently supports at most three outputs",
};

const EIG_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EIG.INVALID_INPUT",
    identifier: Some("RunMat:eig:InvalidInput"),
    when: "Input is unsupported or matrix shape is invalid.",
    message: "eig: input matrix must be square",
};

const EIG_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EIG.INTERNAL",
    identifier: Some("RunMat:eig:Internal"),
    when: "Runtime cannot compute or materialize eig outputs.",
    message: "eig: internal runtime failure",
};

const EIG_ERRORS: [BuiltinErrorDescriptor; 3] = [
    EIG_ERROR_INVALID_ARGUMENT,
    EIG_ERROR_INVALID_INPUT,
    EIG_ERROR_INTERNAL,
];

pub const EIG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &EIG_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &EIG_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::factor::eig")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "eig",
    op_kind: GpuOpKind::Custom("eig-factor"),
    supported_precisions: &[ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("eig")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Prefers the provider `eig` hook for standard eig(A) (WGPU reuploads host-computed results for real spectra) and falls back to the CPU implementation for generalized eig(A,B), complex spectra, or unsupported options.",
};

fn eig_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    eig_error_with_message(error.message, error)
}

fn eig_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn eig_invalid_argument(message: impl Into<String>) -> RuntimeError {
    eig_error_with_message(message, &EIG_ERROR_INVALID_ARGUMENT)
}

fn eig_invalid_input(message: impl Into<String>) -> RuntimeError {
    eig_error_with_message(message, &EIG_ERROR_INVALID_INPUT)
}

fn eig_internal_error(message: impl Into<String>) -> RuntimeError {
    eig_error_with_message(message, &EIG_ERROR_INTERNAL)
}

fn with_eig_context(mut error: RuntimeError) -> RuntimeError {
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::factor::eig")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "eig",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Eigenvalue decomposition executes eagerly and never participates in fusion.",
};

#[runtime_builtin(
    name = "eig",
    category = "math/linalg/factor",
    summary = "Compute eigenvalue decompositions.",
    keywords = "eig,eigenvalues,eigenvectors,linalg",
    accel = "sink",
    sink = true,
    type_resolver(eig_type),
    descriptor(crate::builtins::math::linalg::factor::eig::EIG_DESCRIPTOR),
    extensions(crate::builtins::math::linalg::factor::eig::EIG_EXTENSIONS),
    integer_capabilities(crate::builtins::math::linalg::factor::eig::EIG_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::linalg::factor::eig"
)]
async fn eig_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if let Some(out_count) = crate::output_count::current_output_count() {
        let require_left = out_count >= 3;
        let eval = evaluate(value, &rest, require_left).await?;
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![eval.eigenvalues()]));
        }
        if out_count == 2 {
            return Ok(Value::OutputList(vec![eval.right(), eval.diagonal()]));
        }
        if out_count == 3 {
            let left = eval.left()?;
            return Ok(Value::OutputList(vec![eval.right(), eval.diagonal(), left]));
        }
        return Err(eig_error(&EIG_ERROR_INVALID_ARGUMENT));
    }
    let eval = evaluate(value, &rest, false).await?;
    Ok(eval.eigenvalues())
}

#[derive(Clone, Debug)]
pub struct EigEval {
    eigenvalues: Value,
    diagonal_matrix: Value,
    diagonal_output: Value,
    right: Value,
    left: Option<Value>,
}

impl EigEval {
    pub fn eigenvalues(&self) -> Value {
        self.eigenvalues.clone()
    }

    pub fn diagonal(&self) -> Value {
        self.diagonal_output.clone()
    }

    pub fn diagonal_matrix(&self) -> Value {
        self.diagonal_matrix.clone()
    }

    pub fn right(&self) -> Value {
        self.right.clone()
    }

    pub fn left(&self) -> BuiltinResult<Value> {
        self.left.clone().ok_or_else(|| {
            eig_internal_error(
                "eig: left eigenvectors are not available from the active acceleration provider",
            )
        })
    }

    fn from_provider(result: runmat_accelerate_api::ProviderEigResult) -> Self {
        let runmat_accelerate_api::ProviderEigResult {
            eigenvalues,
            diagonal,
            right,
            left,
        } = result;
        EigEval {
            eigenvalues: Value::GpuTensor(eigenvalues),
            diagonal_matrix: Value::GpuTensor(diagonal.clone()),
            diagonal_output: Value::GpuTensor(diagonal),
            right: Value::GpuTensor(right),
            left: left.map(Value::GpuTensor),
        }
    }
}

#[derive(Clone, Copy)]
struct EigOptions {
    balance: bool,
    vector_output: bool,
}

impl Default for EigOptions {
    fn default() -> Self {
        Self {
            balance: true,
            vector_output: false,
        }
    }
}

#[derive(Clone)]
struct EigRequest {
    b: Option<Value>,
    options: EigOptions,
}

pub async fn evaluate(value: Value, args: &[Value], require_left: bool) -> BuiltinResult<EigEval> {
    let request = parse_request(args)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&value, BUILTIN_NAME)?;
    ensure_eig_coefficient_extension_enabled(&value)?;
    if let Some(b) = request.b {
        crate::builtins::common::validation::reject_typed_complex_integer(&b, BUILTIN_NAME)?;
        ensure_eig_coefficient_extension_enabled(&b)?;
        return evaluate_generalized(value, b, request.options, require_left).await;
    }
    let options = request.options;
    match value {
        Value::GpuTensor(handle) => {
            if is_nonfloating_coefficient(&Value::GpuTensor(handle.clone())) {
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(with_eig_context)?;
                ensure_exact_eig_integer_boundary(&Value::Tensor(tensor.clone()), "A")?;
                let eval = evaluate_host(Value::Tensor(tensor), options, require_left).await?;
                return upload_eig_eval_to_owner(eval, &handle);
            }
            if let Some(eval) = evaluate_gpu(&handle, options, require_left).await? {
                return Ok(eval);
            }
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(with_eig_context)?;
            let eval = evaluate_host(Value::Tensor(tensor), options, require_left).await?;
            upload_eig_eval_to_owner(eval, &handle)
        }
        other => {
            ensure_exact_eig_integer_boundary(&other, "A")?;
            evaluate_host(other, options, require_left).await
        }
    }
}

async fn evaluate_gpu(
    handle: &GpuTensorHandle,
    options: EigOptions,
    require_left: bool,
) -> BuiltinResult<Option<EigEval>> {
    if runmat_accelerate_api::handle_precision(handle)
        == Some(runmat_accelerate_api::ProviderPrecision::F32)
    {
        return Ok(None);
    }
    if options.vector_output {
        return Ok(None);
    }
    if !options.balance {
        return Ok(None);
    }
    let provider = match resolved_actual_eig_owner(handle) {
        Some(p) => p,
        None => return Ok(None),
    };
    let Some(order) = eig_gpu_order(handle) else {
        return Ok(None);
    };
    match provider.eig(handle, require_left).await {
        Ok(result) => {
            if valid_provider_eig_result(&result, provider, handle, order, require_left) {
                Ok(Some(EigEval::from_provider(result)))
            } else {
                free_provider_eig_result_unique(&result, provider, handle);
                Ok(None)
            }
        }
        Err(_) => Ok(None),
    }
}

fn eig_gpu_order(handle: &GpuTensorHandle) -> Option<usize> {
    match handle.shape.as_slice() {
        [rows, cols] if rows == cols => Some(*rows),
        _ => None,
    }
}

fn resolved_actual_eig_owner(
    handle: &GpuTensorHandle,
) -> Option<&'static dyn runmat_accelerate_api::AccelProvider> {
    runmat_accelerate_api::provider_for_handle(handle)
        .filter(|owner| owner.device_id() == handle.device_id)
}

fn eig_result_handles(result: &runmat_accelerate_api::ProviderEigResult) -> Vec<&GpuTensorHandle> {
    let mut handles = vec![&result.eigenvalues, &result.diagonal, &result.right];
    if let Some(left) = result.left.as_ref() {
        handles.push(left);
    }
    handles
}

fn gpu_handles_alias(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    lhs.device_id == rhs.device_id && lhs.buffer_id == rhs.buffer_id
}

fn valid_provider_eig_result(
    result: &runmat_accelerate_api::ProviderEigResult,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
    order: usize,
    require_left: bool,
) -> bool {
    if require_left && result.left.is_none() {
        return false;
    }
    let expected_precision =
        runmat_accelerate_api::handle_precision(input).unwrap_or_else(|| provider.precision());
    let expected_class = match expected_precision {
        runmat_accelerate_api::ProviderPrecision::F32 => "single",
        runmat_accelerate_api::ProviderPrecision::F64 => "double",
    };
    let expected_shapes = [vec![order, 1], vec![order, order], vec![order, order]];
    let mut identities = HashSet::new();
    for (handle, expected_shape) in eig_result_handles(result).into_iter().zip(
        expected_shapes
            .iter()
            .chain(std::iter::once(&expected_shapes[2])),
    ) {
        let storage = runmat_accelerate_api::handle_storage(handle);
        if gpu_handles_alias(handle, input)
            || !identities.insert((handle.device_id, handle.buffer_id))
            || handle.device_id != input.device_id
            || handle.shape != *expected_shape
            || resolved_actual_eig_owner(handle).is_none_or(|owner| !std::ptr::eq(owner, provider))
            || !matches!(
                storage,
                runmat_accelerate_api::GpuTensorStorage::Real
                    | runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            )
            || runmat_accelerate_api::handle_precision(handle) != Some(expected_precision)
            || runmat_accelerate_api::handle_class_name(handle)
                .is_some_and(|class_name| !class_name.eq_ignore_ascii_case(expected_class))
            || runmat_accelerate_api::handle_integer_type(handle).is_some()
            || runmat_accelerate_api::handle_is_logical(handle)
        {
            return false;
        }
    }
    true
}

fn free_provider_eig_result_unique(
    result: &runmat_accelerate_api::ProviderEigResult,
    invoked_provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &GpuTensorHandle,
) {
    let mut freed = HashSet::new();
    for handle in eig_result_handles(result) {
        let identity = (handle.device_id, handle.buffer_id);
        if gpu_handles_alias(handle, input) || !freed.insert(identity) {
            continue;
        }
        let owner = resolved_actual_eig_owner(handle).unwrap_or(invoked_provider);
        let _ = owner.free(handle);
    }
}

async fn evaluate_host(
    value: Value,
    options: EigOptions,
    require_left: bool,
) -> BuiltinResult<EigEval> {
    let precision = value_precision(&value);
    let matrix = value_to_complex_matrix(value).await?;
    cast_eig_eval_precision(compute_eigen(matrix, options, require_left)?, precision)
}

async fn evaluate_generalized(
    a_value: Value,
    b_value: Value,
    options: EigOptions,
    require_left: bool,
) -> BuiltinResult<EigEval> {
    let gpu_source = match (&a_value, &b_value) {
        (Value::GpuTensor(handle), _) => Some(handle.clone()),
        (_, Value::GpuTensor(handle)) => Some(handle.clone()),
        _ => None,
    };
    let a_value = gather_eig_coefficient(a_value).await?;
    let b_value = gather_eig_coefficient(b_value).await?;
    ensure_exact_eig_integer_boundary(&a_value, "A")?;
    ensure_exact_eig_integer_boundary(&b_value, "B")?;
    let precision = combined_value_precision(&a_value, &b_value);
    let a = value_to_complex_matrix(a_value).await?;
    let b = value_to_complex_matrix(b_value).await?;
    let eval = cast_eig_eval_precision(
        compute_generalized_eigen(a, b, options, require_left)?,
        precision,
    )?;
    match gpu_source {
        Some(source) => upload_eig_eval_to_owner(eval, &source),
        None => Ok(eval),
    }
}

async fn gather_eig_coefficient(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor_async(&handle)
            .await
            .map(Value::Tensor)
            .map_err(with_eig_context),
        other => Ok(other),
    }
}

fn ensure_eig_coefficient_extension_enabled(value: &Value) -> BuiltinResult<()> {
    if is_nonfloating_coefficient(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EIG_NONFLOATING_COEFFICIENT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn is_nonfloating_coefficient(value: &Value) -> bool {
    match value {
        Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_integer_type(handle).is_some()
                || runmat_accelerate_api::handle_is_logical(handle)
        }
        _ => false,
    }
}

fn ensure_exact_eig_integer_boundary(value: &Value, role: &str) -> BuiltinResult<()> {
    const MAX_EXACT_INTEGER: i128 = 1_i128 << 53;
    let check = |exact: i128| {
        if (-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&exact) {
            Ok(())
        } else {
            Err(eig_invalid_input(format!(
                "eig: integer {role} coefficients must be exactly representable as double"
            )))
        }
    };
    match value {
        Value::Int(value) => check(int_value_i128(value)),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            for index in 0..tensor.len() {
                if let Some(exact) = tensor.numeric_value_at(index).and_then(numeric_scalar_i128) {
                    check(exact)?;
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn int_value_i128(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => i128::from(*value),
        IntValue::I16(value) => i128::from(*value),
        IntValue::I32(value) => i128::from(*value),
        IntValue::I64(value) => i128::from(*value),
        IntValue::U8(value) => i128::from(*value),
        IntValue::U16(value) => i128::from(*value),
        IntValue::U32(value) => i128::from(*value),
        IntValue::U64(value) => i128::from(*value),
    }
}

fn numeric_scalar_i128(value: NumericScalar) -> Option<i128> {
    match value {
        NumericScalar::I8(value) => Some(i128::from(value)),
        NumericScalar::I16(value) => Some(i128::from(value)),
        NumericScalar::I32(value) => Some(i128::from(value)),
        NumericScalar::I64(value) => Some(i128::from(value)),
        NumericScalar::U8(value) => Some(i128::from(value)),
        NumericScalar::U16(value) => Some(i128::from(value)),
        NumericScalar::U32(value) => Some(i128::from(value)),
        NumericScalar::U64(value) => Some(i128::from(value)),
        NumericScalar::F32(_) | NumericScalar::F64(_) => None,
    }
}

fn value_precision(value: &Value) -> NumericDType {
    match value {
        Value::Tensor(tensor) => tensor.numeric_dtype(),
        Value::ComplexTensor(tensor) => tensor.numeric_dtype(),
        Value::GpuTensor(handle) => match runmat_accelerate_api::handle_precision(handle) {
            Some(runmat_accelerate_api::ProviderPrecision::F32) => NumericDType::F32,
            _ => NumericDType::F64,
        },
        _ => NumericDType::F64,
    }
}

fn combined_value_precision(a: &Value, b: &Value) -> NumericDType {
    if value_precision(a) == NumericDType::F32 && value_precision(b) == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    }
}

fn cast_eig_eval_precision(eval: EigEval, precision: NumericDType) -> BuiltinResult<EigEval> {
    if precision != NumericDType::F32 {
        return Ok(eval);
    }
    Ok(EigEval {
        eigenvalues: cast_eig_value_to_single(eval.eigenvalues)?,
        diagonal_matrix: cast_eig_value_to_single(eval.diagonal_matrix)?,
        diagonal_output: cast_eig_value_to_single(eval.diagonal_output)?,
        right: cast_eig_value_to_single(eval.right)?,
        left: eval.left.map(cast_eig_value_to_single).transpose()?,
    })
}

fn cast_eig_value_to_single(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => Tensor::from_f32(
            tensor
                .materialize_f64()
                .into_iter()
                .map(|value| value as f32)
                .collect(),
            tensor.shape,
        )
        .map(Value::Tensor)
        .map_err(|err| eig_internal_error(format!("eig: {err}"))),
        Value::ComplexTensor(tensor) => ComplexTensor::from_f32(
            tensor
                .materialize_f64()
                .into_iter()
                .map(|(real, imag)| (real as f32, imag as f32))
                .collect(),
            tensor.shape,
        )
        .map(Value::ComplexTensor)
        .map_err(|err| eig_internal_error(format!("eig: {err}"))),
        Value::Num(number) => Tensor::from_f32(vec![number as f32], vec![1, 1])
            .map(Value::Tensor)
            .map_err(|err| eig_internal_error(format!("eig: {err}"))),
        other => Ok(other),
    }
}

fn upload_eig_eval_to_owner(eval: EigEval, source: &GpuTensorHandle) -> BuiltinResult<EigEval> {
    let provider = runmat_accelerate_api::provider_for_handle(source)
        .ok_or_else(|| eig_internal_error("eig: source GPU provider is unavailable"))?;
    Ok(EigEval {
        eigenvalues: upload_eig_value(provider, eval.eigenvalues)?,
        diagonal_matrix: upload_eig_value(provider, eval.diagonal_matrix)?,
        diagonal_output: upload_eig_value(provider, eval.diagonal_output)?,
        right: upload_eig_value(provider, eval.right)?,
        left: eval
            .left
            .map(|value| upload_eig_value(provider, value))
            .transpose()?,
    })
}

fn upload_eig_value(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    value: Value,
) -> BuiltinResult<Value> {
    let handle = match value {
        Value::Tensor(tensor) => gpu_helpers::upload_tensor(provider, &tensor)
            .map_err(|err| eig_internal_error(format!("eig: {err}")))?,
        Value::ComplexTensor(tensor) => gpu_helpers::upload_complex_tensor(provider, &tensor)?,
        Value::Num(number) => {
            let tensor = Tensor::new(vec![number], vec![1, 1])
                .map_err(|err| eig_internal_error(format!("eig: {err}")))?;
            gpu_helpers::upload_tensor(provider, &tensor)
                .map_err(|err| eig_internal_error(format!("eig: {err}")))?
        }
        other => return Ok(other),
    };
    Ok(gpu_helpers::resident_gpu_value(handle))
}

fn parse_request(args: &[Value]) -> BuiltinResult<EigRequest> {
    let mut opts = EigOptions::default();
    let mut b = None;
    for (idx, arg) in args.iter().enumerate() {
        if let Some(text) = option_text(arg)? {
            match text.trim().to_ascii_lowercase().as_str() {
                "balance" => opts.balance = true,
                "nobalance" => opts.balance = false,
                "vector" => opts.vector_output = true,
                "matrix" => opts.vector_output = false,
                other => {
                    return Err(eig_invalid_argument(format!(
                        "eig: unknown option '{other}'"
                    )));
                }
            }
        } else if idx == 0 {
            b = Some(arg.clone());
        } else {
            return Err(eig_invalid_argument(
                "eig: option arguments must be character vectors or string scalars",
            ));
        }
    }
    Ok(EigRequest { b, options: opts })
}

fn option_text(value: &Value) -> BuiltinResult<Option<String>> {
    match value {
        Value::String(text) => Ok(Some(text.clone())),
        Value::StringArray(array) if array.data.len() == 1 => Ok(Some(array.data[0].clone())),
        Value::StringArray(_) => Err(eig_invalid_argument(
            "eig: option arguments must be character vectors or string scalars",
        )),
        Value::CharArray(chars) if chars.rows <= 1 => Ok(Some(chars.data.iter().collect())),
        Value::CharArray(_) => Err(eig_invalid_argument(
            "eig: option arguments must be character vectors or string scalars",
        )),
        _ => Ok(None),
    }
}

fn compute_eigen(
    matrix: DMatrix<Complex64>,
    options: EigOptions,
    require_left: bool,
) -> BuiltinResult<EigEval> {
    if matrix.nrows() != matrix.ncols() {
        return Err(eig_error(&EIG_ERROR_INVALID_INPUT));
    }
    let n = matrix.nrows();
    if n == 0 {
        let empty_vals = Tensor::new(Vec::new(), vec![0, 0])
            .map_err(|e| eig_internal_error(format!("eig: {e}")))?;
        let empty_mat = Tensor::new(Vec::new(), vec![0, 0])
            .map_err(|e| eig_internal_error(format!("eig: {e}")))?;
        let eigenvalues_value = Value::Tensor(empty_vals.clone());
        let diagonal_matrix_value = Value::Tensor(empty_mat.clone());
        let diagonal_output = if options.vector_output {
            eigenvalues_value.clone()
        } else {
            diagonal_matrix_value.clone()
        };
        return Ok(EigEval {
            eigenvalues: eigenvalues_value,
            diagonal_matrix: diagonal_matrix_value,
            diagonal_output,
            right: Value::Tensor(empty_mat.clone()),
            left: if require_left {
                Some(Value::Tensor(empty_mat))
            } else {
                None
            },
        });
    }

    let balanced = maybe_balance(&matrix, options.balance);
    let (eigenvalues, right) = schur_eigendecompose(&balanced)?;

    let left_value = if require_left {
        let left_matrix =
            compute_left_vectors(&balanced, &right, &eigenvalues).ok_or_else(|| {
                eig_internal_error(
                    "eig: unable to compute left eigenvectors for the requested matrix",
                )
            })?;
        Some(left_matrix)
    } else {
        None
    };

    build_eval(eigenvalues, right, left_value, options)
}

fn compute_generalized_eigen(
    a: DMatrix<Complex64>,
    b: DMatrix<Complex64>,
    options: EigOptions,
    require_left: bool,
) -> BuiltinResult<EigEval> {
    if a.nrows() != a.ncols() {
        return Err(eig_invalid_input("eig: A matrix must be square"));
    }
    if b.nrows() != b.ncols() {
        return Err(eig_invalid_input("eig: B matrix must be square"));
    }
    if a.shape() != b.shape() {
        return Err(eig_invalid_input(
            "eig: A and B must be the same size for generalized eigenvalue decomposition",
        ));
    }
    let n = a.nrows();
    if n == 0 {
        return compute_eigen(a, options, require_left);
    }

    #[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
    {
        compute_generalized_eigen_lapack(&a, &b, options, require_left)
    }

    #[cfg(not(all(feature = "blas-lapack", not(target_arch = "wasm32"))))]
    {
        compute_generalized_eigen_via_solve(a, b, options, require_left)
    }
}

#[cfg(not(all(feature = "blas-lapack", not(target_arch = "wasm32"))))]
fn compute_generalized_eigen_via_solve(
    a: DMatrix<Complex64>,
    b: DMatrix<Complex64>,
    options: EigOptions,
    require_left: bool,
) -> BuiltinResult<EigEval> {
    let lu = b.clone().lu();
    let transformed = lu.solve(&a).ok_or_else(|| {
        eig_invalid_input(
            "eig: generalized eig(A,B) with singular B requires full QZ support, which is not yet available",
        )
    })?;
    let (eigenvalues, right) = schur_eigendecompose(&transformed)?;

    let left_value = if require_left {
        let standard_left =
            compute_left_vectors(&transformed, &right, &eigenvalues).ok_or_else(|| {
                eig_internal_error(
                    "eig: unable to compute left eigenvectors for the requested generalized problem",
                )
            })?;
        let mut generalized_left = b.adjoint().lu().solve(&standard_left).ok_or_else(|| {
            eig_invalid_input(
                "eig: generalized eig(A,B) with singular B requires full QZ support, which is not yet available",
            )
        })?;
        normalize_generalized_left(&mut generalized_left, &b, &right);
        Some(generalized_left)
    } else {
        None
    };

    build_eval(eigenvalues, right, left_value, options)
}

#[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
fn compute_generalized_eigen_lapack(
    a: &DMatrix<Complex64>,
    b: &DMatrix<Complex64>,
    options: EigOptions,
    require_left: bool,
) -> BuiltinResult<EigEval> {
    let n = a.nrows();
    let n_i32 = i32::try_from(n)
        .map_err(|_| eig_invalid_input("eig: matrix is too large for LAPACK generalized eig"))?;
    let mut a_copy = matrix_to_lapack_complex(a);
    let mut b_copy = matrix_to_lapack_complex(b);
    let mut alpha = vec![lapack::c64::new(0.0, 0.0); n];
    let mut beta = vec![lapack::c64::new(0.0, 0.0); n];
    let mut vl = vec![lapack::c64::new(0.0, 0.0); n * n];
    let mut vr = vec![lapack::c64::new(0.0, 0.0); n * n];
    let mut work = vec![lapack::c64::new(0.0, 0.0); 1];
    let mut rwork = vec![0.0; 8 * n.max(1)];
    let mut info = 0i32;
    let jobvl = if require_left { b'V' } else { b'N' };
    let jobvr = b'V';

    unsafe {
        lapack::zggev(
            jobvl,
            jobvr,
            n_i32,
            &mut a_copy,
            n_i32,
            &mut b_copy,
            n_i32,
            &mut alpha,
            &mut beta,
            &mut vl,
            n_i32,
            &mut vr,
            n_i32,
            &mut work,
            -1,
            &mut rwork,
            &mut info,
        );
    }
    if info != 0 {
        return Err(eig_internal_error(format!(
            "eig: LAPACK ZGGEV workspace query failed with info = {info}"
        )));
    }

    let lwork = work[0].re.max(1.0) as i32;
    work.resize(lwork as usize, lapack::c64::new(0.0, 0.0));
    unsafe {
        lapack::zggev(
            jobvl,
            jobvr,
            n_i32,
            &mut a_copy,
            n_i32,
            &mut b_copy,
            n_i32,
            &mut alpha,
            &mut beta,
            &mut vl,
            n_i32,
            &mut vr,
            n_i32,
            &mut work,
            lwork,
            &mut rwork,
            &mut info,
        );
    }
    if info < 0 {
        return Err(eig_internal_error(format!(
            "eig: LAPACK ZGGEV rejected argument {}",
            -info
        )));
    }
    if info > 0 {
        return Err(eig_internal_error(format!(
            "eig: LAPACK ZGGEV failed to converge with info = {info}"
        )));
    }

    let eigenvalues = DVector::from_iterator(
        n,
        alpha
            .iter()
            .zip(beta.iter())
            .map(|(alpha, beta)| lapack_ratio_to_eigenvalue(alpha, beta)),
    );
    let right = lapack_complex_to_matrix(&vr, n, n);
    let left = if require_left {
        let mut left = lapack_complex_to_matrix(&vl, n, n);
        normalize_generalized_left(&mut left, b, &right);
        Some(left)
    } else {
        None
    };

    build_eval(eigenvalues, right, left, options)
}

#[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
fn matrix_to_lapack_complex(matrix: &DMatrix<Complex64>) -> Vec<lapack::c64> {
    matrix
        .iter()
        .map(|value| lapack::c64::new(value.re, value.im))
        .collect()
}

#[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
fn lapack_complex_to_matrix(
    values: &[lapack::c64],
    rows: usize,
    cols: usize,
) -> DMatrix<Complex64> {
    let data = values
        .iter()
        .map(|value| Complex64::new(value.re, value.im))
        .collect::<Vec<_>>();
    DMatrix::from_column_slice(rows, cols, &data)
}

#[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
fn lapack_ratio_to_eigenvalue(alpha: &lapack::c64, beta: &lapack::c64) -> Complex64 {
    let alpha = Complex64::new(alpha.re, alpha.im);
    let beta = Complex64::new(beta.re, beta.im);
    if beta.norm() <= REAL_EPS {
        if alpha.norm() <= REAL_EPS {
            Complex64::new(f64::NAN, f64::NAN)
        } else {
            Complex64::new(infinite_component(alpha.re), infinite_component(alpha.im))
        }
    } else {
        alpha / beta
    }
}

#[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
fn infinite_component(value: f64) -> f64 {
    if value == 0.0 {
        value
    } else {
        value.signum() * f64::INFINITY
    }
}

fn build_eval(
    eigenvalues: DVector<Complex64>,
    right: DMatrix<Complex64>,
    left: Option<DMatrix<Complex64>>,
    options: EigOptions,
) -> BuiltinResult<EigEval> {
    let eigenvalue_value = vector_to_value(&eigenvalues)?;
    let diag_value = diag_matrix_value(&eigenvalues)?;
    let right_value = matrix_to_value(&right)?;
    let left_value = left
        .map(|left_matrix| matrix_to_value(&left_matrix))
        .transpose()?;

    let spectral_output = if options.vector_output {
        eigenvalue_value.clone()
    } else {
        diag_value.clone()
    };

    Ok(EigEval {
        eigenvalues: eigenvalue_value,
        diagonal_matrix: diag_value,
        diagonal_output: spectral_output,
        right: right_value,
        left: left_value,
    })
}

fn maybe_balance(matrix: &DMatrix<Complex64>, _balance: bool) -> DMatrix<Complex64> {
    matrix.clone()
}

fn schur_eigendecompose(
    matrix: &DMatrix<Complex64>,
) -> BuiltinResult<(DVector<Complex64>, DMatrix<Complex64>)> {
    let n = matrix.nrows();
    if n == 0 {
        return Ok((DVector::from(vec![]), DMatrix::zeros(0, 0)));
    }
    let schur = Schur::new(matrix.clone());
    let (q, t) = schur.unpack();
    let diag = t.diagonal().clone_owned();
    let mut eigenvectors = DMatrix::<Complex64>::zeros(n, n);

    for idx in 0..n {
        let lambda = diag[idx];
        let mut z = DVector::<Complex64>::zeros(n);
        let mut success = false;

        for attempt in 0..2 {
            for k in (0..n).rev() {
                let mut sum = Complex64::new(0.0, 0.0);
                for j in (k + 1)..n {
                    sum += t[(k, j)] * z[j];
                }
                let coeff = t[(k, k)] - lambda;
                if coeff.norm() <= REAL_EPS {
                    if sum.norm() <= REAL_EPS {
                        if attempt == 0 && k == idx {
                            z[k] = Complex64::new(1.0, 0.0);
                        }
                    } else {
                        z[k] = -sum / Complex64::new(REAL_EPS, 0.0);
                    }
                } else {
                    z[k] = -sum / coeff;
                }
            }
            let norm = z.norm();
            if norm > REAL_EPS {
                let scale = Complex64::new(norm, 0.0);
                z /= scale;
                success = true;
                break;
            } else {
                z.fill(Complex64::new(0.0, 0.0));
                z[idx] = Complex64::new(1.0, 0.0);
            }
        }
        if !success {
            z.fill(Complex64::new(0.0, 0.0));
            z[idx] = Complex64::new(1.0, 0.0);
        }
        let vec = q.clone() * z;
        let norm = vec.norm();
        let final_vec = if norm > REAL_EPS {
            vec / Complex64::new(norm, 0.0)
        } else {
            vec
        };
        eigenvectors.set_column(idx, &final_vec);
    }

    Ok((diag, eigenvectors))
}

fn compute_left_vectors(
    matrix: &DMatrix<Complex64>,
    right: &DMatrix<Complex64>,
    eigenvalues: &DVector<Complex64>,
) -> Option<DMatrix<Complex64>> {
    if let Some(inv) = right.clone().try_inverse() {
        let mut left = inv.adjoint();
        normalize_left(&mut left, right);
        return Some(left);
    }

    let (left_vals, left_vecs) = schur_eigendecompose(&matrix.adjoint()).ok()?;
    let n = eigenvalues.len();
    let mut left = DMatrix::<Complex64>::zeros(n, n);
    let mut used = vec![false; n];

    for i in 0..n {
        let target = eigenvalues[i].conj();
        let mut best_idx = None;
        let mut best_err = f64::MAX;
        for j in 0..n {
            if used[j] {
                continue;
            }
            let err = (left_vals[j] - target).norm();
            if err < best_err {
                best_err = err;
                best_idx = Some(j);
            }
        }
        let idx = best_idx.unwrap_or(i);
        used[idx] = true;
        left.set_column(i, &left_vecs.column(idx));
    }

    normalize_left(&mut left, right);
    Some(left)
}

fn normalize_left(left: &mut DMatrix<Complex64>, right: &DMatrix<Complex64>) {
    for i in 0..right.ncols() {
        let dot = left.column(i).dot(&right.column(i));
        let finite = dot.re.is_finite() && dot.im.is_finite();
        if dot.norm() > REAL_EPS && finite {
            let scale = dot.conj();
            for r in 0..left.nrows() {
                left[(r, i)] /= scale;
            }
        }
    }
}

fn normalize_generalized_left(
    left: &mut DMatrix<Complex64>,
    b: &DMatrix<Complex64>,
    right: &DMatrix<Complex64>,
) {
    let b_right = b * right;
    for i in 0..right.ncols() {
        let dot = left.column(i).dot(&b_right.column(i));
        let finite = dot.re.is_finite() && dot.im.is_finite();
        if dot.norm() > REAL_EPS && finite {
            let scale = dot.conj();
            for r in 0..left.nrows() {
                left[(r, i)] /= scale;
            }
        }
    }
}

async fn value_to_complex_matrix(value: Value) -> BuiltinResult<DMatrix<Complex64>> {
    match value {
        Value::Tensor(tensor) => tensor_to_matrix(&tensor),
        Value::ComplexTensor(ct) => complex_tensor_to_matrix(&ct),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(&logical)
                .map_err(|err| eig_internal_error(format!("eig: {err}")))?;
            tensor_to_matrix(&tensor)
        }
        Value::Num(n) => Ok(DMatrix::from_element(1, 1, Complex64::new(n, 0.0))),
        Value::Int(i) => Ok(DMatrix::from_element(1, 1, Complex64::new(i.to_f64(), 0.0))),
        Value::Bool(b) => Ok(DMatrix::from_element(
            1,
            1,
            Complex64::new(if b { 1.0 } else { 0.0 }, 0.0),
        )),
        Value::Complex(re, im) => Ok(DMatrix::from_element(1, 1, Complex64::new(re, im))),
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(with_eig_context)?;
            tensor_to_matrix(&tensor)
        }
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => Err(eig_invalid_input(
            "eig: input must be numeric or logical; convert character data with double() first",
        )),
        other => Err(eig_invalid_input(format!(
            "eig: unsupported input type {other:?}; expected numeric or logical values"
        ))),
    }
}

fn tensor_to_matrix(tensor: &Tensor) -> BuiltinResult<DMatrix<Complex64>> {
    if tensor.shape.len() > 2 {
        return Err(eig_invalid_input("eig: input must be 2-D"));
    }
    let rows = tensor.rows();
    let cols = tensor.cols();
    let values = tensor::tensor_values_f64_cow(tensor);
    let mut data = Vec::with_capacity(values.len());
    for &value in values.iter() {
        data.push(Complex64::new(value, 0.0));
    }
    Ok(DMatrix::from_column_slice(rows, cols, &data))
}

fn complex_tensor_to_matrix(tensor: &ComplexTensor) -> BuiltinResult<DMatrix<Complex64>> {
    if tensor.shape.len() > 2 {
        return Err(eig_invalid_input("eig: input must be 2-D"));
    }
    let rows = tensor.rows;
    let cols = tensor.cols;
    let mut data = Vec::with_capacity(tensor.materialize_f64().len());
    for &(re, im) in &tensor.materialize_f64() {
        data.push(Complex64::new(re, im));
    }
    Ok(DMatrix::from_column_slice(rows, cols, &data))
}

fn vector_to_value(values: &DVector<Complex64>) -> BuiltinResult<Value> {
    if values.is_empty() {
        let tensor = Tensor::new(Vec::new(), vec![0, 0])
            .map_err(|e| eig_internal_error(format!("eig: {e}")))?;
        return Ok(Value::Tensor(tensor));
    }
    if is_all_real(values.iter().copied()) {
        let mut data = Vec::with_capacity(values.len());
        for value in values.iter() {
            data.push(value.re);
        }
        let tensor = Tensor::new(data, vec![values.len(), 1])
            .map_err(|e| eig_internal_error(format!("eig: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    } else {
        let mut data = Vec::with_capacity(values.len());
        for value in values.iter() {
            data.push((value.re, value.im));
        }
        let tensor = ComplexTensor::new(data, vec![values.len(), 1])
            .map_err(|e| eig_internal_error(format!("eig: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn diag_matrix_value(values: &DVector<Complex64>) -> BuiltinResult<Value> {
    if values.is_empty() {
        let tensor = Tensor::new(Vec::new(), vec![0, 0])
            .map_err(|e| eig_internal_error(format!("eig: {e}")))?;
        return Ok(Value::Tensor(tensor));
    }
    let size = values.len();
    if is_all_real(values.iter().copied()) {
        let mut data = vec![0.0f64; size * size];
        for i in 0..size {
            data[i + i * size] = values[i].re;
        }
        let tensor = Tensor::new(data, vec![size, size])
            .map_err(|e| eig_internal_error(format!("eig: {e}")))?;
        Ok(Value::Tensor(tensor))
    } else {
        let mut data = vec![(0.0f64, 0.0f64); size * size];
        for i in 0..size {
            data[i + i * size] = (values[i].re, values[i].im);
        }
        let tensor = ComplexTensor::new(data, vec![size, size])
            .map_err(|e| eig_internal_error(format!("eig: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn matrix_to_value(matrix: &DMatrix<Complex64>) -> BuiltinResult<Value> {
    if is_all_real(matrix.iter().copied()) {
        let mut data = Vec::with_capacity(matrix.len());
        for value in matrix.iter() {
            data.push(value.re);
        }
        let tensor = Tensor::new(data, vec![matrix.nrows(), matrix.ncols()])
            .map_err(|e| eig_internal_error(format!("eig: {e}")))?;
        Ok(Value::Tensor(tensor))
    } else {
        let mut data = Vec::with_capacity(matrix.len());
        for value in matrix.iter() {
            data.push((value.re, value.im));
        }
        let tensor = ComplexTensor::new(data, vec![matrix.nrows(), matrix.ncols()])
            .map_err(|e| eig_internal_error(format!("eig: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn is_all_real<I>(iter: I) -> bool
where
    I: IntoIterator<Item = Complex64>,
{
    iter.into_iter()
        .all(|value| value.im.is_nan() || value.im.abs() <= REAL_EPS)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{
        builtin_function_by_name, IntValue, IntegerStorage, ResolveContext, Type,
    };

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
    }

    fn matrix_from_value(value: Value) -> DMatrix<Complex64> {
        match value {
            Value::Tensor(t) => tensor_to_matrix(&t).expect("matrix"),
            Value::ComplexTensor(ct) => complex_tensor_to_matrix(&ct).expect("matrix"),
            other => panic!("expected matrix value, got {other:?}"),
        }
    }

    #[test]
    fn eig_type_returns_eigenvalue_vector() {
        let out = eig_type(
            &[Type::Tensor {
                shape: Some(vec![Some(3), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(1)])
            }
        );
    }

    #[test]
    fn eig_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = EIG_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"d = eig(A)"));
        assert!(labels.contains(&"d = eig(A, options...)"));
        assert!(labels.contains(&"[V, D] = eig(A)"));
        assert!(labels.contains(&"[V, D] = eig(A, options...)"));
        assert!(labels.contains(&"[V, D, W] = eig(A)"));
        assert!(labels.contains(&"[V, D, W] = eig(A, options...)"));
        assert!(labels.contains(&"d = eig(A, B)"));
        assert!(labels.contains(&"d = eig(A, B, options...)"));
        assert!(labels.contains(&"[V, D] = eig(A, B)"));
        assert!(labels.contains(&"[V, D] = eig(A, B, options...)"));
        assert!(labels.contains(&"[V, D, W] = eig(A, B)"));
        assert!(labels.contains(&"[V, D, W] = eig(A, B, options...)"));
    }

    #[test]
    fn eig_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = EIG_DESCRIPTOR.errors.iter().map(|err| err.code).collect();
        assert!(codes.contains(&"RM.EIG.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.EIG.INVALID_INPUT"));
        assert!(codes.contains(&"RM.EIG.INTERNAL"));
    }

    #[test]
    fn eig_provider_results_require_owner_shape_precision_storage_and_unique_handles() {
        use runmat_accelerate_api::{HostTensorView, ProviderEigResult, ProviderPrecision};

        test_support::with_test_provider(|provider| {
            let upload = |data: &[f64], shape: &[usize]| {
                let handle = provider
                    .upload(&HostTensorView { data, shape })
                    .expect("upload");
                runmat_accelerate_api::set_handle_precision(&handle, ProviderPrecision::F64);
                handle
            };
            let input = upload(&[1.0, 0.0, 0.0, 2.0], &[2, 2]);
            let eigenvalues = upload(&[1.0, 2.0], &[2, 1]);
            let diagonal = upload(&[1.0, 0.0, 0.0, 2.0], &[2, 2]);
            let right = upload(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
            let left = upload(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
            let valid = ProviderEigResult {
                eigenvalues: eigenvalues.clone(),
                diagonal: diagonal.clone(),
                right: right.clone(),
                left: Some(left.clone()),
            };
            assert!(valid_provider_eig_result(&valid, provider, &input, 2, true));

            let mut duplicate = valid.clone();
            duplicate.right = duplicate.diagonal.clone();
            assert!(!valid_provider_eig_result(
                &duplicate, provider, &input, 2, true
            ));

            let mut wrong_shape = valid.clone();
            wrong_shape.eigenvalues.shape = vec![1, 2];
            assert!(!valid_provider_eig_result(
                &wrong_shape,
                provider,
                &input,
                2,
                true
            ));

            runmat_accelerate_api::set_handle_logical(&right, true);
            assert!(!valid_provider_eig_result(
                &valid, provider, &input, 2, true
            ));
            runmat_accelerate_api::set_handle_logical(&right, false);

            runmat_accelerate_api::set_handle_class_name(&right, "uint8");
            assert!(!valid_provider_eig_result(
                &valid, provider, &input, 2, true
            ));
            runmat_accelerate_api::set_handle_class_name(&right, "double");
            assert!(valid_provider_eig_result(&valid, provider, &input, 2, true));
            runmat_accelerate_api::clear_handle_class_name(&right);

            let alias = upload(&[3.0, 4.0], &[2, 1]);
            let cleanup = ProviderEigResult {
                eigenvalues: input.clone(),
                diagonal: alias.clone(),
                right: alias.clone(),
                left: None,
            };
            free_provider_eig_result_unique(&cleanup, provider, &input);
            assert!(block_on(provider.download(&input)).is_ok());
            assert!(block_on(provider.download(&alias)).is_err());

            for handle in [&eigenvalues, &diagonal, &right, &left, &input] {
                provider.free(handle).ok();
            }
        });
    }

    #[test]
    fn eig_matrix_conversion_reads_typed_integer_storage_exactly() {
        let tensor = Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3, 4, 5, 6]), vec![3, 2])
            .expect("typed integer tensor");

        let matrix = tensor_to_matrix(&tensor).expect("matrix");
        assert_eq!(matrix.nrows(), 3);
        assert_eq!(matrix.ncols(), 2);
        assert_eq!(matrix[(0, 0)], Complex64::new(1.0, 0.0));
        assert_eq!(matrix[(1, 0)], Complex64::new(2.0, 0.0));
        assert_eq!(matrix[(2, 0)], Complex64::new(3.0, 0.0));
        assert_eq!(matrix[(0, 1)], Complex64::new(4.0, 0.0));
        assert_eq!(matrix[(1, 1)], Complex64::new(5.0, 0.0));
        assert_eq!(matrix[(2, 1)], Complex64::new(6.0, 0.0));
    }

    #[test]
    fn eig_nonfloating_coefficients_follow_strict_mode_and_exact_boundary() {
        let integer_matrix = || {
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![1, 0, 0, 2]), vec![2, 2])
                    .expect("integer matrix"),
            )
        };
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = evaluate(integer_matrix(), &[], false).expect_err("integer A must gate");
            assert_eq!(
                error.identifier(),
                EIG_NONFLOATING_COEFFICIENT_EXTENSION.error_identifier
            );
            let error = evaluate(Value::Num(2.0), &[Value::Int(IntValue::I16(1))], false)
                .expect_err("integer B must gate");
            assert_eq!(
                error.identifier(),
                EIG_NONFLOATING_COEFFICIENT_EXTENSION.error_identifier
            );
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            assert!(evaluate(integer_matrix(), &[], false).is_ok());
            let wide = Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1])
                    .expect("wide matrix"),
            );
            let error = evaluate(wide, &[], false).expect_err("wide integer must reject");
            assert!(error.message().contains("exactly representable as double"));
        }
    }

    #[test]
    fn eig_admits_all_eight_small_integer_classes_with_double_outputs() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let storages = [
            IntegerStorage::I8(vec![2]),
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I64(vec![2]),
            IntegerStorage::U8(vec![2]),
            IntegerStorage::U16(vec![2]),
            IntegerStorage::U32(vec![2]),
            IntegerStorage::U64(vec![2]),
        ];
        for storage in storages {
            let input = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).unwrap());
            let output = evaluate(input, &[], false).unwrap().eigenvalues();
            assert!(matches!(output, Value::Num(_) | Value::Tensor(_)));
        }
    }

    #[test]
    fn eig_classifies_resident_integer_coefficients_before_provider_access() {
        let handle = GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 9_300_042,
        };
        runmat_accelerate_api::set_handle_integer_type(
            &handle,
            runmat_accelerate_api::IntegerElementType::U16,
        );
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = evaluate(Value::GpuTensor(handle.clone()), &[], false)
            .expect_err("resident integer A must gate before provider access");
        runmat_accelerate_api::clear_handle_integer_type(&handle);
        assert_eq!(
            error.identifier(),
            EIG_NONFLOATING_COEFFICIENT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn eig_preserves_single_host_outputs_and_descriptor_policy() {
        let input = Value::Tensor(
            Tensor::from_f32(vec![2.0, 0.0, 0.0, 3.0], vec![2, 2]).expect("single matrix"),
        );
        let eval = evaluate(input, &[], false).expect("single eig");
        let Value::Tensor(values) = eval.eigenvalues() else {
            panic!("expected real single eigenvalues");
        };
        assert_eq!(values.numeric_dtype(), NumericDType::F32);
        let Value::Tensor(vectors) = eval.right() else {
            panic!("expected real single vectors");
        };
        assert_eq!(vectors.numeric_dtype(), NumericDType::F32);
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("registered eig");
        assert_eq!(builtin.extensions, &EIG_EXTENSIONS);
        assert_eq!(builtin.integer_capabilities.len(), 2);
    }

    fn column_vector_from_value(value: Value) -> Vec<Complex64> {
        match value {
            Value::Num(v) => vec![Complex64::new(v, 0.0)],
            Value::Int(v) => vec![Complex64::new(v.to_f64(), 0.0)],
            Value::Complex(re, im) => vec![Complex64::new(re, im)],
            Value::Tensor(t) => t
                .materialize_f64()
                .iter()
                .map(|&v| Complex64::new(v, 0.0))
                .collect::<Vec<_>>(),
            Value::ComplexTensor(ct) => ct
                .materialize_f64()
                .iter()
                .map(|&(re, im)| Complex64::new(re, im))
                .collect::<Vec<_>>(),
            other => panic!("expected tensor for eigenvalues, got {other:?}"),
        }
    }

    fn assert_matrix_close(a: &DMatrix<Complex64>, b: &DMatrix<Complex64>, tol: f64) {
        assert_eq!(a.nrows(), b.nrows(), "row mismatch");
        assert_eq!(a.ncols(), b.ncols(), "column mismatch");
        for r in 0..a.nrows() {
            for c in 0..a.ncols() {
                let diff = (a[(r, c)] - b[(r, c)]).norm();
                assert!(
                    diff <= tol,
                    "matrix mismatch at ({r},{c}): {} vs {} (diff {diff})",
                    a[(r, c)],
                    b[(r, c)]
                );
            }
        }
    }

    fn assert_values_close_unordered(mut actual: Vec<Complex64>, mut expected: Vec<Complex64>) {
        actual.sort_by(|a, b| a.re.total_cmp(&b.re));
        expected.sort_by(|a, b| a.re.total_cmp(&b.re));
        assert_eq!(actual.len(), expected.len());
        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            assert!(
                (lhs - rhs).norm() <= 1e-10,
                "expected eigenvalue {rhs}, got {lhs}"
            );
        }
    }

    #[cfg(feature = "wgpu")]
    fn assert_tensor_close(a: &Tensor, b: &Tensor, tol: f64) {
        assert_eq!(
            a.shape, b.shape,
            "shape mismatch: {:?} vs {:?}",
            a.shape, b.shape
        );
        for (idx, (lhs, rhs)) in a
            .materialize_f64()
            .iter()
            .zip(b.materialize_f64().iter())
            .enumerate()
        {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= tol,
                "tensor mismatch at index {}: {} vs {} (diff {diff})",
                idx,
                lhs,
                rhs
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_scalar_real() {
        let result = eig_builtin(Value::Num(5.0), Vec::new()).expect("eig");
        match result {
            Value::Num(v) => assert!((v - 5.0).abs() < 1e-12),
            other => panic!("expected scalar result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_two_outputs_reconstruct() {
        let tensor = Tensor::new(vec![0.0, -2.0, 1.0, -3.0], vec![2, 2]).unwrap();
        let eval = evaluate(Value::Tensor(tensor.clone()), &[], false).expect("evaluate");
        let v = matrix_from_value(eval.right());
        let d = matrix_from_value(eval.diagonal_matrix());
        let a = matrix_from_value(Value::Tensor(tensor));
        let recon = &v * &d * v.try_inverse().unwrap();
        assert_matrix_close(&a, &recon, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_three_outputs_biorthogonality() {
        let tensor = Tensor::new(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let eval = evaluate(Value::Tensor(tensor), &[], true).expect("evaluate");
        let v = matrix_from_value(eval.right());
        let d = matrix_from_value(eval.diagonal_matrix());
        let w = matrix_from_value(eval.left().expect("left eigenvectors"));
        let vw = w.adjoint() * &v;
        let identity = DMatrix::<Complex64>::identity(v.ncols(), v.ncols());
        assert_matrix_close(&vw, &identity, 1e-10);
        let tensor = Tensor::new(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let a = matrix_from_value(Value::Tensor(tensor));
        let lhs = w.adjoint() * &a;
        let rhs = &d * w.adjoint();
        assert_matrix_close(&lhs, &rhs, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_complex_matrix() {
        let data = vec![(1.0, 2.0), (0.0, 0.0), (2.0, -1.0), (0.0, -3.0)];
        let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let result = eig_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("eig");
        let values = column_vector_from_value(result);
        assert_eq!(values.len(), 2);
        assert!((values[0] - Complex64::new(1.0, 2.0)).norm() < 1e-10);
        assert!((values[1] - Complex64::new(0.0, -3.0)).norm() < 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_errors_on_non_square() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = eig_builtin(Value::Tensor(tensor), Vec::new()).unwrap_err();
        assert_eq!(err.identifier(), EIG_ERROR_INVALID_INPUT.identifier);
        let err = error_message(err);
        assert!(err.contains("square"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_accepts_nobalance_option() {
        let tensor = Tensor::new(vec![1.0, 1.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let base = eig_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("eig");
        let opt = eig_builtin(
            Value::Tensor(tensor),
            vec![Value::String("nobalance".into())],
        )
        .expect("eig with option");
        let base_vals = column_vector_from_value(base);
        let opt_vals = column_vector_from_value(opt);
        for (a, b) in base_vals.iter().zip(opt_vals.iter()) {
            assert!((a - b).norm() < 1e-10);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_vector_option_returns_column_vector() {
        let tensor = Tensor::new(vec![0.0, 1.0, -2.0, -3.0], vec![2, 2]).unwrap();
        let eval =
            evaluate(Value::Tensor(tensor), &[Value::from("vector")], false).expect("evaluate");
        match eval.diagonal() {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 1]);
            }
            other => panic!("expected vector second output, got {other:?}"),
        }
        // Matrix form stays available for reconstruction.
        let _ = matrix_from_value(eval.diagonal_matrix());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_vector_option_allows_left_eigenvectors() {
        let tensor = Tensor::new(vec![4.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let eval =
            evaluate(Value::Tensor(tensor), &[Value::from("vector")], true).expect("evaluate");
        match eval.diagonal() {
            Value::Tensor(t) => assert_eq!(t.shape, vec![2, 1]),
            other => panic!("expected vector second output, got {other:?}"),
        }
        let left = eval.left().expect("left eigenvectors");
        match left {
            Value::Tensor(t) => assert_eq!(t.shape, vec![2, 2]),
            Value::ComplexTensor(ct) => assert_eq!(ct.shape, vec![2, 2]),
            other => panic!("unexpected type for left eigenvectors: {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_generalized_identity_matches_standard() {
        let a = Tensor::new(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let result = eig_builtin(Value::Tensor(a), vec![Value::Tensor(b)]).expect("eig(A,B)");
        let values = column_vector_from_value(result);
        assert_values_close_unordered(
            values,
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_generalized_diagonal_pair() {
        let a = Tensor::new(vec![2.0, 0.0, 0.0, 9.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 3.0], vec![2, 2]).unwrap();
        let result = eig_builtin(Value::Tensor(a), vec![Value::Tensor(b)]).expect("eig(A,B)");
        let values = column_vector_from_value(result);
        assert_values_close_unordered(
            values,
            vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_generalized_two_outputs_reconstruct() {
        let a_tensor = Tensor::new(vec![4.0, 0.0, 1.0, 9.0], vec![2, 2]).unwrap();
        let b_tensor = Tensor::new(vec![2.0, 0.0, 0.0, 3.0], vec![2, 2]).unwrap();
        let eval = evaluate(
            Value::Tensor(a_tensor.clone()),
            &[Value::Tensor(b_tensor.clone())],
            false,
        )
        .expect("evaluate");
        let a = matrix_from_value(Value::Tensor(a_tensor));
        let b = matrix_from_value(Value::Tensor(b_tensor));
        let v = matrix_from_value(eval.right());
        let d = matrix_from_value(eval.diagonal_matrix());
        let lhs = &a * &v;
        let rhs = &b * &v * &d;
        assert_matrix_close(&lhs, &rhs, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_generalized_three_outputs_satisfy_left_equation() {
        let a_tensor = Tensor::new(vec![4.0, 0.0, 1.0, 9.0], vec![2, 2]).unwrap();
        let b_tensor = Tensor::new(vec![2.0, 0.0, 0.0, 3.0], vec![2, 2]).unwrap();
        let eval = evaluate(
            Value::Tensor(a_tensor.clone()),
            &[Value::Tensor(b_tensor.clone())],
            true,
        )
        .expect("evaluate");
        let a = matrix_from_value(Value::Tensor(a_tensor));
        let b = matrix_from_value(Value::Tensor(b_tensor));
        let v = matrix_from_value(eval.right());
        let d = matrix_from_value(eval.diagonal_matrix());
        let w = matrix_from_value(eval.left().expect("left eigenvectors"));

        let identity = DMatrix::<Complex64>::identity(v.ncols(), v.ncols());
        let gram = w.adjoint() * &b * &v;
        assert_matrix_close(&gram, &identity, 1e-10);

        let lhs = w.adjoint() * &a;
        let rhs = &d * w.adjoint() * &b;
        assert_matrix_close(&lhs, &rhs, 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_generalized_vector_option_returns_column_vector_second_output() {
        let a = Tensor::new(vec![2.0, 0.0, 0.0, 9.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 3.0], vec![2, 2]).unwrap();
        let eval = evaluate(
            Value::Tensor(a),
            &[Value::Tensor(b), Value::from("vector")],
            false,
        )
        .expect("evaluate");
        match eval.diagonal() {
            Value::Tensor(t) => assert_eq!(t.shape, vec![2, 1]),
            Value::ComplexTensor(ct) => assert_eq!(ct.shape, vec![2, 1]),
            other => panic!("expected vector second output, got {other:?}"),
        }
        let _ = matrix_from_value(eval.diagonal_matrix());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(not(all(feature = "blas-lapack", not(target_arch = "wasm32"))))]
    fn eig_generalized_singular_b_errors_with_qz_message() {
        let a = Tensor::new(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let err = evaluate(Value::Tensor(a), &[Value::Tensor(b)], false).unwrap_err();
        assert_eq!(err.identifier(), EIG_ERROR_INVALID_INPUT.identifier);
        let err = error_message(err);
        assert!(err.contains("singular B"));
        assert!(err.contains("QZ"));
    }

    #[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
    #[test]
    fn eig_generalized_singular_b_returns_infinite_eigenvalue_with_lapack() {
        let a = Tensor::new(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let result = eig_builtin(Value::Tensor(a), vec![Value::Tensor(b)]).expect("eig(A,B)");
        let values = column_vector_from_value(result);
        assert!(values.iter().any(|value| (value.re - 1.0).abs() < 1e-10));
        assert!(values.iter().any(|value| value.re.is_infinite()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_vector_option_gpu_fallback_restores_owner_residency() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 1.0, 0.0, 3.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let eval = evaluate(Value::GpuTensor(handle), &[Value::from("vector")], false)
                .expect("evaluate");
            match eval.diagonal() {
                Value::GpuTensor(handle) => assert_eq!(handle.shape, vec![2, 1]),
                other => panic!("unexpected eigenvalue output: {other:?}"),
            }
            assert!(matches!(eval.right(), Value::GpuTensor(_)));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_gpu_provider_fallback_restores_owner_residency() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 0.0, 0.0, 3.0], vec![2, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                eig_builtin(Value::GpuTensor(handle), vec![Value::from("nobalance")]).expect("eig");
            let tensor = test_support::gather(result).expect("gather owner-resident output");
            assert_eq!(tensor.materialize_f64(), vec![2.0, 3.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_generalized_scalar_pair() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let args = vec![Value::Int(IntValue::I32(3))];
        let result = evaluate(Value::Num(4.0), &args, false)
            .expect("scalar generalized eig")
            .eigenvalues();
        let values = column_vector_from_value(result);
        assert_eq!(values.len(), 1);
        assert!((values[0] - Complex64::new(4.0 / 3.0, 0.0)).norm() < 1e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eig_rejects_unknown_option_with_stable_identifier() {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = evaluate(Value::Tensor(tensor), &[Value::from("invalid")], false).unwrap_err();
        assert_eq!(err.identifier(), EIG_ERROR_INVALID_ARGUMENT.identifier);
        assert!(error_message(err).contains("unknown option"));
    }

    #[test]
    fn eig_invalid_input_identifier_is_stable_for_nd_arrays() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]).expect("tensor");
        let err = evaluate(Value::Tensor(tensor), &[], false).unwrap_err();
        assert_eq!(err.identifier(), EIG_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn eig_wgpu_matches_cpu_for_real_spectrum() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        let tensor = Tensor::new(vec![4.0, 2.0, 1.0, 3.0], vec![2, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");

        let gpu_eval = evaluate(Value::GpuTensor(handle), &[], true).expect("gpu evaluate");

        let eig_gpu_value = gpu_eval.eigenvalues();
        assert!(matches!(eig_gpu_value, Value::GpuTensor(_)));
        let diag_gpu_value = gpu_eval.diagonal_matrix();
        assert!(matches!(diag_gpu_value, Value::GpuTensor(_)));
        let right_gpu_value = gpu_eval.right();
        assert!(matches!(right_gpu_value, Value::GpuTensor(_)));
        let left_gpu_value = gpu_eval.left().expect("left eigenvectors");
        assert!(matches!(left_gpu_value, Value::GpuTensor(_)));

        let eig_gpu = test_support::gather(eig_gpu_value).expect("gather eigenvalues");
        let diag_gpu = test_support::gather(diag_gpu_value).expect("gather diagonal");
        let right_gpu = test_support::gather(right_gpu_value).expect("gather right vectors");
        let left_gpu = test_support::gather(left_gpu_value).expect("gather left vectors");

        let host_eval = evaluate(Value::Tensor(tensor.clone()), &[], true).expect("host evaluate");

        let eig_host = match host_eval.eigenvalues() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor eigenvalues, got {other:?}"),
        };
        let diag_host = match host_eval.diagonal_matrix() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor diagonal, got {other:?}"),
        };
        let right_host = match host_eval.right() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor right eigenvectors, got {other:?}"),
        };
        let left_host = match host_eval.left().expect("host left eigenvectors") {
            Value::Tensor(t) => t,
            other => panic!("expected tensor left eigenvectors, got {other:?}"),
        };

        assert_tensor_close(&eig_gpu, &eig_host, tol);
        assert_tensor_close(&diag_gpu, &diag_host, tol);
        assert_tensor_close(&right_gpu, &right_host, tol);
        assert_tensor_close(&left_gpu, &left_host, tol);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn eig_wgpu_nobalance_fallback_restores_residency() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![1.0, 1.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let eval = evaluate(Value::GpuTensor(handle), &[Value::from("nobalance")], false)
            .expect("evaluate");
        assert!(matches!(eval.eigenvalues(), Value::GpuTensor(_)));
    }

    fn eig_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::eig_builtin(value, rest))
    }

    fn evaluate(value: Value, args: &[Value], require_left: bool) -> BuiltinResult<EigEval> {
        block_on(super::evaluate(value, args, require_left))
    }
}
