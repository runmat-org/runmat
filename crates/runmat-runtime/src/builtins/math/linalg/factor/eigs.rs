//! MATLAB-compatible `eigs` builtin for subset eigenvalue queries.
//!
//! RunMat currently computes the exact dense/generalized decomposition through
//! the shared `eig` implementation, then applies MATLAB-compatible subset
//! selection. Sparse and GPU inputs are accepted by gathering/materialising to
//! host first; native iterative sparse/GPU eigensolver hooks belong in the
//! acceleration layer once the provider API exposes them.

use std::cmp::Ordering;

use num_complex::Complex64;
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
use runmat_value::{
    ComplexTensor, IntValue, NumericDType, NumericScalar, StructValue, Tensor, Value,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::eig;

const NAME: &str = "eigs";
const DEFAULT_K: usize = 6;

const EIGS_NONFLOATING_MATRIX_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "eigs-nonfloating-matrix",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "eigs with a non-single/non-double coefficient matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:EigsNonfloatingMatrixExtension"),
};

const EIGS_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "eigs-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "eigs with a gpuArray argument is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:EigsGpuInputExtension"),
};

const EIGS_INTEGER_SIGMA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "eigs-integer-sigma",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "eigs with a typed-integer sigma is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:EigsIntegerSigmaExtension"),
};

const EIGS_INTEGER_START_VECTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "eigs-integer-start-vector",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "eigs with a typed-integer StartVector is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EigsIntegerStartVectorExtension"),
    };

pub const EIGS_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    EIGS_NONFLOATING_MATRIX_EXTENSION,
    EIGS_GPU_INPUT_EXTENSION,
    EIGS_INTEGER_SIGMA_EXTENSION,
    EIGS_INTEGER_START_VECTOR_EXTENSION,
];

const EIGS_INTEGER_COEFFICIENT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Primary coefficient matrix.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Optional generalized coefficient matrix.",
    },
];
const EIGS_INTEGER_K_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "k",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Positive structural count parsed exactly without a binary64 mirror.",
}];
const EIGS_INTEGER_SIGMA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sigma",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat-only numeric shift crossing an exact binary64 boundary.",
    }];
const EIGS_INTEGER_START_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "StartVector",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat-only initial-vector class; current dense fallback validates shape but does not consume iterative values.",
    }];

pub const EIGS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "eigs(A, B, ...) with a typed-integer coefficient",
        inputs: &EIGS_INTEGER_COEFFICIENT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only coefficients are checked before dense binary64 fallback.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "eigs(A, k, ...) with typed-integer k",
        inputs: &EIGS_INTEGER_K_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Documented exact structural control.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "eigs(A, k, sigma, ...) with typed-integer sigma",
        inputs: &EIGS_INTEGER_SIGMA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "RunMat-only shift extension.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "eigs(..., 'StartVector', v) with typed-integer v",
        inputs: &EIGS_INTEGER_START_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "RunMat-only iterative-control class extension.",
    },
];

const OUTPUT_D: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "d",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Selected eigenvalues as a column vector.",
}];

const OUTPUT_VD: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Selected right eigenvectors.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Diagonal matrix of selected eigenvalues.",
    },
];

const OUTPUT_VDFLAG: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Selected right eigenvectors.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Diagonal matrix of selected eigenvalues.",
    },
    BuiltinParamDescriptor {
        name: "flag",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Convergence flag; exact host fallback returns 0.",
    },
];

const INPUT_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square numeric, sparse, logical, complex, or gpuArray matrix.",
}];

const INPUT_A_REST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Square numeric, sparse, logical, complex, or gpuArray matrix.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional B, k, sigma, opts, and name-value arguments.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "d = eigs(A)",
        inputs: &INPUT_A,
        outputs: &OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "d = eigs(A, k)",
        inputs: &INPUT_A_REST,
        outputs: &OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "d = eigs(A, k, sigma)",
        inputs: &INPUT_A_REST,
        outputs: &OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "d = eigs(A, k, sigma, opts)",
        inputs: &INPUT_A_REST,
        outputs: &OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "d = eigs(A, k, sigma, Name, Value)",
        inputs: &INPUT_A_REST,
        outputs: &OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "d = eigs(A, B, ...)",
        inputs: &INPUT_A_REST,
        outputs: &OUTPUT_D,
    },
    BuiltinSignatureDescriptor {
        label: "[V, D] = eigs(...)",
        inputs: &INPUT_A_REST,
        outputs: &OUTPUT_VD,
    },
    BuiltinSignatureDescriptor {
        label: "[V, D, flag] = eigs(...)",
        inputs: &INPUT_A_REST,
        outputs: &OUTPUT_VDFLAG,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EIGS.INVALID_ARGUMENT",
    identifier: Some("RunMat:eigs:InvalidArgument"),
    when: "Inputs, options, or requested output count are invalid.",
    message: "eigs: invalid argument",
};

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EIGS.INVALID_INPUT",
    identifier: Some("RunMat:eigs:InvalidInput"),
    when: "The matrix input is unsupported or has invalid shape.",
    message: "eigs: input matrix must be square",
};

const ERROR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EIGS.UNSUPPORTED",
    identifier: Some("RunMat:eigs:UnsupportedInput"),
    when: "Function-handle or provider-only eigs behavior is requested.",
    message: "eigs: unsupported input form",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EIGS.INTERNAL",
    identifier: Some("RunMat:eigs:Internal"),
    when: "Runtime cannot materialize selected eigen outputs.",
    message: "eigs: internal runtime failure",
};

const ERRORS: [BuiltinErrorDescriptor; 4] = [
    ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_INPUT,
    ERROR_UNSUPPORTED,
    ERROR_INTERNAL,
];

pub const EIGS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::factor::eigs")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("eigs-subset"),
    supported_precisions: &[ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Subset eigenvalue selection is currently a host residency sink: gpuArray inputs gather before exact dense fallback and selected outputs are returned on the host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::linalg::factor::eigs")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "eigs executes eagerly and terminates fusion planning.",
};

#[runtime_builtin(
    name = "eigs",
    category = "math/linalg/factor",
    summary = "Compute selected eigenvalues and eigenvectors.",
    keywords = "eigs,eigenvalues,eigenvectors,sparse,linalg,largestabs,smallestabs",
    accel = "cpu",
    type_resolver(eigs_type),
    descriptor(crate::builtins::math::linalg::factor::eigs::EIGS_DESCRIPTOR),
    extensions(crate::builtins::math::linalg::factor::eigs::EIGS_EXTENSIONS),
    integer_capabilities(crate::builtins::math::linalg::factor::eigs::EIGS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::linalg::factor::eigs"
)]
async fn eigs_builtin(a: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if crate::value_contains_gpu(&a) || rest.iter().any(crate::value_contains_gpu) {
        crate::compatibility::ensure_builtin_extension_enabled(&EIGS_GPU_INPUT_EXTENSION, NAME)?;
    }
    ensure_eigs_coefficient_class_enabled(&a)?;
    preflight_eigs_extensions(&rest)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&a, "eigs")?;
    for value in &rest {
        crate::builtins::common::validation::reject_typed_complex_integer(value, "eigs")?;
    }
    let request = Request::parse(a, rest).await?;
    let eval = compute_subset(request).await?;

    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![eval.values])),
        Some(2) => Ok(Value::OutputList(vec![eval.vectors, eval.diagonal])),
        Some(3) => Ok(Value::OutputList(vec![
            eval.vectors,
            eval.diagonal,
            Value::Num(0.0),
        ])),
        Some(_) => Err(eigs_error(&ERROR_INVALID_ARGUMENT)),
        None => Ok(eval.values),
    }
}

fn ensure_eigs_coefficient_class_enabled(value: &Value) -> BuiltinResult<()> {
    if is_nonfloating_supported_coefficient_matrix(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &EIGS_NONFLOATING_MATRIX_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

fn preflight_eigs_extensions(rest: &[Value]) -> BuiltinResult<()> {
    let mut cursor = 0usize;
    if let Some(candidate) = rest.first() {
        if is_empty_matrix(candidate) {
            cursor = 1;
        } else if is_generalized_b(candidate) {
            ensure_eigs_coefficient_class_enabled(candidate)?;
            cursor = 1;
        }
    }
    if cursor < rest.len() && parse_integer_scalar(&rest[cursor])?.is_some() {
        cursor += 1;
    }
    if cursor < rest.len() && (is_text(&rest[cursor]) || is_numeric_sigma_value(&rest[cursor])) {
        if is_typed_integer_value(&rest[cursor]) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &EIGS_INTEGER_SIGMA_EXTENSION,
                NAME,
            )?;
        }
        cursor += 1;
    }
    if cursor < rest.len() {
        preflight_start_vector_extension(&rest[cursor..])?;
    }
    Ok(())
}

fn preflight_start_vector_extension(args: &[Value]) -> BuiltinResult<()> {
    if args.len() == 1 {
        if let Value::Struct(options) = &args[0] {
            for (name, value) in &options.fields {
                if matches!(
                    name.trim().to_ascii_lowercase().as_str(),
                    "startvector" | "v0"
                ) && is_typed_integer_value(value)
                {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &EIGS_INTEGER_START_VECTOR_EXTENSION,
                        NAME,
                    )?;
                }
            }
            return Ok(());
        }
    }
    let mut idx = 0usize;
    while idx + 1 < args.len() {
        if text_value(&args[idx]).is_some_and(|name| {
            matches!(
                name.trim().to_ascii_lowercase().as_str(),
                "startvector" | "v0"
            )
        }) && is_typed_integer_value(&args[idx + 1])
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &EIGS_INTEGER_START_VECTOR_EXTENSION,
                NAME,
            )?;
        }
        idx += 2;
    }
    Ok(())
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::SparseTensor(sparse) if sparse.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_numeric_sigma_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Complex(_, _)
            | Value::Tensor(_)
            | Value::ComplexTensor(_)
            | Value::LogicalArray(_)
            | Value::Bool(_)
    )
}

fn is_nonfloating_supported_coefficient_matrix(value: &Value) -> bool {
    match value {
        Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::Tensor(tensor) => !matches!(
            tensor.numeric_dtype(),
            NumericDType::F64 | NumericDType::F32
        ),
        Value::SparseTensor(sparse) => {
            sparse.is_logical()
                || sparse
                    .numeric_dtype()
                    .is_some_and(|dtype| !matches!(dtype, NumericDType::F64 | NumericDType::F32))
        }
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_is_logical(handle)
                || runmat_accelerate_api::handle_integer_type(handle).is_some()
        }
        Value::ComplexTensor(_) => false,
        _ => false,
    }
}

fn eigs_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::tensor(),
    }
}

#[derive(Clone, Copy)]
enum Sigma {
    LargestAbs,
    SmallestAbs,
    LargestReal,
    SmallestReal,
    BothEndsReal,
    LargestImag,
    SmallestImag,
    BothEndsImag,
    Near(Complex64),
}

impl Default for Sigma {
    fn default() -> Self {
        Self::LargestAbs
    }
}

struct Request {
    a: Value,
    b: Option<Value>,
    k: usize,
    sigma: Sigma,
}

#[derive(Default)]
struct ParsedOptions {
    is_cholesky: bool,
    cholesky_permutation: Option<Vec<usize>>,
    is_function_symmetric: bool,
    is_symmetric_definite: bool,
}

struct SubsetEval {
    values: Value,
    vectors: Value,
    diagonal: Value,
}

impl Request {
    async fn parse(a: Value, rest: Vec<Value>) -> BuiltinResult<Self> {
        if matches!(
            a,
            Value::FunctionHandle(_)
                | Value::ExternalFunctionHandle(_)
                | Value::MethodFunctionHandle(_)
                | Value::BoundFunctionHandle { .. }
                | Value::Closure(_)
        ) {
            return Err(error_with_detail(
                "eigs: function-handle matrix forms require iterative operator support",
                &ERROR_UNSUPPORTED,
            ));
        }

        let mut a = gather_matrix_value(a).await?;
        ensure_exact_eigs_integer_boundary(&a, "A")?;
        let n = matrix_order(&a)?;
        let mut args = gather_args(rest).await?;
        let mut cursor = 0usize;
        let mut b = None;
        let mut k = DEFAULT_K.min(n);
        let mut sigma = Sigma::LargestAbs;
        let mut options = ParsedOptions::default();

        if let Some(candidate) = args.first() {
            if is_empty_matrix(candidate) {
                cursor += 1;
            } else if is_generalized_b(candidate) {
                ensure_eigs_coefficient_class_enabled(candidate)?;
                let candidate = matrix_host_value(args.remove(0))?;
                ensure_exact_eigs_integer_boundary(&candidate, "B")?;
                b = Some(candidate);
                cursor = 0;
            }
        }

        if cursor < args.len() {
            if let Some(parsed_k) = parse_k(&args[cursor])? {
                k = parsed_k.min(n);
                cursor += 1;
            }
        }

        if cursor < args.len() {
            if let Some(parsed_sigma) = parse_sigma(&args[cursor])? {
                sigma = parsed_sigma;
                cursor += 1;
            }
        }

        if cursor < args.len() {
            parse_options(&args[cursor..], n, k, &mut options)?;
        }

        validate_option_combination(sigma, &options)?;

        if let Some(b_value) = b.as_mut() {
            let b_order = matrix_order(b_value)?;
            if b_order != n {
                return Err(error_with_detail(
                    "eigs: A and B must be the same size",
                    &ERROR_INVALID_INPUT,
                ));
            }
            if options.is_cholesky {
                *b_value =
                    cholesky_factor_to_b(b_value.clone(), options.cholesky_permutation.as_deref())?;
            }
        } else if options.is_cholesky || options.cholesky_permutation.is_some() {
            return Err(error_with_detail(
                "eigs: Cholesky options require a generalized B matrix",
                &ERROR_INVALID_ARGUMENT,
            ));
        }

        a = matrix_host_value(a)?;

        Ok(Self { a, b, k, sigma })
    }
}

async fn gather_args(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for arg in args {
        out.push(
            gpu_helpers::gather_value_async(&arg)
                .await
                .map_err(with_context)?,
        );
    }
    Ok(out)
}

async fn gather_matrix_value(value: Value) -> BuiltinResult<Value> {
    let value = gpu_helpers::gather_value_async(&value)
        .await
        .map_err(with_context)?;
    matrix_host_value(value)
}

fn matrix_host_value(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::SparseTensor(sparse) => sparse
            .to_dense()
            .map(Value::Tensor)
            .map_err(|err| error_with_detail(format!("eigs: {err}"), &ERROR_INTERNAL)),
        other => Ok(other),
    }
}

fn matrix_order(value: &Value) -> BuiltinResult<usize> {
    let (rows, cols) = match value {
        Value::Tensor(tensor) => (tensor.rows(), tensor.cols()),
        Value::ComplexTensor(tensor) => (tensor.rows, tensor.cols),
        Value::LogicalArray(logical) => matrix_shape_from_slice(&logical.shape),
        Value::SparseTensor(sparse) => (sparse.rows, sparse.cols),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => (1, 1),
        other => {
            return Err(error_with_detail(
                format!("eigs: unsupported matrix input {other:?}"),
                &ERROR_INVALID_INPUT,
            ));
        }
    };
    if rows != cols {
        return Err(error_with_detail(
            ERROR_INVALID_INPUT.message,
            &ERROR_INVALID_INPUT,
        ));
    }
    Ok(cols)
}

fn matrix_shape_from_slice(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (shape[0], 1),
        _ => (shape[0], shape[1]),
    }
}

fn is_empty_matrix(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor_element_len(tensor) == 0,
        Value::ComplexTensor(tensor) => complex_tensor_element_len(tensor) == 0,
        Value::LogicalArray(logical) => logical.data.is_empty(),
        Value::SparseTensor(sparse) => sparse.rows == 0 || sparse.cols == 0,
        _ => false,
    }
}

fn is_generalized_b(value: &Value) -> bool {
    if is_text(value) {
        return false;
    }
    if parse_integer_scalar(value).ok().flatten().is_some() {
        return false;
    }
    matches!(
        value,
        Value::Tensor(_)
            | Value::ComplexTensor(_)
            | Value::LogicalArray(_)
            | Value::SparseTensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Complex(_, _)
    )
}

fn parse_k(value: &Value) -> BuiltinResult<Option<usize>> {
    parse_integer_scalar(value).map(|parsed| parsed.map(|k| k as usize))
}

fn parse_integer_scalar(value: &Value) -> BuiltinResult<Option<i64>> {
    if let Some(value) = exact_integer_scalar(value) {
        let Some(raw) = value.try_to_i64() else {
            return Err(error_with_detail(
                "eigs: k is too large for this platform",
                &ERROR_INVALID_ARGUMENT,
            ));
        };
        if raw <= 0 {
            return Err(error_with_detail(
                "eigs: k must be a positive integer scalar",
                &ERROR_INVALID_ARGUMENT,
            ));
        }
        return Ok(Some(raw));
    }
    let raw = match value {
        Value::Num(n) => *n,
        Value::Tensor(tensor) if tensor_element_len(tensor) == 1 => scalar_tensor_f64(tensor),
        Value::LogicalArray(logical) if logical.len() == 1 => {
            if logical.data[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        Value::Bool(flag) => {
            if *flag {
                1.0
            } else {
                0.0
            }
        }
        _ => return Ok(None),
    };
    if !raw.is_finite() || raw <= 0.0 || raw.fract() != 0.0 {
        return Err(error_with_detail(
            "eigs: k must be a positive integer scalar",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    if raw >= i64::MAX as f64 {
        return Err(error_with_detail(
            "eigs: k is too large for this platform",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok(Some(raw as i64))
}

fn parse_sigma(value: &Value) -> BuiltinResult<Option<Sigma>> {
    if let Some(text) = text_value(value) {
        return Ok(Some(parse_sigma_text(&text)?));
    }
    match value {
        Value::Num(n) => Ok(Some(Sigma::Near(Complex64::new(*n, 0.0)))),
        Value::Int(i) => {
            ensure_exact_eigs_integer_boundary(value, "sigma")?;
            Ok(Some(Sigma::Near(Complex64::new(i.to_f64(), 0.0))))
        }
        Value::Complex(re, im) => Ok(Some(Sigma::Near(Complex64::new(*re, *im)))),
        Value::Tensor(tensor) if tensor_element_len(tensor) == 1 => {
            ensure_exact_eigs_integer_boundary(value, "sigma")?;
            Ok(Some(Sigma::Near(Complex64::new(
                scalar_tensor_f64(tensor),
                0.0,
            ))))
        }
        Value::ComplexTensor(tensor) if tensor::is_scalar_complex_tensor(tensor) => {
            let value = tensor::complex_tensor_value_complex64(tensor, 0);
            Ok(Some(Sigma::Near(value)))
        }
        Value::LogicalArray(logical) if logical.len() == 1 => Ok(Some(Sigma::Near(
            Complex64::new(if logical.data[0] != 0 { 1.0 } else { 0.0 }, 0.0),
        ))),
        Value::Bool(flag) => Ok(Some(Sigma::Near(Complex64::new(
            if *flag { 1.0 } else { 0.0 },
            0.0,
        )))),
        _ => Ok(None),
    }
}

fn parse_sigma_text(text: &str) -> BuiltinResult<Sigma> {
    match text.trim().to_ascii_lowercase().as_str() {
        "largestabs" | "lm" => Ok(Sigma::LargestAbs),
        "smallestabs" | "sm" => Ok(Sigma::SmallestAbs),
        "largestreal" | "lr" | "la" => Ok(Sigma::LargestReal),
        "smallestreal" | "sr" | "sa" => Ok(Sigma::SmallestReal),
        "bothendsreal" | "be" => Ok(Sigma::BothEndsReal),
        "largestimag" | "li" => Ok(Sigma::LargestImag),
        "smallestimag" | "si" => Ok(Sigma::SmallestImag),
        "bothendsimag" => Ok(Sigma::BothEndsImag),
        other => Err(error_with_detail(
            format!("eigs: unsupported sigma '{other}'"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn parse_options(
    args: &[Value],
    n: usize,
    k: usize,
    options: &mut ParsedOptions,
) -> BuiltinResult<()> {
    if args.len() == 1 {
        if let Value::Struct(opts) = &args[0] {
            return parse_options_struct(opts, n, k, options);
        }
    }
    if args.is_empty() {
        return Ok(());
    }
    if !args.len().is_multiple_of(2) {
        return Err(error_with_detail(
            "eigs: name-value options must appear in pairs",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    let mut idx = 0usize;
    while idx < args.len() {
        let name = text_value(&args[idx]).ok_or_else(|| {
            error_with_detail(
                "eigs: option names must be character vectors or string scalars",
                &ERROR_INVALID_ARGUMENT,
            )
        })?;
        parse_named_option(&name, &args[idx + 1], n, k, options)?;
        idx += 2;
    }
    Ok(())
}

fn parse_options_struct(
    opts: &StructValue,
    n: usize,
    k: usize,
    options: &mut ParsedOptions,
) -> BuiltinResult<()> {
    for (field, value) in &opts.fields {
        parse_named_option(field, value, n, k, options)?;
    }
    Ok(())
}

fn parse_named_option(
    name: &str,
    value: &Value,
    n: usize,
    k: usize,
    options: &mut ParsedOptions,
) -> BuiltinResult<()> {
    match canonical_option_name(name)?.as_str() {
        "tolerance" => {
            let tol = numeric_scalar(value, "Tolerance")?;
            if !tol.is_finite() || tol <= 0.0 {
                return Err(error_with_detail(
                    "eigs: Tolerance must be a positive real scalar",
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
        }
        "maxiterations" => {
            positive_integer_option(value, "MaxIterations")?;
        }
        "subspacedimension" => {
            let p = positive_integer_option(value, "SubspaceDimension")?;
            let minimum = if options.is_function_symmetric || options.is_symmetric_definite {
                k + 1
            } else {
                k + 2
            };
            if p < minimum {
                return Err(error_with_detail(
                    format!("eigs: SubspaceDimension must be at least {minimum}"),
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
        }
        "startvector" => {
            ensure_exact_eigs_integer_boundary(value, "StartVector")?;
            let len = numeric_vector_len(value, "StartVector")?;
            if len != n {
                return Err(error_with_detail(
                    "eigs: StartVector length must match size(A,1)",
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
        }
        "failuretreatment" => {
            let text = text_value(value).ok_or_else(|| {
                error_with_detail(
                    "eigs: FailureTreatment must be a text scalar",
                    &ERROR_INVALID_ARGUMENT,
                )
            })?;
            match text.trim().to_ascii_lowercase().as_str() {
                "replacenan" | "keep" | "drop" => {}
                other => {
                    return Err(error_with_detail(
                        format!("eigs: unsupported FailureTreatment '{other}'"),
                        &ERROR_INVALID_ARGUMENT,
                    ));
                }
            }
        }
        "display" => {
            parse_bool(value, name)?;
        }
        "isfunctionsymmetric" => {
            options.is_function_symmetric = parse_bool(value, name)?;
        }
        "issymmetricdefinite" => {
            options.is_symmetric_definite = parse_bool(value, name)?;
        }
        "ischolesky" => {
            options.is_cholesky = parse_bool(value, name)?;
        }
        "choleskypermutation" => {
            options.cholesky_permutation = Some(parse_permutation(value, n)?);
        }
        other => {
            return Err(error_with_detail(
                format!("eigs: unsupported option '{other}'"),
                &ERROR_INVALID_ARGUMENT,
            ));
        }
    }
    Ok(())
}

fn canonical_option_name(name: &str) -> BuiltinResult<String> {
    match name.trim().to_ascii_lowercase().as_str() {
        "tol" | "tolerance" => Ok("tolerance".to_string()),
        "maxit" | "maxiterations" => Ok("maxiterations".to_string()),
        "p" | "subspacedimension" => Ok("subspacedimension".to_string()),
        "v0" | "startvector" => Ok("startvector".to_string()),
        "fail" | "failuretreatment" => Ok("failuretreatment".to_string()),
        "disp" | "display" => Ok("display".to_string()),
        "issym" | "isfunctionsymmetric" => Ok("isfunctionsymmetric".to_string()),
        "cholb" | "ischolesky" => Ok("ischolesky".to_string()),
        "permb" | "choleskypermutation" => Ok("choleskypermutation".to_string()),
        "spdb" | "issymmetricdefinite" => Ok("issymmetricdefinite".to_string()),
        other => Err(error_with_detail(
            format!("eigs: unsupported option '{other}'"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn validate_option_combination(sigma: Sigma, options: &ParsedOptions) -> BuiltinResult<()> {
    if (options.is_cholesky || options.cholesky_permutation.is_some())
        && matches!(sigma, Sigma::Near(_) | Sigma::SmallestAbs)
    {
        return Err(error_with_detail(
            "eigs: IsCholesky and CholeskyPermutation cannot be used with numeric sigma or 'smallestabs'",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    if options.is_symmetric_definite && !matches!(sigma, Sigma::Near(_) | Sigma::SmallestAbs) {
        return Err(error_with_detail(
            "eigs: IsSymmetricDefinite requires numeric sigma or 'smallestabs'",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok(())
}

async fn compute_subset(request: Request) -> BuiltinResult<SubsetEval> {
    let mut eig_args = Vec::new();
    if let Some(b) = request.b {
        eig_args.push(b);
    }
    eig_args.push(Value::String("vector".to_string()));
    let eval = eig::evaluate(request.a, &eig_args, false)
        .await
        .map_err(with_context)?;
    let output_dtype = floating_output_dtype(&eval.eigenvalues());
    let all_values = eigenvalue_list(&eval.eigenvalues())?;
    let selected = select_indices(&all_values, request.k, request.sigma);
    let values = select_vector(&all_values, &selected, output_dtype)?;
    let diagonal = diagonal_from_selected(&all_values, &selected, output_dtype)?;
    let vectors = select_columns(&eval.right(), &selected)?;
    Ok(SubsetEval {
        values,
        vectors,
        diagonal,
    })
}

fn eigenvalue_list(value: &Value) -> BuiltinResult<Vec<Complex64>> {
    match value {
        Value::Tensor(tensor) => Ok(tensor::tensor_values_f64_cow(tensor)
            .iter()
            .map(|&re| Complex64::new(re, 0.0))
            .collect()),
        Value::ComplexTensor(tensor) => Ok(tensor
            .materialize_f64()
            .iter()
            .map(|&(re, im)| Complex64::new(re, im))
            .collect()),
        Value::Num(n) => Ok(vec![Complex64::new(*n, 0.0)]),
        Value::Complex(re, im) => Ok(vec![Complex64::new(*re, *im)]),
        other => Err(error_with_detail(
            format!("eigs: unexpected eigenvalue output {other:?}"),
            &ERROR_INTERNAL,
        )),
    }
}

fn select_indices(values: &[Complex64], k: usize, sigma: Sigma) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    let k = k.min(indices.len());
    match sigma {
        Sigma::LargestAbs => sort_take(&mut indices, values, k, |value| -value.norm()),
        Sigma::SmallestAbs => sort_take(&mut indices, values, k, |value| value.norm()),
        Sigma::LargestReal => sort_take(&mut indices, values, k, |value| -value.re),
        Sigma::SmallestReal => sort_take(&mut indices, values, k, |value| value.re),
        Sigma::LargestImag => sort_take(&mut indices, values, k, |value| -value.im),
        Sigma::SmallestImag => sort_take(&mut indices, values, k, |value| value.im),
        Sigma::Near(target) => sort_take(&mut indices, values, k, |value| (*value - target).norm()),
        Sigma::BothEndsReal => both_ends_real(values, k),
        Sigma::BothEndsImag => both_ends_imag(values, k),
    }
}

fn sort_take(
    indices: &mut Vec<usize>,
    values: &[Complex64],
    k: usize,
    metric: impl Fn(&Complex64) -> f64,
) -> Vec<usize> {
    indices.sort_by(|&lhs, &rhs| total_cmp(metric(&values[lhs]), metric(&values[rhs])));
    indices.truncate(k);
    indices.clone()
}

fn both_ends_real(values: &[Complex64], k: usize) -> Vec<usize> {
    let mut by_real: Vec<usize> = (0..values.len()).collect();
    by_real.sort_by(|&lhs, &rhs| total_cmp(values[lhs].re, values[rhs].re));
    let low = k / 2;
    let high = k - low;
    let mut selected = Vec::with_capacity(k);
    selected.extend(by_real.iter().take(low).copied());
    selected.extend(by_real.iter().rev().take(high).copied());
    selected.sort_by(|&lhs, &rhs| total_cmp(values[lhs].re, values[rhs].re));
    selected.dedup();
    selected.truncate(k);
    selected
}

fn both_ends_imag(values: &[Complex64], k: usize) -> Vec<usize> {
    let mut by_imag: Vec<usize> = (0..values.len()).collect();
    by_imag.sort_by(|&lhs, &rhs| total_cmp(values[lhs].im, values[rhs].im));
    let low = k / 2;
    let high = k - low;
    let mut selected = Vec::with_capacity(k);
    selected.extend(by_imag.iter().take(low).copied());
    selected.extend(by_imag.iter().rev().take(high).copied());
    selected.sort_by(|&lhs, &rhs| {
        total_cmp(values[rhs].im.abs(), values[lhs].im.abs())
            .then_with(|| total_cmp(values[rhs].im, values[lhs].im))
    });
    selected.dedup();
    selected.truncate(k);
    selected
}

fn total_cmp(lhs: f64, rhs: f64) -> Ordering {
    lhs.total_cmp(&rhs)
}

fn select_vector(
    values: &[Complex64],
    selected: &[usize],
    dtype: NumericDType,
) -> BuiltinResult<Value> {
    let data = selected.iter().map(|&idx| values[idx]).collect::<Vec<_>>();
    complex_vector_value(data, dtype)
}

fn diagonal_from_selected(
    values: &[Complex64],
    selected: &[usize],
    dtype: NumericDType,
) -> BuiltinResult<Value> {
    let k = selected.len();
    let mut data = vec![Complex64::new(0.0, 0.0); k * k];
    for (out_idx, &value_idx) in selected.iter().enumerate() {
        data[out_idx + out_idx * k] = values[value_idx];
    }
    complex_matrix_value_with_dtype(data, k, k, dtype)
}

fn select_columns(matrix: &Value, selected: &[usize]) -> BuiltinResult<Value> {
    match matrix {
        Value::Tensor(tensor) => {
            let rows = tensor.rows();
            let mut indices = Vec::with_capacity(rows * selected.len());
            for &col in selected {
                if col >= tensor.cols() {
                    return Err(error_with_detail(
                        "eigs: eigenvector column index out of range",
                        &ERROR_INTERNAL,
                    ));
                }
                for row in 0..rows {
                    indices.push(row + col * rows);
                }
            }
            let storage = tensor
                .clone()
                .into_numeric_storage()
                .map_err(|err| error_with_detail(format!("eigs: {err}"), &ERROR_INTERNAL))?
                .gather(&indices)
                .map_err(|err| error_with_detail(format!("eigs: {err}"), &ERROR_INTERNAL))?;
            Tensor::from_numeric_storage(storage, vec![rows, selected.len()])
                .map(Value::Tensor)
                .map_err(|err| error_with_detail(format!("eigs: {err}"), &ERROR_INTERNAL))
        }
        Value::ComplexTensor(tensor) => {
            let rows = tensor.rows;
            let mut data = Vec::with_capacity(rows * selected.len());
            for &col in selected {
                if col >= tensor.cols {
                    return Err(error_with_detail(
                        "eigs: eigenvector column index out of range",
                        &ERROR_INTERNAL,
                    ));
                }
                for row in 0..rows {
                    let (re, im) = tensor.materialize_f64()[row + col * rows];
                    data.push(Complex64::new(re, im));
                }
            }
            complex_matrix_value_with_dtype(data, rows, selected.len(), tensor.numeric_dtype())
        }
        other => Err(error_with_detail(
            format!("eigs: unexpected eigenvector output {other:?}"),
            &ERROR_INTERNAL,
        )),
    }
}

fn floating_output_dtype(value: &Value) -> NumericDType {
    match value {
        Value::Tensor(tensor) => tensor.numeric_dtype(),
        Value::ComplexTensor(tensor) => tensor.numeric_dtype(),
        _ => NumericDType::F64,
    }
}

fn complex_vector_value(data: Vec<Complex64>, dtype: NumericDType) -> BuiltinResult<Value> {
    if data.iter().all(|value| value.im.abs() <= 1e-12) {
        let shape = vec![data.len(), 1];
        let tensor = if dtype == NumericDType::F32 {
            Tensor::from_f32(data.iter().map(|value| value.re as f32).collect(), shape)
        } else {
            Tensor::new(data.iter().map(|value| value.re).collect(), shape)
        };
        tensor
            .map(Value::Tensor)
            .map_err(|err| error_with_detail(format!("eigs: {err}"), &ERROR_INTERNAL))
    } else {
        ComplexTensor::from_f64_values_with_dtype(
            data.iter().map(|value| (value.re, value.im)).collect(),
            vec![data.len(), 1],
            dtype,
        )
        .map(Value::ComplexTensor)
        .map_err(|err| error_with_detail(format!("eigs: {err}"), &ERROR_INTERNAL))
    }
}

fn complex_matrix_value(data: Vec<Complex64>, rows: usize, cols: usize) -> BuiltinResult<Value> {
    complex_matrix_value_with_dtype(data, rows, cols, NumericDType::F64)
}

fn complex_matrix_value_with_dtype(
    data: Vec<Complex64>,
    rows: usize,
    cols: usize,
    dtype: NumericDType,
) -> BuiltinResult<Value> {
    if data.iter().all(|value| value.im.abs() <= 1e-12) {
        let tensor = if dtype == NumericDType::F32 {
            Tensor::from_f32(
                data.iter().map(|value| value.re as f32).collect(),
                vec![rows, cols],
            )
        } else {
            Tensor::new_2d(data.iter().map(|value| value.re).collect(), rows, cols)
        };
        tensor
            .map(Value::Tensor)
            .map_err(|err| error_with_detail(format!("eigs: {err}"), &ERROR_INTERNAL))
    } else {
        ComplexTensor::from_f64_values_with_dtype(
            data.iter().map(|value| (value.re, value.im)).collect(),
            vec![rows, cols],
            dtype,
        )
        .map(Value::ComplexTensor)
        .map_err(|err| error_with_detail(format!("eigs: {err}"), &ERROR_INTERNAL))
    }
}

fn cholesky_factor_to_b(value: Value, permutation: Option<&[usize]>) -> BuiltinResult<Value> {
    let r = dense_complex_matrix(value)?;
    let n = r.rows;
    if r.rows != r.cols {
        return Err(error_with_detail(
            "eigs: Cholesky factor B must be square",
            &ERROR_INVALID_INPUT,
        ));
    }
    let mut b = vec![Complex64::new(0.0, 0.0); n * n];
    for col in 0..n {
        for row in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..n {
                sum += r.data[k + row * n].conj() * r.data[k + col * n];
            }
            b[row + col * n] = sum;
        }
    }
    if let Some(perm) = permutation {
        let mut unpermuted = vec![Complex64::new(0.0, 0.0); n * n];
        for new_col in 0..n {
            for new_row in 0..n {
                let old_row = perm[new_row] - 1;
                let old_col = perm[new_col] - 1;
                unpermuted[old_row + old_col * n] = b[new_row + new_col * n];
            }
        }
        b = unpermuted;
    }
    complex_matrix_value(b, n, n)
}

struct DenseComplexMatrix {
    data: Vec<Complex64>,
    rows: usize,
    cols: usize,
}

fn dense_complex_matrix(value: Value) -> BuiltinResult<DenseComplexMatrix> {
    match matrix_host_value(value)? {
        Value::Tensor(tensor) => Ok(DenseComplexMatrix {
            rows: tensor.rows(),
            cols: tensor.cols(),
            data: tensor::tensor_into_values_f64(tensor)
                .into_iter()
                .map(|re| Complex64::new(re, 0.0))
                .collect(),
        }),
        Value::ComplexTensor(tensor) => Ok(DenseComplexMatrix {
            rows: tensor.rows,
            cols: tensor.cols,
            data: tensor
                .materialize_f64()
                .into_iter()
                .map(|(re, im)| Complex64::new(re, im))
                .collect(),
        }),
        Value::LogicalArray(logical) => {
            let (rows, cols) = matrix_shape_from_slice(&logical.shape);
            Ok(DenseComplexMatrix {
                rows,
                cols,
                data: logical
                    .data
                    .into_iter()
                    .map(|bit| Complex64::new(if bit != 0 { 1.0 } else { 0.0 }, 0.0))
                    .collect(),
            })
        }
        Value::Num(n) => Ok(DenseComplexMatrix {
            data: vec![Complex64::new(n, 0.0)],
            rows: 1,
            cols: 1,
        }),
        Value::Int(i) => Ok(DenseComplexMatrix {
            data: vec![Complex64::new(i.to_f64(), 0.0)],
            rows: 1,
            cols: 1,
        }),
        Value::Bool(b) => Ok(DenseComplexMatrix {
            data: vec![Complex64::new(if b { 1.0 } else { 0.0 }, 0.0)],
            rows: 1,
            cols: 1,
        }),
        Value::Complex(re, im) => Ok(DenseComplexMatrix {
            data: vec![Complex64::new(re, im)],
            rows: 1,
            cols: 1,
        }),
        other => Err(error_with_detail(
            format!("eigs: unsupported Cholesky B input {other:?}"),
            &ERROR_INVALID_INPUT,
        )),
    }
}

fn numeric_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(tensor) if tensor_element_len(tensor) == 1 => Ok(scalar_tensor_f64(tensor)),
        Value::LogicalArray(logical) if logical.len() == 1 => {
            Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
        }
        Value::Bool(flag) => Ok(if *flag { 1.0 } else { 0.0 }),
        _ => Err(error_with_detail(
            format!("eigs: {name} must be a real scalar"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn positive_integer_option(value: &Value, name: &str) -> BuiltinResult<usize> {
    if let Some(value) = exact_integer_scalar(value) {
        return value
            .try_to_usize()
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                error_with_detail(
                    format!("eigs: {name} must be a positive integer"),
                    &ERROR_INVALID_ARGUMENT,
                )
            });
    }
    let parsed = numeric_scalar(value, name)?;
    if !parsed.is_finite() || parsed <= 0.0 || parsed.fract() != 0.0 {
        return Err(error_with_detail(
            format!("eigs: {name} must be a positive integer"),
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    if parsed > usize::MAX as f64 || (usize::BITS == 64 && parsed == usize::MAX as f64) {
        return Err(error_with_detail(
            format!("eigs: {name} is too large for this platform"),
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    Ok(parsed as usize)
}

fn parse_bool(value: &Value, name: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::LogicalArray(logical) if logical.len() == 1 => Ok(logical.data[0] != 0),
        Value::Num(n) if *n == 0.0 || *n == 1.0 => Ok(*n != 0.0),
        Value::Int(i) if i.to_i64() == 0 || i.to_i64() == 1 => Ok(i.to_i64() != 0),
        Value::Tensor(tensor) if tensor_element_len(tensor) == 1 => {
            if let Some(value) = exact_integer_scalar(value) {
                let raw = value.try_to_i64();
                if raw == Some(0) || raw == Some(1) {
                    return Ok(raw != Some(0));
                }
            } else {
                let raw = scalar_tensor_f64(tensor);
                if raw == 0.0 || raw == 1.0 {
                    return Ok(raw != 0.0);
                }
            }
            Err(error_with_detail(
                format!("eigs: {name} must be logical scalar true or false"),
                &ERROR_INVALID_ARGUMENT,
            ))
        }
        _ => Err(error_with_detail(
            format!("eigs: {name} must be logical scalar true or false"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn numeric_vector_len(value: &Value, name: &str) -> BuiltinResult<usize> {
    match value {
        Value::Tensor(tensor) if tensor.rows() == 1 || tensor.cols() == 1 => {
            Ok(tensor_element_len(tensor))
        }
        Value::ComplexTensor(tensor) if tensor.rows == 1 || tensor.cols == 1 => {
            Ok(complex_tensor_element_len(tensor))
        }
        Value::LogicalArray(logical) => {
            let (rows, cols) = matrix_shape_from_slice(&logical.shape);
            if rows == 1 || cols == 1 {
                Ok(logical.data.len())
            } else {
                Err(error_with_detail(
                    format!("eigs: {name} must be a vector"),
                    &ERROR_INVALID_ARGUMENT,
                ))
            }
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(1),
        _ => Err(error_with_detail(
            format!("eigs: {name} must be a numeric vector"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn parse_permutation(value: &Value, n: usize) -> BuiltinResult<Vec<usize>> {
    if let Some(values) = exact_integer_vector(value)? {
        if values.len() != n {
            return Err(error_with_detail(
                "eigs: CholeskyPermutation length must match size(A,1)",
                &ERROR_INVALID_ARGUMENT,
            ));
        }
        let mut seen = vec![false; n];
        let mut out = Vec::with_capacity(n);
        for raw in values {
            if raw < 1 || raw > n as i128 {
                return Err(error_with_detail(
                    "eigs: CholeskyPermutation must contain 1-based integer indices",
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
            let idx = raw as usize;
            if seen[idx - 1] {
                return Err(error_with_detail(
                    "eigs: CholeskyPermutation entries must be unique",
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
            seen[idx - 1] = true;
            out.push(idx);
        }
        return Ok(out);
    }
    let values = numeric_vector_values(value, "CholeskyPermutation")?;
    if values.len() != n {
        return Err(error_with_detail(
            "eigs: CholeskyPermutation length must match size(A,1)",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    let mut seen = vec![false; n];
    let mut out = Vec::with_capacity(n);
    for raw in values {
        if !raw.is_finite() || raw.fract() != 0.0 || raw < 1.0 || raw > n as f64 {
            return Err(error_with_detail(
                "eigs: CholeskyPermutation must contain 1-based integer indices",
                &ERROR_INVALID_ARGUMENT,
            ));
        }
        let idx = raw as usize;
        if seen[idx - 1] {
            return Err(error_with_detail(
                "eigs: CholeskyPermutation entries must be unique",
                &ERROR_INVALID_ARGUMENT,
            ));
        }
        seen[idx - 1] = true;
        out.push(idx);
    }
    Ok(out)
}

fn exact_integer_vector(value: &Value) -> BuiltinResult<Option<Vec<i128>>> {
    match value {
        Value::Int(value) => Ok(Some(vec![int_value_i128(value)])),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            if tensor.rows() != 1 && tensor.cols() != 1 {
                return Err(error_with_detail(
                    "eigs: CholeskyPermutation must be a vector",
                    &ERROR_INVALID_ARGUMENT,
                ));
            }
            let mut values = Vec::with_capacity(tensor.len());
            for index in 0..tensor.len() {
                let exact = tensor
                    .numeric_value_at(index)
                    .and_then(numeric_scalar_i128)
                    .ok_or_else(|| {
                        error_with_detail(
                            "eigs: unable to read integer CholeskyPermutation",
                            &ERROR_INTERNAL,
                        )
                    })?;
                values.push(exact);
            }
            Ok(Some(values))
        }
        _ => Ok(None),
    }
}

fn numeric_vector_values(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Tensor(tensor) if tensor.rows() == 1 || tensor.cols() == 1 => {
            Ok(tensor::tensor_values_f64(tensor))
        }
        Value::LogicalArray(logical) => {
            let (rows, cols) = matrix_shape_from_slice(&logical.shape);
            if rows == 1 || cols == 1 {
                Ok(logical
                    .data
                    .iter()
                    .map(|&bit| if bit != 0 { 1.0 } else { 0.0 })
                    .collect())
            } else {
                Err(error_with_detail(
                    format!("eigs: {name} must be a vector"),
                    &ERROR_INVALID_ARGUMENT,
                ))
            }
        }
        Value::Num(n) => Ok(vec![*n]),
        Value::Int(i) => Ok(vec![i.to_f64()]),
        Value::Bool(flag) => Ok(vec![if *flag { 1.0 } else { 0.0 }]),
        _ => Err(error_with_detail(
            format!("eigs: {name} must be a numeric vector"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn text_value(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(chars) if chars.rows == 1 => Some(chars.data.iter().collect()),
        _ => None,
    }
}

fn is_text(value: &Value) -> bool {
    text_value(value).is_some()
}

fn error_with_detail(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn eigs_error(descriptor: &'static BuiltinErrorDescriptor) -> RuntimeError {
    error_with_detail(descriptor.message, descriptor)
}

fn with_context(mut error: RuntimeError) -> RuntimeError {
    if error.message() == "interaction pending..." {
        return build_runtime_error("interaction pending...")
            .with_builtin(NAME)
            .build();
    }
    if error.context.builtin.is_none() {
        error.context = error.context.with_builtin(NAME);
    }
    error
}

fn exact_integer_scalar(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor_element_len(tensor) == 1 => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    }
}

fn ensure_exact_eigs_integer_boundary(value: &Value, role: &str) -> BuiltinResult<()> {
    const MAX_EXACT_INTEGER: i128 = 1_i128 << 53;
    let check = |exact: i128| {
        if (-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&exact) {
            Ok(())
        } else {
            Err(error_with_detail(
                format!("eigs: integer {role} values must be exactly representable as double"),
                &ERROR_INVALID_ARGUMENT,
            ))
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

fn scalar_tensor_f64(tensor: &Tensor) -> f64 {
    tensor::tensor_value_f64(tensor, 0)
}

fn tensor_element_len(tensor: &Tensor) -> usize {
    tensor.len()
}

fn complex_tensor_element_len(tensor: &ComplexTensor) -> usize {
    tensor
        .integer_storage()
        .as_ref()
        .map_or(tensor.materialize_f64().len(), |storage| storage.real.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::builtin_function_by_name;
    use runmat_value::{CharArray, IntegerComplexStorage, IntegerStorage, SparseTensor};

    fn real_matrix(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new_2d(data, rows, cols).unwrap())
    }

    #[test]
    fn eigs_rejects_typed_complex_integer_inputs() {
        let tensor = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![u64::MAX]),
                IntegerStorage::U64(vec![1]),
            )
            .expect("storage"),
            vec![1, 1],
        )
        .expect("tensor");
        let err = block_on(eigs_builtin(Value::ComplexTensor(tensor), Vec::new()))
            .expect_err("typed complex integer input must reject");
        assert!(err.message().contains("complex numbers with integer types"));
    }

    #[test]
    fn eigs_sigma_parser_reads_typed_integer_complex_scalar_without_mirror() {
        let storage =
            IntegerComplexStorage::new(IntegerStorage::I16(vec![3]), IntegerStorage::I16(vec![-2]))
                .expect("storage");
        let tensor = ComplexTensor::new_integer(storage, vec![1, 1]).expect("tensor");

        match parse_sigma(&Value::ComplexTensor(tensor)).expect("sigma") {
            Some(Sigma::Near(value)) => assert_eq!(value, Complex64::new(3.0, -2.0)),
            _ => panic!("expected near sigma"),
        }
    }

    fn tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn outputs(value: Value) -> Vec<Value> {
        match value {
            Value::OutputList(values) => values,
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn eigs_selected_real_columns_preserve_native_single_storage() {
        let matrix = Value::Tensor(
            Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("single vectors"),
        );
        let selected = tensor(select_columns(&matrix, &[1]).expect("selected column"));
        assert_eq!(selected.numeric_dtype(), runmat_value::NumericDType::F32);
        assert_eq!(selected.shape, vec![2, 1]);
        assert_eq!(selected.materialize_f64(), vec![3.0, 4.0]);
    }

    fn call(a: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(eigs_builtin(a, rest))
    }

    #[test]
    fn eigs_nonfloating_coefficients_follow_compatibility_mode() {
        let integer_matrix = || {
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![1, 0, 0, 8]), vec![2, 2])
                    .expect("integer coefficient matrix"),
            )
        };
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error =
                call(integer_matrix(), Vec::new()).expect_err("MATLAB mode rejects integer A");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:EigsNonfloatingMatrixExtension")
            );

            let a = real_matrix(vec![1.0, 0.0, 0.0, 8.0], 2, 2);
            let error = call(a, vec![integer_matrix()])
                .expect_err("MATLAB mode rejects integer generalized B");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:EigsNonfloatingMatrixExtension")
            );

            let a = real_matrix(vec![1.0, 0.0, 0.0, 8.0], 2, 2);
            let k = Value::Int(IntValue::I16(1));
            assert!(call(a, vec![k]).is_ok(), "integer k is not a coefficient");
        }
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            assert!(call(integer_matrix(), Vec::new()).is_ok());
        }

        let handle = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![2, 2],
            device_id: 0,
            buffer_id: 9_300_002,
        };
        runmat_accelerate_api::set_handle_integer_type(
            &handle,
            runmat_accelerate_api::IntegerElementType::I16,
        );
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = call(Value::GpuTensor(handle.clone()), Vec::new())
                .expect_err("MATLAB mode rejects resident integer A before gather");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:EigsGpuInputExtension")
            );
        }
        runmat_accelerate_api::clear_handle_integer_type(&handle);
    }

    #[test]
    fn eigs_integer_controls_have_role_specific_strict_mode_errors() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let a = real_matrix(vec![1.0, 0.0, 0.0, 8.0], 2, 2);
        let sigma =
            Value::Tensor(Tensor::new_integer(IntegerStorage::I16(vec![7]), vec![1, 1]).unwrap());
        let error = call(a.clone(), vec![Value::Num(1.0), sigma])
            .expect_err("typed-integer sigma must gate");
        assert_eq!(
            error.identifier(),
            EIGS_INTEGER_SIGMA_EXTENSION.error_identifier
        );

        let mut options = StructValue::new();
        options.insert(
            "StartVector",
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![2, 1]).unwrap(),
            ),
        );
        let error = call(a, vec![Value::Num(1.0), Value::Struct(options)])
            .expect_err("typed-integer StartVector must gate");
        assert_eq!(
            error.identifier(),
            EIGS_INTEGER_START_VECTOR_EXTENSION.error_identifier
        );
    }

    #[test]
    fn eigs_nested_resident_start_vector_gates_before_recursive_gather() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let a = real_matrix(vec![1.0, 0.0, 0.0, 8.0], 2, 2);
        let mut options = StructValue::new();
        options.insert(
            "StartVector",
            Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
                shape: vec![2, 1],
                device_id: u32::MAX,
                buffer_id: u64::MAX,
            }),
        );
        let error = call(a, vec![Value::Num(1.0), Value::Struct(options)])
            .expect_err("nested resident StartVector must gate before gather");
        assert_eq!(
            error.identifier(),
            EIGS_GPU_INPUT_EXTENSION.error_identifier
        );
    }

    #[test]
    fn eigs_admits_all_integer_coefficient_classes_and_rejects_wide_boundaries() {
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
            let a = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).unwrap());
            assert!(call(a, Vec::new()).is_ok());
        }

        let wide = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]).unwrap(),
        );
        let error = call(wide, Vec::new()).expect_err("wide coefficient must reject");
        assert!(error.message().contains("exactly representable as double"));

        let a = real_matrix(vec![1.0, 0.0, 0.0, 8.0], 2, 2);
        let wide_sigma = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]).unwrap(),
        );
        let error = call(a, vec![Value::Num(1.0), wide_sigma]).expect_err("wide sigma must reject");
        assert!(error.message().contains("exactly representable as double"));
    }

    #[test]
    fn eigs_preserves_single_outputs_and_registers_integer_policy() {
        let a = Value::Tensor(Tensor::from_f32(vec![1.0, 0.0, 0.0, 8.0], vec![2, 2]).unwrap());
        let output = tensor(call(a, vec![Value::Num(1.0)]).unwrap());
        assert_eq!(output.numeric_dtype(), NumericDType::F32);

        let builtin = builtin_function_by_name(NAME).expect("registered eigs");
        assert_eq!(builtin.extensions, &EIGS_EXTENSIONS);
        assert_eq!(builtin.integer_capabilities.len(), 4);
    }

    #[test]
    fn eigs_defaults_to_largest_magnitude_six_clamped() {
        let a = real_matrix(vec![1.0, 0.0, 0.0, 4.0], 2, 2);
        let out = tensor(call(a, Vec::new()).unwrap());
        assert_eq!(out.shape, vec![2, 1]);
        assert_eq!(out.materialize_f64(), vec![4.0, 1.0]);
    }

    #[test]
    fn eigs_selects_smallestabs_from_sparse_input() {
        let sparse =
            SparseTensor::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![10.0, 2.0, -4.0])
                .unwrap();
        let out = tensor(
            call(
                Value::SparseTensor(sparse),
                vec![
                    Value::Num(2.0),
                    Value::CharArray(CharArray::new_row("smallestabs")),
                ],
            )
            .unwrap(),
        );
        assert_eq!(out.materialize_f64(), vec![2.0, -4.0]);
    }

    #[test]
    fn eigs_supports_two_and_three_output_forms() {
        let guard = crate::output_count::push_output_count(Some(3));
        let a = real_matrix(vec![2.0, 0.0, 0.0, 5.0], 2, 2);
        let out = outputs(call(a, vec![Value::Num(1.0)]).unwrap());
        drop(guard);
        assert_eq!(out.len(), 3);
        assert_eq!(tensor(out[0].clone()).shape, vec![2, 1]);
        let d = tensor(out[1].clone());
        assert_eq!(d.shape, vec![1, 1]);
        assert_eq!(d.materialize_f64(), vec![5.0]);
        assert_eq!(out[2], Value::Num(0.0));
    }

    #[test]
    fn eigs_numeric_sigma_selects_nearest() {
        let a = real_matrix(vec![1.0, 0.0, 0.0, 8.0], 2, 2);
        let out = tensor(call(a, vec![Value::Num(1.0), Value::Num(7.25)]).unwrap());
        assert_eq!(out.materialize_f64(), vec![8.0]);
    }

    #[test]
    fn eigs_reads_integer_tensor_k_and_sigma_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let a = real_matrix(vec![1.0, 0.0, 0.0, 8.0], 2, 2);
        let k = Tensor::new_integer(IntegerStorage::U64(vec![1]), vec![1, 1]).unwrap();
        let sigma = Tensor::new_integer(IntegerStorage::I16(vec![7]), vec![1, 1]).unwrap();
        let out = tensor(call(a, vec![Value::Tensor(k), Value::Tensor(sigma)]).unwrap());
        assert_eq!(out.materialize_f64(), vec![8.0]);
    }

    #[test]
    fn eigs_reads_integer_tensor_vector_options_exactly() {
        let permutation =
            Tensor::new_integer(IntegerStorage::U16(vec![2, 1, 3]), vec![1, 3]).unwrap();
        assert_eq!(
            parse_permutation(&Value::Tensor(permutation), 3).unwrap(),
            vec![2, 1, 3]
        );

        let initial = Tensor::new_integer(IntegerStorage::I16(vec![3, 2, 1]), vec![3, 1]).unwrap();
        assert_eq!(
            numeric_vector_values(&Value::Tensor(initial), "InitialVector").unwrap(),
            vec![3.0, 2.0, 1.0]
        );

        let fail = Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).unwrap();
        assert!(parse_bool(&Value::Tensor(fail), "Fail").unwrap());
    }

    #[test]
    fn eigs_generalized_reuses_eig_path() {
        let a = real_matrix(vec![2.0, 0.0, 0.0, 9.0], 2, 2);
        let b = real_matrix(vec![1.0, 0.0, 0.0, 3.0], 2, 2);
        let out = tensor(
            call(
                a,
                vec![
                    b,
                    Value::Num(1.0),
                    Value::CharArray(CharArray::new_row("smallestreal")),
                ],
            )
            .unwrap(),
        );
        assert_eq!(out.materialize_f64(), vec![2.0]);
    }

    #[test]
    fn eigs_validates_options_without_affecting_exact_fallback() {
        let mut opts = StructValue::new();
        opts.insert("tol", Value::Num(1e-9));
        let maxit = Tensor::new_integer(IntegerStorage::U16(vec![100]), vec![1, 1]).unwrap();
        opts.insert("maxit", Value::Tensor(maxit));
        let p = Tensor::new_integer(IntegerStorage::U16(vec![4]), vec![1, 1]).unwrap();
        opts.insert("p", Value::Tensor(p));
        opts.insert(
            "v0",
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap()),
        );
        opts.insert("fail", Value::CharArray(CharArray::new_row("keep")));
        let disp = Tensor::new_integer(IntegerStorage::U8(vec![0]), vec![1, 1]).unwrap();
        opts.insert("disp", Value::Tensor(disp));
        let a = real_matrix(vec![1.0, 0.0, 0.0, 3.0], 2, 2);
        let out = tensor(
            call(
                a,
                vec![
                    Value::Num(1.0),
                    Value::String("lm".into()),
                    Value::Struct(opts),
                ],
            )
            .unwrap(),
        );
        assert_eq!(out.materialize_f64(), vec![3.0]);
    }

    #[test]
    fn eigs_validates_generalized_option_sigma_constraints() {
        let a = real_matrix(vec![2.0, 0.0, 0.0, 9.0], 2, 2);
        let b = real_matrix(vec![1.0, 0.0, 0.0, 3.0], 2, 2);
        let mut chol_opts = StructValue::new();
        chol_opts.insert("IsCholesky", Value::Bool(true));
        assert!(call(
            a.clone(),
            vec![
                b.clone(),
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Struct(chol_opts)
            ],
        )
        .is_err());

        let mut spd_opts = StructValue::new();
        spd_opts.insert("IsSymmetricDefinite", Value::Bool(true));
        assert!(call(
            a,
            vec![
                b,
                Value::Num(1.0),
                Value::CharArray(CharArray::new_row("largestabs")),
                Value::Struct(spd_opts),
            ],
        )
        .is_err());
    }

    #[test]
    fn eigs_rejects_too_small_subspace_dimension() {
        let a = real_matrix(vec![1.0, 0.0, 0.0, 3.0], 2, 2);
        let mut opts = StructValue::new();
        opts.insert("p", Value::Num(2.0));
        assert!(call(
            a,
            vec![
                Value::Num(1.0),
                Value::String("lm".into()),
                Value::Struct(opts)
            ]
        )
        .is_err());
    }

    #[test]
    fn eigs_positive_integer_options_read_typed_storage_and_reject_float_boundary() {
        let wide = if usize::BITS == 64 {
            9_007_199_254_740_993
        } else {
            u32::MAX as u64
        };
        let typed =
            Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("typed option");

        assert_eq!(
            positive_integer_option(&Value::Tensor(typed), "MaxIterations").expect("typed option"),
            wide as usize
        );

        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        assert!(positive_integer_option(&Value::Num(boundary), "MaxIterations").is_err());
        assert!(parse_k(&Value::Num(i64::MAX as f64)).is_err());
    }

    #[test]
    fn eigs_accepts_cholesky_factor_for_generalized_b() {
        let a = real_matrix(vec![2.0, 0.0, 0.0, 9.0], 2, 2);
        let r = real_matrix(vec![1.0, 0.0, 0.0, 2.0], 2, 2);
        let mut opts = StructValue::new();
        opts.insert("IsCholesky", Value::Bool(true));
        let out = tensor(
            call(
                a,
                vec![
                    r,
                    Value::Num(1.0),
                    Value::CharArray(CharArray::new_row("smallestreal")),
                    Value::Struct(opts),
                ],
            )
            .unwrap(),
        );
        assert_eq!(out.materialize_f64(), vec![2.0]);
    }

    #[test]
    fn eigs_cholesky_factor_reads_typed_integer_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let a = real_matrix(vec![2.0, 0.0, 0.0, 9.0], 2, 2);
        let r = Tensor::new_integer(IntegerStorage::U16(vec![1, 0, 0, 2]), vec![2, 2]).unwrap();
        let mut opts = StructValue::new();
        opts.insert("IsCholesky", Value::Bool(true));

        let out = tensor(
            call(
                a,
                vec![
                    Value::Tensor(r),
                    Value::Num(1.0),
                    Value::CharArray(CharArray::new_row("smallestreal")),
                    Value::Struct(opts),
                ],
            )
            .unwrap(),
        );

        assert_eq!(out.materialize_f64(), vec![2.0]);
    }

    #[test]
    fn eigs_start_vector_length_reads_typed_integer_storage_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let a = real_matrix(vec![1.0, 0.0, 0.0, 3.0], 2, 2);
        let start = Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![2, 1]).unwrap();
        let mut opts = StructValue::new();
        opts.insert("StartVector", Value::Tensor(start));

        let out = tensor(call(a, vec![Value::Num(1.0), Value::Struct(opts)]).unwrap());

        assert_eq!(out.shape, vec![1, 1]);
        assert_eq!(out.materialize_f64(), vec![3.0]);
    }

    #[test]
    fn eigs_rejects_bad_k_sigma_and_function_handle() {
        let a = real_matrix(vec![1.0, 0.0, 0.0, 3.0], 2, 2);
        assert!(call(a.clone(), vec![Value::Num(0.0)]).is_err());
        assert!(call(
            a.clone(),
            vec![Value::Num(1.0), Value::CharArray(CharArray::new_row("bad"))],
        )
        .is_err());
        assert!(call(Value::FunctionHandle("afun".into()), vec![Value::Num(2.0)]).is_err());
    }
}
