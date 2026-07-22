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
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, ResolveContext, StructValue, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use super::eig;

const NAME: &str = "eigs";
const DEFAULT_K: usize = 6;

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
    builtin_path = "crate::builtins::math::linalg::factor::eigs"
)]
async fn eigs_builtin(a: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
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
                b = Some(matrix_host_value(args.remove(0))?);
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
            "eigs: input matrix must be square",
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
        Value::Tensor(tensor) => tensor.data.is_empty(),
        Value::ComplexTensor(tensor) => tensor.data.is_empty(),
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
    let raw = match value {
        Value::Int(int) => int.to_f64(),
        Value::Num(n) => *n,
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data[0],
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
    if raw > (usize::MAX as f64) {
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
        Value::Int(i) => Ok(Some(Sigma::Near(Complex64::new(i.to_f64(), 0.0)))),
        Value::Complex(re, im) => Ok(Some(Sigma::Near(Complex64::new(*re, *im)))),
        Value::Tensor(tensor) if tensor.data.len() == 1 => {
            Ok(Some(Sigma::Near(Complex64::new(tensor.data[0], 0.0))))
        }
        Value::ComplexTensor(tensor) if tensor.data.len() == 1 => {
            let (re, im) = tensor.data[0];
            Ok(Some(Sigma::Near(Complex64::new(re, im))))
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
    let all_values = eigenvalue_list(&eval.eigenvalues())?;
    let selected = select_indices(&all_values, request.k, request.sigma);
    let values = select_vector(&all_values, &selected)?;
    let diagonal = diagonal_from_selected(&all_values, &selected)?;
    let vectors = select_columns(&eval.right(), &selected)?;
    Ok(SubsetEval {
        values,
        vectors,
        diagonal,
    })
}

fn eigenvalue_list(value: &Value) -> BuiltinResult<Vec<Complex64>> {
    match value {
        Value::Tensor(tensor) => Ok(tensor
            .data
            .iter()
            .map(|&re| Complex64::new(re, 0.0))
            .collect()),
        Value::ComplexTensor(tensor) => Ok(tensor
            .data
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

fn select_vector(values: &[Complex64], selected: &[usize]) -> BuiltinResult<Value> {
    let data = selected.iter().map(|&idx| values[idx]).collect::<Vec<_>>();
    complex_vector_value(data)
}

fn diagonal_from_selected(values: &[Complex64], selected: &[usize]) -> BuiltinResult<Value> {
    let k = selected.len();
    let mut data = vec![Complex64::new(0.0, 0.0); k * k];
    for (out_idx, &value_idx) in selected.iter().enumerate() {
        data[out_idx + out_idx * k] = values[value_idx];
    }
    complex_matrix_value(data, k, k)
}

fn select_columns(matrix: &Value, selected: &[usize]) -> BuiltinResult<Value> {
    match matrix {
        Value::Tensor(tensor) => {
            let rows = tensor.rows();
            let mut data = Vec::with_capacity(rows * selected.len());
            for &col in selected {
                if col >= tensor.cols() {
                    return Err(error_with_detail(
                        "eigs: eigenvector column index out of range",
                        &ERROR_INTERNAL,
                    ));
                }
                for row in 0..rows {
                    data.push(tensor.data[row + col * rows]);
                }
            }
            Tensor::new_2d(data, rows, selected.len())
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
                    let (re, im) = tensor.data[row + col * rows];
                    data.push(Complex64::new(re, im));
                }
            }
            complex_matrix_value(data, rows, selected.len())
        }
        other => Err(error_with_detail(
            format!("eigs: unexpected eigenvector output {other:?}"),
            &ERROR_INTERNAL,
        )),
    }
}

fn complex_vector_value(data: Vec<Complex64>) -> BuiltinResult<Value> {
    if data.iter().all(|value| value.im.abs() <= 1e-12) {
        Tensor::new(
            data.iter().map(|value| value.re).collect(),
            vec![data.len(), 1],
        )
        .map(Value::Tensor)
        .map_err(|err| error_with_detail(format!("eigs: {err}"), &ERROR_INTERNAL))
    } else {
        ComplexTensor::new(
            data.iter().map(|value| (value.re, value.im)).collect(),
            vec![data.len(), 1],
        )
        .map(Value::ComplexTensor)
        .map_err(|err| error_with_detail(format!("eigs: {err}"), &ERROR_INTERNAL))
    }
}

fn complex_matrix_value(data: Vec<Complex64>, rows: usize, cols: usize) -> BuiltinResult<Value> {
    if data.iter().all(|value| value.im.abs() <= 1e-12) {
        Tensor::new_2d(data.iter().map(|value| value.re).collect(), rows, cols)
            .map(Value::Tensor)
            .map_err(|err| error_with_detail(format!("eigs: {err}"), &ERROR_INTERNAL))
    } else {
        ComplexTensor::new_2d(
            data.iter().map(|value| (value.re, value.im)).collect(),
            rows,
            cols,
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
            data: tensor
                .data
                .into_iter()
                .map(|re| Complex64::new(re, 0.0))
                .collect(),
        }),
        Value::ComplexTensor(tensor) => Ok(DenseComplexMatrix {
            rows: tensor.rows,
            cols: tensor.cols,
            data: tensor
                .data
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
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(tensor.data[0]),
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
    let parsed = numeric_scalar(value, name)?;
    if !parsed.is_finite() || parsed <= 0.0 || parsed.fract() != 0.0 {
        return Err(error_with_detail(
            format!("eigs: {name} must be a positive integer"),
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
        Value::Tensor(tensor)
            if tensor.data.len() == 1 && (tensor.data[0] == 0.0 || tensor.data[0] == 1.0) =>
        {
            Ok(tensor.data[0] != 0.0)
        }
        _ => Err(error_with_detail(
            format!("eigs: {name} must be logical scalar true or false"),
            &ERROR_INVALID_ARGUMENT,
        )),
    }
}

fn numeric_vector_len(value: &Value, name: &str) -> BuiltinResult<usize> {
    match value {
        Value::Tensor(tensor) if tensor.rows() == 1 || tensor.cols() == 1 => Ok(tensor.data.len()),
        Value::ComplexTensor(tensor) if tensor.rows == 1 || tensor.cols == 1 => {
            Ok(tensor.data.len())
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

fn numeric_vector_values(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Tensor(tensor) if tensor.rows() == 1 || tensor.cols() == 1 => {
            Ok(tensor.data.clone())
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{CharArray, IntValue, SparseTensor};

    fn real_matrix(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new_2d(data, rows, cols).unwrap())
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

    fn call(a: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(eigs_builtin(a, rest))
    }

    #[test]
    fn eigs_defaults_to_largest_magnitude_six_clamped() {
        let a = real_matrix(vec![1.0, 0.0, 0.0, 4.0], 2, 2);
        let out = tensor(call(a, Vec::new()).unwrap());
        assert_eq!(out.shape, vec![2, 1]);
        assert_eq!(out.data, vec![4.0, 1.0]);
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
        assert_eq!(out.data, vec![2.0, -4.0]);
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
        assert_eq!(d.data, vec![5.0]);
        assert_eq!(out[2], Value::Num(0.0));
    }

    #[test]
    fn eigs_numeric_sigma_selects_nearest() {
        let a = real_matrix(vec![1.0, 0.0, 0.0, 8.0], 2, 2);
        let out = tensor(call(a, vec![Value::Num(1.0), Value::Num(7.25)]).unwrap());
        assert_eq!(out.data, vec![8.0]);
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
        assert_eq!(out.data, vec![2.0]);
    }

    #[test]
    fn eigs_validates_options_without_affecting_exact_fallback() {
        let mut opts = StructValue::new();
        opts.insert("tol", Value::Num(1e-9));
        opts.insert("maxit", Value::Int(IntValue::I32(100)));
        opts.insert("p", Value::Int(IntValue::I32(4)));
        opts.insert(
            "v0",
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap()),
        );
        opts.insert("fail", Value::CharArray(CharArray::new_row("keep")));
        opts.insert("disp", Value::Bool(false));
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
        assert_eq!(out.data, vec![3.0]);
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
        assert_eq!(out.data, vec![2.0]);
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
