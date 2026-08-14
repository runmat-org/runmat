//! Multivariate normal random variates.

use nalgebra::{DMatrix, SymmetricEigen};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, NumericScalar, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor;
use crate::builtins::common::{gpu_helpers, random};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "mvnrnd";
const SYMMETRY_TOL: f64 = 1.0e-10;
const PSD_TOL: f64 = 1.0e-10;
const MAX_OUTPUT_CELLS: usize = 20_000_000;
const MAX_FACTOR_CELLS: usize = 20_000_000;

const INTEGER_MU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mvnrnd-integer-mu",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mvnrnd with a typed-integer mean array is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MvnrndIntegerMuExtension"),
};
const INTEGER_SIGMA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mvnrnd-integer-sigma",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mvnrnd with a typed-integer covariance array is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MvnrndIntegerSigmaExtension"),
};
const INTEGER_N_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mvnrnd-integer-count",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mvnrnd with a typed-integer sample count is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MvnrndIntegerCountExtension"),
};
const LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mvnrnd-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mvnrnd with logical numeric inputs is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MvnrndLogicalInputExtension"),
};
const RESIDENT_N_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mvnrnd-resident-count",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mvnrnd with an explicit gpuArray sample count is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MvnrndResidentCountExtension"),
};
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    INTEGER_MU_EXTENSION,
    INTEGER_SIGMA_EXTENSION,
    INTEGER_N_EXTENSION,
    LOGICAL_INPUT_EXTENSION,
    RESIDENT_N_EXTENSION,
];

const INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "mu",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer means are independently gated and require exact binary64 representation.",
    },
    BuiltinIntegerInputCapability {
        name: "Sigma",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer covariance values are independently gated and require exact binary64 representation.",
    },
];
const INTEGER_N_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "n",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Current MATLAB documents single and double n; native integer n is a gated RunMat structural extension.",
}];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "r = mvnrnd(integer_mu, integer_Sigma, ___)",
        inputs: &INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer distribution arrays are RunMat-only, cross one checked binary64 boundary, and return double unless another documented data input is single; fallback restores through the exact owner when it can preserve the required class, otherwise automatic residency may remain host and explicit residency errors.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "r = mvnrnd(mu, Sigma, integer_n)",
        inputs: &INTEGER_N_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer n is parsed exactly without selecting output class or residency; explicit resident n is separately gated.",
    },
];

const OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Random samples with observations in rows.",
}];

const INPUT_MU: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "mu",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Mean vector or matrix with one mean vector per row.",
};

const INPUT_SIGMA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Sigma",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Covariance matrix or covariance matrix pages.",
};

const INPUT_N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Number of random rows to draw when mu is a single mean vector.",
};

const INPUTS_MU_SIGMA: [BuiltinParamDescriptor; 2] = [INPUT_MU, INPUT_SIGMA];
const INPUTS_MU_SIGMA_N: [BuiltinParamDescriptor; 3] = [INPUT_MU, INPUT_SIGMA, INPUT_N];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "r = mvnrnd(mu, Sigma)",
        inputs: &INPUTS_MU_SIGMA,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = mvnrnd(mu, Sigma, n)",
        inputs: &INPUTS_MU_SIGMA_N,
        outputs: &OUTPUT_R,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MVNRND.INVALID_ARGUMENT",
    identifier: Some("RunMat:mvnrnd:InvalidArgument"),
    when: "Inputs are malformed, dimensions are incompatible, or covariance pages are not symmetric positive semidefinite.",
    message: "mvnrnd: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MVNRND.INTERNAL",
    identifier: Some("RunMat:mvnrnd:Internal"),
    when: "RunMat cannot allocate or construct the requested random samples.",
    message: "mvnrnd: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const MVNRND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn mvnrnd_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    if args.len() >= 2 {
        Type::Tensor { shape: None }
    } else {
        Type::Unknown
    }
}

fn mvnrnd_error(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    mvnrnd_error(&ERROR_INVALID_ARGUMENT, message)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    mvnrnd_error(&ERROR_INTERNAL, message)
}

#[runtime_builtin(
    name = "mvnrnd",
    category = "stats/random",
    summary = "Generate multivariate normal random samples.",
    keywords = "mvnrnd,multivariate normal,gaussian,random,statistics",
    type_resolver(mvnrnd_type),
    descriptor(crate::builtins::stats::random::mvnrnd::MVNRND_DESCRIPTOR),
    extensions(crate::builtins::stats::random::mvnrnd::EXTENSIONS),
    integer_capabilities(crate::builtins::stats::random::mvnrnd::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::random::mvnrnd"
)]
async fn mvnrnd_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_extensions(&args)?;
    let output = MvnrndOutputPlan::inspect(&args)?;
    let parsed = parse_args(args).await?;
    let output_len = parsed
        .sample_count
        .checked_mul(parsed.dimension)
        .ok_or_else(|| internal("mvnrnd: output size overflow"))?;
    if output_len > MAX_OUTPUT_CELLS {
        return Err(invalid("mvnrnd: requested output is too large"));
    }
    let normals = random::generate_normal_scaled(0.0, 1.0, output_len, NAME)?;
    let mut out = vec![0.0; output_len];
    for sample in 0..parsed.sample_count {
        let mean = &parsed.means[parsed.mean_index(sample)];
        let factor = &parsed.factors[parsed.factor_index(sample)];
        for dim in 0..parsed.dimension {
            let mut value = mean[dim];
            for latent in 0..parsed.dimension {
                value += factor[(dim, latent)] * normals[latent * parsed.sample_count + sample];
            }
            out[dim * parsed.sample_count + sample] = value;
        }
    }
    output.finish(out, vec![parsed.sample_count, parsed.dimension])
}

struct MvnrndOutputPlan {
    single: bool,
    source: Option<runmat_accelerate_api::GpuTensorHandle>,
}

impl MvnrndOutputPlan {
    fn inspect(args: &[Value]) -> BuiltinResult<Self> {
        let single = args.iter().take(2).any(|value| {
            matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
                || matches!(value, Value::GpuTensor(handle)
                    if runmat_accelerate_api::handle_integer_type(handle).is_none()
                        && !runmat_accelerate_api::handle_is_logical(handle)
                        && runmat_accelerate_api::handle_precision(handle)
                            == Some(runmat_accelerate_api::ProviderPrecision::F32))
        });
        let source = gpu_helpers::select_resident_output_source(
            args.iter().take(2).filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            }),
            NAME,
        )?;
        Ok(Self { single, source })
    }

    fn finish(&self, data: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
        let host = if self.single {
            Tensor::from_f32(data.into_iter().map(|value| value as f32).collect(), shape)
                .map(Value::Tensor)
                .map_err(|err| internal(format!("mvnrnd: {err}")))?
        } else {
            Tensor::new(data, shape)
                .map(tensor::tensor_into_value)
                .map_err(|err| internal(format!("mvnrnd: {err}")))?
        };
        match &self.source {
            Some(source) => {
                let restored = gpu_helpers::restore_class_preserving_value(source, host, NAME)?;
                if runmat_accelerate_api::handle_is_explicit(source)
                    && !matches!(restored, Value::GpuTensor(_))
                {
                    return Err(internal(
                        "mvnrnd: provider cannot preserve explicit gpuArray output residency",
                    ));
                }
                Ok(restored)
            }
            None => Ok(host),
        }
    }
}

fn ensure_extensions(args: &[Value]) -> BuiltinResult<()> {
    if args.first().is_some_and(is_typed_integer) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_MU_EXTENSION, NAME)?;
    }
    if args.get(1).is_some_and(is_typed_integer) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_SIGMA_EXTENSION, NAME)?;
    }
    if args.get(2).is_some_and(is_typed_integer) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_N_EXTENSION, NAME)?;
    }
    if args.iter().any(is_logical) {
        crate::compatibility::ensure_builtin_extension_enabled(&LOGICAL_INPUT_EXTENSION, NAME)?;
    }
    if matches!(args.get(2), Some(Value::GpuTensor(handle)) if runmat_accelerate_api::handle_is_explicit(handle))
    {
        crate::compatibility::ensure_builtin_extension_enabled(&RESIDENT_N_EXTENSION, NAME)?;
    }
    Ok(())
}

fn is_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

struct ParsedArgs {
    means: Vec<Vec<f64>>,
    factors: Vec<DMatrix<f64>>,
    sample_count: usize,
    dimension: usize,
    use_single_mean: bool,
    use_single_factor: bool,
}

impl ParsedArgs {
    fn mean_index(&self, sample: usize) -> usize {
        if self.use_single_mean {
            0
        } else {
            sample
        }
    }

    fn factor_index(&self, sample: usize) -> usize {
        if self.use_single_factor {
            0
        } else {
            sample
        }
    }
}

async fn parse_args(args: Vec<Value>) -> BuiltinResult<ParsedArgs> {
    if !(2..=3).contains(&args.len()) {
        return Err(invalid(
            "mvnrnd: expected mvnrnd(mu, Sigma) or mvnrnd(mu, Sigma, n)",
        ));
    }
    let mu = value_to_tensor(&args[0]).await?;
    let sigma = value_to_tensor(&args[1]).await?;
    let means = parse_means(&mu)?;
    let dimension = means
        .first()
        .map(Vec::len)
        .ok_or_else(|| invalid("mvnrnd: mu must contain at least one mean vector"))?;
    let covariances = parse_covariances(&sigma, dimension)?;
    let explicit_n = if args.len() == 3 {
        Some(parse_n(&args[2]).await?)
    } else {
        None
    };
    let sample_count = match explicit_n {
        Some(n) => {
            if means.len() != 1 {
                return Err(invalid(
                    "mvnrnd: n can be supplied only when mu is a single mean vector",
                ));
            }
            if covariances.len() != 1 {
                return Err(invalid(
                    "mvnrnd: n can be supplied only when Sigma has a single page",
                ));
            }
            n
        }
        None => means.len().max(covariances.len()),
    };
    validate_factor_work(dimension, covariances.len())?;
    let use_single_mean = means.len() == 1;
    let use_single_factor = covariances.len() == 1;
    if !use_single_mean && means.len() != sample_count {
        return Err(invalid("mvnrnd: mu rows must match Sigma pages"));
    }
    if !use_single_factor && covariances.len() != sample_count {
        return Err(invalid("mvnrnd: Sigma pages must match mu rows"));
    }
    let factors = covariances
        .into_iter()
        .map(|cov| covariance_factor(cov, dimension))
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(ParsedArgs {
        means,
        factors,
        sample_count,
        dimension,
        use_single_mean,
        use_single_factor,
    })
}

async fn value_to_tensor(value: &Value) -> BuiltinResult<Tensor> {
    let gathered = gather_if_needed_async(value)
        .await
        .map_err(|err| invalid(format!("mvnrnd: {err}")))?;
    let tensor = tensor::value_into_tensor_for(NAME, gathered)
        .map_err(|err| invalid(format!("mvnrnd: {err}")))?;
    ensure_exact_integer_tensor(&tensor)?;
    Ok(tensor)
}

fn ensure_exact_integer_tensor(tensor: &Tensor) -> BuiltinResult<()> {
    if tensor.integer_storage().is_none() {
        return Ok(());
    }
    const MAX_EXACT_INTEGER: i128 = 1_i128 << 53;
    for index in 0..tensor.len() {
        let exact = match tensor.numeric_value_at(index) {
            Some(NumericScalar::I8(value)) => i128::from(value),
            Some(NumericScalar::I16(value)) => i128::from(value),
            Some(NumericScalar::I32(value)) => i128::from(value),
            Some(NumericScalar::I64(value)) => i128::from(value),
            Some(NumericScalar::U8(value)) => i128::from(value),
            Some(NumericScalar::U16(value)) => i128::from(value),
            Some(NumericScalar::U32(value)) => i128::from(value),
            Some(NumericScalar::U64(value)) => i128::from(value),
            _ => continue,
        };
        if !(-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&exact) {
            return Err(invalid(
                "mvnrnd: integer distribution values must be exactly representable as double",
            ));
        }
    }
    Ok(())
}

fn parse_means(mu: &Tensor) -> BuiltinResult<Vec<Vec<f64>>> {
    if mu.shape.len() > 2 {
        return Err(invalid("mvnrnd: mu must be a vector or 2-D matrix"));
    }
    let values = tensor::tensor_values_f64(mu);
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid("mvnrnd: mu values must be finite"));
    }
    match mu.shape.as_slice() {
        [] => Ok(vec![vec![values[0]]]),
        [len] => Ok(vec![values[..*len].to_vec()]),
        [rows, cols] if *rows == 1 => Ok(vec![(0..*cols).map(|col| values[col]).collect()]),
        [rows, cols] => {
            let mut out = Vec::with_capacity(*rows);
            for row in 0..*rows {
                out.push((0..*cols).map(|col| values[col * rows + row]).collect());
            }
            Ok(out)
        }
        _ => Err(invalid("mvnrnd: mu must be a vector or 2-D matrix")),
    }
}

fn parse_covariances(sigma: &Tensor, dimension: usize) -> BuiltinResult<Vec<DMatrix<f64>>> {
    let values = tensor::tensor_values_f64(sigma);
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid("mvnrnd: Sigma values must be finite"));
    }
    match sigma.shape.as_slice() {
        [] if dimension == 1 => Ok(vec![DMatrix::from_element(1, 1, values[0])]),
        [1] if dimension == 1 => Ok(vec![DMatrix::from_element(1, 1, values[0])]),
        [rows, cols] if *rows == 1 && *cols == dimension => {
            Ok(vec![diagonal_page(&values[..dimension], dimension)])
        }
        [rows, cols] => {
            if *rows != dimension || *cols != dimension {
                return Err(invalid("mvnrnd: Sigma must be d-by-d for d columns in mu"));
            }
            Ok(vec![matrix_page(&values, dimension, 0)])
        }
        [rows, cols, pages] => {
            if *rows == 1 && *cols == dimension {
                let mut out = Vec::with_capacity(*pages);
                for page in 0..*pages {
                    let offset = page * dimension;
                    out.push(diagonal_page(
                        &values[offset..offset + dimension],
                        dimension,
                    ));
                }
                return Ok(out);
            }
            if *rows != dimension || *cols != dimension {
                return Err(invalid("mvnrnd: Sigma pages must be d-by-d-by-m"));
            }
            let mut out = Vec::with_capacity(*pages);
            for page in 0..*pages {
                out.push(matrix_page(&values, dimension, page));
            }
            Ok(out)
        }
        _ => Err(invalid("mvnrnd: Sigma must be a matrix or 3-D page array")),
    }
}

fn matrix_page(data: &[f64], dimension: usize, page: usize) -> DMatrix<f64> {
    let offset = page * dimension * dimension;
    DMatrix::from_column_slice(
        dimension,
        dimension,
        &data[offset..offset + dimension * dimension],
    )
}

fn diagonal_page(diagonal: &[f64], dimension: usize) -> DMatrix<f64> {
    let mut matrix = DMatrix::<f64>::zeros(dimension, dimension);
    for idx in 0..dimension {
        matrix[(idx, idx)] = diagonal[idx];
    }
    matrix
}

async fn parse_n(value: &Value) -> BuiltinResult<usize> {
    if let Value::Int(value) = value {
        return value
            .try_to_usize()
            .filter(|value| *value > 0)
            .ok_or_else(|| invalid("mvnrnd: n must be a positive scalar integer"));
    }
    let tensor = value_to_tensor(value).await?;
    if !tensor::is_scalar_tensor(&tensor) {
        return Err(invalid("mvnrnd: n must be a positive scalar integer"));
    }
    let value = tensor::tensor_value_f64(&tensor, 0);
    if !value.is_finite()
        || value <= 0.0
        || value.fract() != 0.0
        || value > usize::MAX as f64
        || (usize::BITS == 64 && value == usize::MAX as f64)
    {
        return Err(invalid("mvnrnd: n must be a positive scalar integer"));
    }
    Ok(value as usize)
}

fn validate_factor_work(dimension: usize, pages: usize) -> BuiltinResult<()> {
    let factor_cells = dimension
        .checked_mul(dimension)
        .and_then(|value| value.checked_mul(pages))
        .ok_or_else(|| internal("mvnrnd: covariance factor size overflow"))?;
    if factor_cells > MAX_FACTOR_CELLS {
        return Err(invalid("mvnrnd: covariance factor work is too large"));
    }
    Ok(())
}

fn covariance_factor(cov: DMatrix<f64>, dimension: usize) -> BuiltinResult<DMatrix<f64>> {
    validate_symmetric(&cov)?;
    if dimension == 0 {
        return Ok(DMatrix::zeros(0, 0));
    }
    if let Some(chol) = cov.clone().cholesky() {
        return Ok(chol.l());
    }
    let eigen = SymmetricEigen::new(cov);
    let tolerance = scaled_psd_tolerance(&eigen.eigenvalues);
    let mut factor = DMatrix::<f64>::zeros(dimension, dimension);
    for idx in 0..dimension {
        let lambda = eigen.eigenvalues[idx];
        if lambda < -tolerance {
            return Err(invalid(
                "mvnrnd: Sigma must be symmetric positive semidefinite",
            ));
        }
        let scale = lambda.max(0.0).sqrt();
        for row in 0..dimension {
            factor[(row, idx)] = eigen.eigenvectors[(row, idx)] * scale;
        }
    }
    Ok(factor)
}

fn scaled_psd_tolerance(eigenvalues: &nalgebra::DVector<f64>) -> f64 {
    let scale = eigenvalues
        .iter()
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    PSD_TOL * scale * eigenvalues.len().max(1) as f64
}

fn validate_symmetric(cov: &DMatrix<f64>) -> BuiltinResult<()> {
    if cov.nrows() != cov.ncols() {
        return Err(invalid("mvnrnd: Sigma must be square"));
    }
    for row in 0..cov.nrows() {
        for col in 0..cov.ncols() {
            let a = cov[(row, col)];
            let b = cov[(col, row)];
            let scale = a.abs().max(b.abs()).max(1.0);
            if (a - b).abs() > SYMMETRY_TOL * scale {
                return Err(invalid(
                    "mvnrnd: Sigma must be symmetric positive semidefinite",
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::random;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

    fn reset_rng() -> std::sync::MutexGuard<'static, ()> {
        let guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        guard
    }

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>, _poison: f64) -> Tensor {
        let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        tensor
    }

    fn cleared_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
        let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        tensor
    }

    #[test]
    fn mvnrnd_vector_mean_single_covariance_with_n() {
        let _guard = reset_rng();
        let out = block_on(mvnrnd_builtin(vec![
            tensor(vec![1.0, 2.0], vec![1, 2]),
            tensor(vec![4.0, 0.0, 0.0, 9.0], vec![2, 2]),
            Value::Num(5.0),
        ]))
        .expect("mvnrnd");
        match out {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![5, 2]);
                assert!(t.materialize_f64().iter().all(|value| value.is_finite()));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn mvnrnd_identity_covariance_uses_column_major_normal_stream() {
        let _guard = reset_rng();
        let out = block_on(mvnrnd_builtin(vec![
            tensor(vec![0.0, 0.0], vec![1, 2]),
            tensor(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            Value::Num(3.0),
        ]))
        .expect("mvnrnd");
        let expected = random::expected_normal_scaled_sequence(0.0, 1.0, 6);
        match out {
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), expected),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn mvnrnd_matrix_means_reuse_single_covariance() {
        let _guard = reset_rng();
        let out = block_on(mvnrnd_builtin(vec![
            tensor(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0], vec![3, 2]),
            tensor(vec![1.0, 0.25, 0.25, 1.0], vec![2, 2]),
        ]))
        .expect("mvnrnd");
        match out {
            Value::Tensor(t) => assert_eq!(t.shape, vec![3, 2]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn mvnrnd_column_mu_is_multiple_univariate_means() {
        let _guard = reset_rng();
        let out = block_on(mvnrnd_builtin(vec![
            tensor(vec![1.0, 10.0, 100.0], vec![3, 1]),
            Value::Num(4.0),
        ]))
        .expect("mvnrnd");
        match out {
            Value::Tensor(t) => assert_eq!(t.shape, vec![3, 1]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn mvnrnd_accepts_covariance_pages() {
        let _guard = reset_rng();
        let sigma = tensor(
            vec![
                1.0, 0.0, 0.0, 1.0, //
                4.0, 0.0, 0.0, 9.0,
            ],
            vec![2, 2, 2],
        );
        let out = block_on(mvnrnd_builtin(vec![
            tensor(vec![0.0, 10.0, 0.0, 20.0], vec![2, 2]),
            sigma,
        ]))
        .expect("mvnrnd");
        match out {
            Value::Tensor(t) => assert_eq!(t.shape, vec![2, 2]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn mvnrnd_accepts_diagonal_covariance_vector_and_pages() {
        let _guard = reset_rng();
        let out = block_on(mvnrnd_builtin(vec![
            tensor(vec![0.0, 10.0, 0.0, 20.0], vec![2, 2]),
            tensor(vec![1.0, 4.0, 9.0, 16.0], vec![1, 2, 2]),
        ]))
        .expect("mvnrnd");
        match out {
            Value::Tensor(t) => assert_eq!(t.shape, vec![2, 2]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn mvnrnd_accepts_semidefinite_covariance() {
        let _guard = reset_rng();
        let out = block_on(mvnrnd_builtin(vec![
            tensor(vec![0.0, 0.0], vec![1, 2]),
            tensor(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]),
            Value::Num(3.0),
        ]))
        .expect("mvnrnd");
        match out {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 2]);
                for row in 0..3 {
                    assert!(
                        (t.materialize_f64()[row] - t.materialize_f64()[3 + row]).abs() < 1.0e-10
                    );
                }
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn mvnrnd_rejects_bad_covariance() {
        let err = block_on(mvnrnd_builtin(vec![
            tensor(vec![0.0, 0.0], vec![1, 2]),
            tensor(vec![1.0, 2.0, 3.0, 1.0], vec![2, 2]),
        ]))
        .expect_err("non-symmetric covariance should fail");
        assert_eq!(err.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn mvnrnd_rejects_n_with_matrix_mu() {
        let err = block_on(mvnrnd_builtin(vec![
            tensor(vec![0.0, 0.0, 1.0, 1.0], vec![2, 2]),
            tensor(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]),
            Value::Num(2.0),
        ]))
        .expect_err("n with matrix mu should fail");
        assert_eq!(err.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn mvnrnd_rejects_zero_n() {
        let err = block_on(mvnrnd_builtin(vec![
            Value::Num(0.0),
            Value::Num(1.0),
            Value::Num(0.0),
        ]))
        .expect_err("zero n should fail");
        assert_eq!(err.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn mvnrnd_typed_count_is_exact_and_lossy_f64_is_rejected() {
        assert_eq!(
            block_on(parse_n(&Value::Int(runmat_builtins::IntValue::U16(3)))).unwrap(),
            3
        );
        for value in [
            Value::Int(runmat_builtins::IntValue::I8(-1)),
            Value::Num(1.5),
            Value::Num(usize::MAX as f64 + 1.0),
        ] {
            assert!(block_on(parse_n(&value)).is_err());
        }
    }

    #[test]
    fn mvnrnd_parsers_read_typed_integer_tensor_storage_exactly() {
        let n = cleared_int_tensor(IntegerStorage::U16(vec![4]), vec![1, 1]);
        assert_eq!(block_on(parse_n(&Value::Tensor(n))).unwrap(), 4);

        let mu = poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 3, 4]), vec![2, 2], f64::NAN);
        assert_eq!(
            parse_means(&mu).unwrap(),
            vec![vec![1.0, 3.0], vec![2.0, 4.0]]
        );

        let sigma =
            poisoned_int_tensor(IntegerStorage::U16(vec![4, 0, 0, 9]), vec![2, 2], f64::NAN);
        let covariances = parse_covariances(&sigma, 2).unwrap();
        assert_eq!(covariances.len(), 1);
        assert_eq!(covariances[0][(0, 0)], 4.0);
        assert_eq!(covariances[0][(1, 1)], 9.0);
        assert_eq!(covariances[0][(0, 1)], 0.0);
        assert_eq!(covariances[0][(1, 0)], 0.0);
    }

    #[test]
    fn mvnrnd_integer_roles_gate_and_wide_distribution_values_reject() {
        {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(mvnrnd_builtin(vec![
                Value::Int(runmat_builtins::IntValue::I16(0)),
                Value::Num(1.0),
            ]))
            .expect_err("integer mu must gate");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:MvnrndIntegerMuExtension")
            );
        }
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let wide = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]).unwrap(),
        );
        let error = block_on(mvnrnd_builtin(vec![wide, Value::Num(1.0)]))
            .expect_err("wide integer mu must reject");
        assert!(error.message.contains("exactly representable"));
    }

    #[test]
    fn mvnrnd_single_distribution_input_selects_single_output() {
        let _guard = reset_rng();
        let mu = Value::Tensor(Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap());
        let out = block_on(mvnrnd_builtin(vec![mu, Value::Num(1.0)])).unwrap();
        let Value::Tensor(out) = out else {
            panic!("expected single tensor output");
        };
        assert_eq!(out.numeric_dtype(), NumericDType::F32);
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn mvnrnd_wgpu_integer_fallback_preserves_class_and_explicit_intent() {
        let _guard = reset_rng();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("wgpu provider");
        let mu = Tensor::new_integer(IntegerStorage::I16(vec![0, 0]), vec![1, 2]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &mu).expect("integer upload");
        let sigma = Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap());
        let out = block_on(mvnrnd_builtin(vec![
            Value::GpuTensor(handle.clone()),
            sigma.clone(),
            Value::Num(2.0),
        ]))
        .expect("mvnrnd");
        let Value::Tensor(host) = out else {
            panic!("F32 owner cannot relabel required double output");
        };
        assert_eq!(host.numeric_dtype(), NumericDType::F64);
        assert_eq!(host.shape, vec![2, 2]);
        runmat_accelerate_api::set_handle_provenance(
            &handle,
            runmat_accelerate_api::GpuHandleProvenance::Explicit,
        );
        let error = block_on(mvnrnd_builtin(vec![
            Value::GpuTensor(handle),
            sigma,
            Value::Num(2.0),
        ]))
        .expect_err("explicit output class mismatch must reject");
        assert!(error.message.contains("cannot preserve explicit gpuArray"));
    }
}
