//! Multivariate normal random variates.

use nalgebra::{DMatrix, SymmetricEigen};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "mvnrnd";
const SYMMETRY_TOL: f64 = 1.0e-10;
const PSD_TOL: f64 = 1.0e-10;
const MAX_OUTPUT_CELLS: usize = 20_000_000;
const MAX_FACTOR_CELLS: usize = 20_000_000;

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
    builtin_path = "crate::builtins::stats::random::mvnrnd"
)]
async fn mvnrnd_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
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
    Tensor::new(out, vec![parsed.sample_count, parsed.dimension])
        .map(tensor::tensor_into_value)
        .map_err(|err| internal(format!("mvnrnd: {err}")))
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
    tensor::value_into_tensor_for(NAME, gathered).map_err(|err| invalid(format!("mvnrnd: {err}")))
}

fn parse_means(mu: &Tensor) -> BuiltinResult<Vec<Vec<f64>>> {
    if mu.shape.len() > 2 {
        return Err(invalid("mvnrnd: mu must be a vector or 2-D matrix"));
    }
    if mu.data.iter().any(|value| !value.is_finite()) {
        return Err(invalid("mvnrnd: mu values must be finite"));
    }
    match mu.shape.as_slice() {
        [] => Ok(vec![vec![mu.data[0]]]),
        [len] => Ok(vec![mu.data[..*len].to_vec()]),
        [rows, cols] if *rows == 1 => Ok(vec![(0..*cols).map(|col| mu.data[col]).collect()]),
        [rows, cols] => {
            let mut out = Vec::with_capacity(*rows);
            for row in 0..*rows {
                out.push((0..*cols).map(|col| mu.data[col * rows + row]).collect());
            }
            Ok(out)
        }
        _ => Err(invalid("mvnrnd: mu must be a vector or 2-D matrix")),
    }
}

fn parse_covariances(sigma: &Tensor, dimension: usize) -> BuiltinResult<Vec<DMatrix<f64>>> {
    if sigma.data.iter().any(|value| !value.is_finite()) {
        return Err(invalid("mvnrnd: Sigma values must be finite"));
    }
    match sigma.shape.as_slice() {
        [] if dimension == 1 => Ok(vec![DMatrix::from_element(1, 1, sigma.data[0])]),
        [1] if dimension == 1 => Ok(vec![DMatrix::from_element(1, 1, sigma.data[0])]),
        [rows, cols] if *rows == 1 && *cols == dimension => {
            Ok(vec![diagonal_page(&sigma.data[..dimension], dimension)])
        }
        [rows, cols] => {
            if *rows != dimension || *cols != dimension {
                return Err(invalid("mvnrnd: Sigma must be d-by-d for d columns in mu"));
            }
            Ok(vec![matrix_page(&sigma.data, dimension, 0)])
        }
        [rows, cols, pages] => {
            if *rows == 1 && *cols == dimension {
                let mut out = Vec::with_capacity(*pages);
                for page in 0..*pages {
                    let offset = page * dimension;
                    out.push(diagonal_page(
                        &sigma.data[offset..offset + dimension],
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
                out.push(matrix_page(&sigma.data, dimension, page));
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
    let tensor = value_to_tensor(value).await?;
    if tensor.data.len() != 1 {
        return Err(invalid("mvnrnd: n must be a positive scalar integer"));
    }
    let value = tensor.data[0];
    if !value.is_finite() || value <= 0.0 || value.fract() != 0.0 {
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

    fn reset_rng() -> std::sync::MutexGuard<'static, ()> {
        let guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        guard
    }

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
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
                assert!(t.data.iter().all(|value| value.is_finite()));
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
            Value::Tensor(t) => assert_eq!(t.data, expected),
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
                    assert!((t.data[row] - t.data[3 + row]).abs() < 1.0e-10);
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
}
