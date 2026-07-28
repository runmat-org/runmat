//! Linear least-squares with observation covariance weighting.

use nalgebra::{DMatrix, SymmetricEigen};
use num_complex::Complex64;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "lscov";
const EPS: f64 = 1.0e-12;
const MAX_LSCOV_CELLS: usize = 50_000_000;

const OUTPUT_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Least-squares coefficient estimates.",
}];

const OUTPUT_X_STDX: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Least-squares coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "stdx",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard errors for the coefficient estimates.",
    },
];

const OUTPUT_X_STDX_MSE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Least-squares coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "stdx",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard errors for the coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "mse",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Mean squared weighted residual error for each response.",
    },
];

const OUTPUT_X_STDX_MSE_S: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Least-squares coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "stdx",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard errors for the coefficient estimates.",
    },
    BuiltinParamDescriptor {
        name: "mse",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Mean squared weighted residual error.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Estimated covariance matrix for the coefficient estimates.",
    },
];

const PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Design matrix with observations in rows.",
};

const PARAM_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Observation vector or matrix with one row per observation.",
};

const PARAM_V: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "V",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: Some("eye(size(A,1))"),
    description: "Observation covariance matrix or vector of observation weights.",
};

const PARAM_ALG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "alg",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("chol"),
    description: "Algorithm selector, either \"orth\" or \"chol\".",
};

const INPUTS_A_B: [BuiltinParamDescriptor; 2] = [PARAM_A, PARAM_B];
const INPUTS_A_B_V: [BuiltinParamDescriptor; 3] = [PARAM_A, PARAM_B, PARAM_V];
const INPUTS_A_B_V_ALG: [BuiltinParamDescriptor; 4] = [PARAM_A, PARAM_B, PARAM_V, PARAM_ALG];

const SIGNATURES: [BuiltinSignatureDescriptor; 12] = [
    BuiltinSignatureDescriptor {
        label: "x = lscov(A, B)",
        inputs: &INPUTS_A_B,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = lscov(A, B, V)",
        inputs: &INPUTS_A_B_V,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = lscov(A, B, V, alg)",
        inputs: &INPUTS_A_B_V_ALG,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx] = lscov(A, B)",
        inputs: &INPUTS_A_B,
        outputs: &OUTPUT_X_STDX,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx] = lscov(A, B, V)",
        inputs: &INPUTS_A_B_V,
        outputs: &OUTPUT_X_STDX,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx] = lscov(A, B, V, alg)",
        inputs: &INPUTS_A_B_V_ALG,
        outputs: &OUTPUT_X_STDX,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx, mse] = lscov(A, B)",
        inputs: &INPUTS_A_B,
        outputs: &OUTPUT_X_STDX_MSE,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx, mse] = lscov(A, B, V)",
        inputs: &INPUTS_A_B_V,
        outputs: &OUTPUT_X_STDX_MSE,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx, mse] = lscov(A, B, V, alg)",
        inputs: &INPUTS_A_B_V_ALG,
        outputs: &OUTPUT_X_STDX_MSE,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx, mse, S] = lscov(A, b)",
        inputs: &INPUTS_A_B,
        outputs: &OUTPUT_X_STDX_MSE_S,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx, mse, S] = lscov(A, b, V)",
        inputs: &INPUTS_A_B_V,
        outputs: &OUTPUT_X_STDX_MSE_S,
    },
    BuiltinSignatureDescriptor {
        label: "[x, stdx, mse, S] = lscov(A, b, V, alg)",
        inputs: &INPUTS_A_B_V_ALG,
        outputs: &OUTPUT_X_STDX_MSE_S,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSCOV.INVALID_ARGUMENT",
    identifier: Some("RunMat:lscov:InvalidArgument"),
    when: "Inputs, dimensions, weighting arguments, algorithm, or requested output count are malformed.",
    message: "lscov: invalid argument",
};

const ERROR_NUMERICAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSCOV.NUMERICAL",
    identifier: Some("RunMat:lscov:Numerical"),
    when: "The weighted least-squares system cannot be solved numerically.",
    message: "lscov: numerical failure",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LSCOV.INTERNAL",
    identifier: Some("RunMat:lscov:Internal"),
    when: "RunMat cannot allocate or construct lscov outputs.",
    message: "lscov: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INVALID_ARGUMENT, ERROR_NUMERICAL, ERROR_INTERNAL];

pub const LSCOV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn lscov_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn lscov_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    lscov_error(message, &ERROR_INVALID_ARGUMENT)
}

fn numerical(message: impl Into<String>) -> RuntimeError {
    lscov_error(message, &ERROR_NUMERICAL)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    lscov_error(message, &ERROR_INTERNAL)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Algorithm {
    Orth,
    Chol,
}

#[derive(Clone, Debug)]
enum NumericMatrix {
    Real(Tensor),
    Complex(ComplexTensor),
}

#[derive(Clone, Debug)]
enum Weighting {
    Identity,
    Weights(Vec<f64>),
    Covariance(DMatrix<f64>),
}

#[derive(Clone, Debug)]
struct ParsedArgs {
    weighting: Weighting,
    algorithm: Algorithm,
}

#[runtime_builtin(
    name = "lscov",
    category = "stats/ml",
    summary = "Solve linear least-squares systems with observation covariance weighting.",
    keywords = "lscov,least squares,weighted least squares,generalized least squares,statistics",
    type_resolver(lscov_type),
    descriptor(crate::builtins::stats::ml::lscov::LSCOV_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::ml::lscov"
)]
async fn lscov_builtin(a: Value, b: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 2 {
        return Err(invalid("lscov: accepts at most four input arguments"));
    }
    let a = gather_numeric_matrix(a).await?;
    let b = gather_numeric_matrix(b).await?;
    let parsed = parse_rest(rest, matrix_rows(&a)).await?;

    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(out_count) if out_count > 4 => Err(invalid("lscov: too many output arguments")),
        Some(out_count) => {
            let outputs = lscov_compute(a, b, parsed, out_count)?;
            Ok(crate::output_count::output_list_with_padding(
                out_count, outputs,
            ))
        }
        None => Ok(lscov_compute(a, b, parsed, 1)?
            .into_iter()
            .next()
            .expect("lscov always returns x for scalar-output calls")),
    }
}

async fn gather_numeric_matrix(value: Value) -> BuiltinResult<NumericMatrix> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid(format!("lscov: {err}")))?;
    match gathered {
        Value::ComplexTensor(tensor) => Ok(NumericMatrix::Complex(tensor)),
        Value::Complex(re, im) => ComplexTensor::new(vec![(re, im)], vec![1, 1])
            .map(NumericMatrix::Complex)
            .map_err(|err| invalid(format!("lscov: {err}"))),
        other => {
            let tensor = tensor::value_into_tensor_for(NAME, other)
                .map_err(|err| invalid(format!("lscov: {err}")))?;
            Ok(NumericMatrix::Real(tensor))
        }
    }
}

async fn parse_rest(rest: Vec<Value>, rows: usize) -> BuiltinResult<ParsedArgs> {
    let mut weighting_arg = None;
    let mut algorithm = Algorithm::Chol;
    match rest.as_slice() {
        [] => {}
        [single] => {
            if let Some(keyword) = keyword_of(single) {
                algorithm = parse_algorithm(&keyword)?;
            } else {
                weighting_arg = Some(single.clone());
            }
        }
        [weighting, alg] => {
            weighting_arg = Some(weighting.clone());
            let keyword = keyword_of(alg).ok_or_else(|| {
                invalid("lscov: algorithm must be a string scalar or character row")
            })?;
            algorithm = parse_algorithm(&keyword)?;
        }
        _ => unreachable!("rest length checked by caller"),
    }
    let weighting = match weighting_arg {
        Some(value) => parse_weighting(value, rows, algorithm).await?,
        None => Weighting::Identity,
    };
    Ok(ParsedArgs {
        weighting,
        algorithm,
    })
}

fn parse_algorithm(keyword: &str) -> BuiltinResult<Algorithm> {
    match keyword {
        "orth" => Ok(Algorithm::Orth),
        "chol" => Ok(Algorithm::Chol),
        _ => Err(invalid("lscov: algorithm must be \"orth\" or \"chol\"")),
    }
}

async fn parse_weighting(
    value: Value,
    rows: usize,
    _algorithm: Algorithm,
) -> BuiltinResult<Weighting> {
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid(format!("lscov: {err}")))?;
    let tensor = tensor::value_into_tensor_for(NAME, gathered)
        .map_err(|err| invalid(format!("lscov: {err}")))?;
    if tensor::tensor_element_len(&tensor) == 0 {
        return Ok(Weighting::Identity);
    }
    let values = tensor::tensor_values_f64(&tensor);
    if is_vector(&tensor) {
        let weights = if values.len() == 1 && rows > 1 {
            vec![values[0]; rows]
        } else {
            values
        };
        if weights.len() != rows {
            return Err(invalid(
                "lscov: V vector length must match the number of rows in A",
            ));
        }
        if !weights.iter().any(|weight| *weight > 0.0) {
            return Err(invalid(
                "lscov: V weights must include at least one positive value",
            ));
        }
        for weight in &weights {
            if !weight.is_finite() || *weight < 0.0 {
                return Err(invalid(
                    "lscov: V weights must be finite nonnegative values",
                ));
            }
        }
        return Ok(Weighting::Weights(weights));
    }
    if tensor.shape.len() > 2 || tensor.rows != rows || tensor.cols != rows {
        return Err(invalid(
            "lscov: V must be empty, a weight vector, or a square covariance matrix matching A rows",
        ));
    }
    ensure_budget(rows, rows, "covariance matrix")?;
    let cov = DMatrix::from_column_slice(rows, rows, &values);
    validate_symmetric_covariance(&cov)?;
    Ok(Weighting::Covariance(cov))
}

fn lscov_compute(
    a: NumericMatrix,
    b: NumericMatrix,
    parsed: ParsedArgs,
    requested_outputs: usize,
) -> BuiltinResult<Vec<Value>> {
    let rows = matrix_rows(&a);
    let cols = matrix_cols(&a);
    if rows == 0 || cols == 0 {
        return Err(invalid("lscov: A must be a nonempty 2-D matrix"));
    }
    ensure_2d(&a, "A")?;
    ensure_2d(&b, "B")?;
    ensure_budget(rows, cols, "design matrix")?;
    let rhs_cols = rhs_columns(&b, rows)?;
    ensure_work_budget(cols)?;
    ensure_output_budget(cols, rhs_cols, requested_outputs)?;
    if requested_outputs == 4 && rhs_cols != 1 {
        return Err(invalid(
            "lscov: fourth output S is supported only when B is a vector",
        ));
    }
    match (a, b) {
        (NumericMatrix::Real(a), NumericMatrix::Real(b)) => {
            lscov_real(a, b, parsed, rhs_cols, requested_outputs)
        }
        (a, b) => lscov_complex(a, b, parsed, rhs_cols, requested_outputs),
    }
}

fn lscov_real(
    a: Tensor,
    b: Tensor,
    parsed: ParsedArgs,
    rhs_cols: usize,
    requested_outputs: usize,
) -> BuiltinResult<Vec<Value>> {
    let rows = a.rows;
    let cols = a.cols;
    let a_values = tensor::tensor_values_f64_cow(&a);
    let a_mat = DMatrix::from_column_slice(rows, cols, a_values.as_ref());
    let b_mat = real_rhs_matrix(&b, rows, rhs_cols)?;
    let transformed = transform_real_problem(&a_mat, &b_mat, &parsed.weighting, parsed.algorithm)?;
    let solve = solve_real_least_squares(&transformed.a, &transformed.b, rows, cols)?;
    let transformed_residual = &transformed.b - &transformed.a * &solve.x;
    let mse = mse_real(&transformed_residual, rows, cols, solve.rank);

    let mut outputs = Vec::with_capacity(requested_outputs.max(1));
    outputs.push(real_tensor_value(
        solve.x.as_slice().to_vec(),
        vec![cols, rhs_cols],
        "x",
    )?);
    if requested_outputs == 1 {
        return Ok(outputs);
    }

    outputs.push(real_tensor_value(
        stdx_real(&solve.covariance_base, &mse, cols, rhs_cols),
        vec![cols, rhs_cols],
        "stdx",
    )?);
    if requested_outputs == 2 {
        return Ok(outputs);
    }

    outputs.push(real_tensor_value(mse.clone(), vec![1, rhs_cols], "mse")?);
    if requested_outputs == 3 {
        return Ok(outputs);
    }

    outputs.push(real_tensor_value(
        scaled_covariance_real(&solve.covariance_base, mse[0]),
        vec![cols, cols],
        "S",
    )?);
    Ok(outputs)
}

fn lscov_complex(
    a: NumericMatrix,
    b: NumericMatrix,
    parsed: ParsedArgs,
    rhs_cols: usize,
    requested_outputs: usize,
) -> BuiltinResult<Vec<Value>> {
    let rows = matrix_rows(&a);
    let cols = matrix_cols(&a);
    let a_mat = complex_matrix(&a)?;
    let b_mat = complex_rhs_matrix(&b, rows, rhs_cols)?;
    let transformed =
        transform_complex_problem(&a_mat, &b_mat, &parsed.weighting, parsed.algorithm)?;
    let solve = solve_complex_least_squares(&transformed.a, &transformed.b, rows, cols)?;
    let transformed_residual = &transformed.b - &transformed.a * &solve.x;
    let mse = mse_complex(&transformed_residual, rows, cols, solve.rank);

    let mut outputs = Vec::with_capacity(requested_outputs.max(1));
    outputs.push(complex_matrix_value(&solve.x, cols, rhs_cols, "x")?);
    if requested_outputs == 1 {
        return Ok(outputs);
    }

    outputs.push(real_tensor_value(
        stdx_complex(&solve.covariance_base, &mse, cols, rhs_cols),
        vec![cols, rhs_cols],
        "stdx",
    )?);
    if requested_outputs == 2 {
        return Ok(outputs);
    }

    outputs.push(real_tensor_value(mse.clone(), vec![1, rhs_cols], "mse")?);
    if requested_outputs == 3 {
        return Ok(outputs);
    }

    outputs.push(complex_matrix_value(
        &(solve.covariance_base * Complex64::new(mse[0], 0.0)),
        cols,
        cols,
        "S",
    )?);
    Ok(outputs)
}

#[derive(Debug)]
struct TransformedReal {
    a: DMatrix<f64>,
    b: DMatrix<f64>,
}

#[derive(Debug)]
struct TransformedComplex {
    a: DMatrix<Complex64>,
    b: DMatrix<Complex64>,
}

#[derive(Debug)]
struct RealSolve {
    x: DMatrix<f64>,
    covariance_base: DMatrix<f64>,
    rank: usize,
}

#[derive(Debug)]
struct ComplexSolve {
    x: DMatrix<Complex64>,
    covariance_base: DMatrix<Complex64>,
    rank: usize,
}

fn transform_real_problem(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    weighting: &Weighting,
    algorithm: Algorithm,
) -> BuiltinResult<TransformedReal> {
    match weighting {
        Weighting::Identity => Ok(TransformedReal {
            a: a.clone(),
            b: b.clone(),
        }),
        Weighting::Weights(weights) => {
            let rows = a.nrows();
            let cols = a.ncols();
            let rhs_cols = b.ncols();
            let mut transformed_a = DMatrix::<f64>::zeros(rows, cols);
            let mut transformed_b = DMatrix::<f64>::zeros(rows, rhs_cols);
            for row in 0..rows {
                let scale = weights[row].sqrt();
                for col in 0..cols {
                    transformed_a[(row, col)] = a[(row, col)] * scale;
                }
                for rhs_col in 0..rhs_cols {
                    transformed_b[(row, rhs_col)] = b[(row, rhs_col)] * scale;
                }
            }
            Ok(TransformedReal {
                a: transformed_a,
                b: transformed_b,
            })
        }
        Weighting::Covariance(covariance) => {
            let transform = covariance_transform(covariance, algorithm)?;
            Ok(TransformedReal {
                a: &transform * a,
                b: &transform * b,
            })
        }
    }
}

fn transform_complex_problem(
    a: &DMatrix<Complex64>,
    b: &DMatrix<Complex64>,
    weighting: &Weighting,
    algorithm: Algorithm,
) -> BuiltinResult<TransformedComplex> {
    match weighting {
        Weighting::Identity => Ok(TransformedComplex {
            a: a.clone(),
            b: b.clone(),
        }),
        Weighting::Weights(weights) => {
            let rows = a.nrows();
            let cols = a.ncols();
            let rhs_cols = b.ncols();
            let mut transformed_a = DMatrix::<Complex64>::zeros(rows, cols);
            let mut transformed_b = DMatrix::<Complex64>::zeros(rows, rhs_cols);
            for row in 0..rows {
                let scale = weights[row].sqrt();
                for col in 0..cols {
                    transformed_a[(row, col)] = a[(row, col)] * scale;
                }
                for rhs_col in 0..rhs_cols {
                    transformed_b[(row, rhs_col)] = b[(row, rhs_col)] * scale;
                }
            }
            Ok(TransformedComplex {
                a: transformed_a,
                b: transformed_b,
            })
        }
        Weighting::Covariance(covariance) => {
            let transform = covariance_transform(covariance, algorithm)?;
            let transform_complex = transform.map(|value| Complex64::new(value, 0.0));
            Ok(TransformedComplex {
                a: &transform_complex * a,
                b: &transform_complex * b,
            })
        }
    }
}

fn covariance_transform(
    covariance: &DMatrix<f64>,
    algorithm: Algorithm,
) -> BuiltinResult<DMatrix<f64>> {
    if matches!(algorithm, Algorithm::Chol) {
        if let Some(chol) = covariance.clone().cholesky() {
            let lower = chol.l();
            let identity = DMatrix::<f64>::identity(covariance.nrows(), covariance.ncols());
            if let Some(transform) = lower.lu().solve(&identity) {
                return Ok(transform);
            }
        }
    }
    covariance_orth_transform(covariance)
}

fn covariance_orth_transform(covariance: &DMatrix<f64>) -> BuiltinResult<DMatrix<f64>> {
    let eigen = SymmetricEigen::new(covariance.clone());
    let tolerance = scaled_psd_tolerance(eigen.eigenvalues.as_slice());
    let positive = eigen
        .eigenvalues
        .iter()
        .copied()
        .filter(|value| *value > tolerance)
        .count();
    if positive == 0 {
        return Err(numerical(
            "lscov: V covariance matrix must have at least one positive direction",
        ));
    }
    let mut transform = DMatrix::<f64>::zeros(positive, covariance.ncols());
    let mut out_row = 0usize;
    for eig_idx in 0..eigen.eigenvalues.len() {
        let lambda = eigen.eigenvalues[eig_idx];
        if lambda < -tolerance {
            return Err(invalid(
                "lscov: V covariance matrix must be positive semidefinite",
            ));
        }
        if lambda <= tolerance {
            continue;
        }
        let scale = 1.0 / lambda.sqrt();
        for col in 0..covariance.ncols() {
            transform[(out_row, col)] = eigen.eigenvectors[(col, eig_idx)] * scale;
        }
        out_row += 1;
    }
    Ok(transform)
}

fn solve_real_least_squares(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    original_rows: usize,
    original_cols: usize,
) -> BuiltinResult<RealSolve> {
    let cols = a.ncols();
    let rhs_cols = b.ncols();
    if a.nrows() == 0 {
        return Ok(RealSolve {
            x: DMatrix::zeros(cols, rhs_cols),
            covariance_base: DMatrix::zeros(cols, cols),
            rank: 0,
        });
    }
    let svd = a.clone().svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| numerical("lscov: SVD did not return left singular vectors"))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| numerical("lscov: SVD did not return right singular vectors"))?;
    let singular_values = svd.singular_values.as_slice().to_vec();
    let largest = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let tolerance = (original_rows.max(original_cols) as f64) * f64::EPSILON * largest.max(1.0);
    let mut x = DMatrix::<f64>::zeros(cols, rhs_cols);
    let mut covariance_base = DMatrix::<f64>::zeros(cols, cols);
    let mut rank = 0usize;
    for (idx, singular_value) in singular_values.iter().copied().enumerate() {
        if singular_value.abs() <= tolerance {
            continue;
        }
        rank += 1;
        for rhs_col in 0..rhs_cols {
            let projection = u.column(idx).dot(&b.column(rhs_col)) / singular_value;
            for row in 0..cols {
                x[(row, rhs_col)] += v_t[(idx, row)] * projection;
            }
        }
        let inv_s2 = 1.0 / (singular_value * singular_value);
        for row in 0..cols {
            for col in 0..cols {
                covariance_base[(row, col)] += v_t[(idx, row)] * v_t[(idx, col)] * inv_s2;
            }
        }
    }
    Ok(RealSolve {
        x,
        covariance_base,
        rank,
    })
}

fn solve_complex_least_squares(
    a: &DMatrix<Complex64>,
    b: &DMatrix<Complex64>,
    original_rows: usize,
    original_cols: usize,
) -> BuiltinResult<ComplexSolve> {
    let cols = a.ncols();
    let rhs_cols = b.ncols();
    if a.nrows() == 0 {
        return Ok(ComplexSolve {
            x: DMatrix::zeros(cols, rhs_cols),
            covariance_base: DMatrix::zeros(cols, cols),
            rank: 0,
        });
    }
    let svd = a.clone().svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| numerical("lscov: SVD did not return left singular vectors"))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| numerical("lscov: SVD did not return right singular vectors"))?;
    let singular_values = svd.singular_values.as_slice().to_vec();
    let largest = singular_values.iter().copied().fold(0.0_f64, f64::max);
    let tolerance = (original_rows.max(original_cols) as f64) * f64::EPSILON * largest.max(1.0);
    let mut x = DMatrix::<Complex64>::zeros(cols, rhs_cols);
    let mut covariance_base = DMatrix::<Complex64>::zeros(cols, cols);
    let mut rank = 0usize;
    for (idx, singular_value) in singular_values.iter().copied().enumerate() {
        if singular_value.abs() <= tolerance {
            continue;
        }
        rank += 1;
        for rhs_col in 0..rhs_cols {
            let projection = u.column(idx).dotc(&b.column(rhs_col)) / singular_value;
            for row in 0..cols {
                x[(row, rhs_col)] += v_t[(idx, row)].conj() * projection;
            }
        }
        let inv_s2 = 1.0 / (singular_value * singular_value);
        for row in 0..cols {
            for col in 0..cols {
                covariance_base[(row, col)] += v_t[(idx, row)].conj() * v_t[(idx, col)] * inv_s2;
            }
        }
    }
    Ok(ComplexSolve {
        x,
        covariance_base,
        rank,
    })
}

fn mse_real(residual: &DMatrix<f64>, rows: usize, cols: usize, rank: usize) -> Vec<f64> {
    let rhs_cols = residual.ncols();
    if rows < cols {
        return vec![0.0; rhs_cols];
    }
    let dfe = rows as f64 - rank as f64;
    if dfe <= 0.0 {
        return vec![f64::NAN; rhs_cols];
    }
    (0..rhs_cols)
        .map(|col| residual.column(col).dot(&residual.column(col)) / dfe)
        .collect()
}

fn mse_complex(residual: &DMatrix<Complex64>, rows: usize, cols: usize, rank: usize) -> Vec<f64> {
    let rhs_cols = residual.ncols();
    if rows < cols {
        return vec![0.0; rhs_cols];
    }
    let dfe = rows as f64 - rank as f64;
    if dfe <= 0.0 {
        return vec![f64::NAN; rhs_cols];
    }
    (0..rhs_cols)
        .map(|col| {
            residual
                .column(col)
                .iter()
                .map(|value| value.norm_sqr())
                .sum::<f64>()
                / dfe
        })
        .collect()
}

fn stdx_real(
    covariance_base: &DMatrix<f64>,
    mse: &[f64],
    cols: usize,
    rhs_cols: usize,
) -> Vec<f64> {
    let mut out = vec![f64::NAN; cols * rhs_cols];
    for rhs_col in 0..rhs_cols {
        for col in 0..cols {
            let variance = covariance_base[(col, col)] * mse[rhs_col];
            out[col + rhs_col * cols] = if variance >= 0.0 {
                variance.sqrt()
            } else {
                f64::NAN
            };
        }
    }
    out
}

fn stdx_complex(
    covariance_base: &DMatrix<Complex64>,
    mse: &[f64],
    cols: usize,
    rhs_cols: usize,
) -> Vec<f64> {
    let mut out = vec![f64::NAN; cols * rhs_cols];
    for rhs_col in 0..rhs_cols {
        for col in 0..cols {
            let variance = covariance_base[(col, col)].re * mse[rhs_col];
            out[col + rhs_col * cols] = if variance >= -EPS {
                variance.max(0.0).sqrt()
            } else {
                f64::NAN
            };
        }
    }
    out
}

fn scaled_covariance_real(covariance_base: &DMatrix<f64>, mse: f64) -> Vec<f64> {
    covariance_base.iter().map(|value| value * mse).collect()
}

fn real_rhs_matrix(tensor: &Tensor, rows: usize, rhs_cols: usize) -> BuiltinResult<DMatrix<f64>> {
    let values = tensor::tensor_values_f64_cow(tensor);
    if is_vector(tensor) && tensor::tensor_element_len(tensor) == rows {
        Ok(DMatrix::from_column_slice(rows, 1, values.as_ref()))
    } else if tensor.rows == rows && tensor.cols == rhs_cols {
        Ok(DMatrix::from_column_slice(rows, rhs_cols, values.as_ref()))
    } else {
        Err(invalid(
            "lscov: B must have one row per observation or be an observation vector",
        ))
    }
}

fn complex_matrix(value: &NumericMatrix) -> BuiltinResult<DMatrix<Complex64>> {
    match value {
        NumericMatrix::Real(tensor) => {
            let values = tensor::tensor_values_f64(tensor)
                .into_iter()
                .map(|value| Complex64::new(value, 0.0))
                .collect::<Vec<_>>();
            Ok(DMatrix::from_column_slice(
                tensor.rows,
                tensor.cols,
                &values,
            ))
        }
        NumericMatrix::Complex(tensor) => Ok(DMatrix::from_column_slice(
            tensor.rows,
            tensor.cols,
            &tensor::complex_tensor_values_complex64(tensor),
        )),
    }
}

fn complex_rhs_matrix(
    value: &NumericMatrix,
    rows: usize,
    rhs_cols: usize,
) -> BuiltinResult<DMatrix<Complex64>> {
    let matrix = complex_matrix(value)?;
    if matrix.nrows() == rows && matrix.ncols() == rhs_cols {
        return Ok(matrix);
    }
    let len = numeric_len(value);
    if len == rows && is_numeric_vector(value) {
        return Ok(DMatrix::from_column_slice(rows, 1, matrix.as_slice()));
    }
    Err(invalid(
        "lscov: B must have one row per observation or be an observation vector",
    ))
}

fn rhs_columns(value: &NumericMatrix, rows: usize) -> BuiltinResult<usize> {
    match value {
        NumericMatrix::Real(tensor) => {
            rhs_columns_from_shape(tensor.rows, tensor.cols, tensor, rows)
        }
        NumericMatrix::Complex(tensor) => {
            if is_complex_vector(tensor) && tensor::complex_tensor_element_len(tensor) == rows {
                Ok(1)
            } else if tensor.rows == rows {
                Ok(tensor.cols)
            } else {
                Err(invalid(
                    "lscov: B must have one row per observation or be an observation vector",
                ))
            }
        }
    }
}

fn rhs_columns_from_shape(
    tensor_rows: usize,
    tensor_cols: usize,
    tensor: &Tensor,
    rows: usize,
) -> BuiltinResult<usize> {
    if is_vector(tensor) && tensor::tensor_element_len(tensor) == rows {
        Ok(1)
    } else if tensor_rows == rows {
        Ok(tensor_cols)
    } else {
        Err(invalid(
            "lscov: B must have one row per observation or be an observation vector",
        ))
    }
}

fn matrix_rows(value: &NumericMatrix) -> usize {
    match value {
        NumericMatrix::Real(tensor) => tensor.rows,
        NumericMatrix::Complex(tensor) => tensor.rows,
    }
}

fn matrix_cols(value: &NumericMatrix) -> usize {
    match value {
        NumericMatrix::Real(tensor) => tensor.cols,
        NumericMatrix::Complex(tensor) => tensor.cols,
    }
}

fn numeric_len(value: &NumericMatrix) -> usize {
    match value {
        NumericMatrix::Real(tensor) => tensor::tensor_element_len(tensor),
        NumericMatrix::Complex(tensor) => tensor::complex_tensor_element_len(tensor),
    }
}

fn is_numeric_vector(value: &NumericMatrix) -> bool {
    match value {
        NumericMatrix::Real(tensor) => is_vector(tensor),
        NumericMatrix::Complex(tensor) => is_complex_vector(tensor),
    }
}

fn ensure_2d(value: &NumericMatrix, label: &str) -> BuiltinResult<()> {
    let shape_len = match value {
        NumericMatrix::Real(tensor) => tensor.shape.len(),
        NumericMatrix::Complex(tensor) => tensor.shape.len(),
    };
    if shape_len > 2 {
        return Err(invalid(format!("lscov: {label} must be a 2-D matrix")));
    }
    Ok(())
}

fn is_vector(tensor: &Tensor) -> bool {
    tensor.shape.len() <= 2 && (tensor.rows == 1 || tensor.cols == 1)
}

fn is_complex_vector(tensor: &ComplexTensor) -> bool {
    tensor.shape.len() <= 2 && (tensor.rows == 1 || tensor.cols == 1)
}

fn validate_symmetric_covariance(covariance: &DMatrix<f64>) -> BuiltinResult<()> {
    for row in 0..covariance.nrows() {
        for col in (row + 1)..covariance.ncols() {
            let left = covariance[(row, col)];
            let right = covariance[(col, row)];
            let scale = left.abs().max(right.abs()).max(1.0);
            if (left - right).abs() > EPS * scale {
                return Err(invalid("lscov: V covariance matrix must be symmetric"));
            }
        }
    }
    Ok(())
}

fn scaled_psd_tolerance(values: &[f64]) -> f64 {
    let scale = values
        .iter()
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    EPS * scale * values.len().max(1) as f64
}

fn ensure_budget(rows: usize, cols: usize, label: &str) -> BuiltinResult<()> {
    let cells = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid(format!("lscov: {label} is too large")))?;
    if cells > MAX_LSCOV_CELLS {
        return Err(invalid(format!("lscov: {label} is too large")));
    }
    Ok(())
}

fn ensure_work_budget(cols: usize) -> BuiltinResult<()> {
    let cells = cols
        .checked_mul(cols)
        .ok_or_else(|| invalid("lscov: covariance work array is too large"))?;
    if cells > MAX_LSCOV_CELLS {
        return Err(invalid("lscov: covariance work array is too large"));
    }
    Ok(())
}

fn ensure_output_budget(
    cols: usize,
    rhs_cols: usize,
    requested_outputs: usize,
) -> BuiltinResult<()> {
    let primary = cols
        .checked_mul(rhs_cols)
        .ok_or_else(|| invalid("lscov: output is too large"))?;
    let mut cells = primary;
    if requested_outputs >= 2 {
        cells = cells
            .checked_add(primary)
            .ok_or_else(|| invalid("lscov: output is too large"))?;
    }
    if requested_outputs >= 3 {
        cells = cells
            .checked_add(rhs_cols)
            .ok_or_else(|| invalid("lscov: output is too large"))?;
    }
    if requested_outputs >= 4 {
        cells = cells
            .checked_add(
                cols.checked_mul(cols)
                    .ok_or_else(|| invalid("lscov: output is too large"))?,
            )
            .ok_or_else(|| invalid("lscov: output is too large"))?;
    }
    if cells > MAX_LSCOV_CELLS {
        return Err(invalid("lscov: output is too large"));
    }
    Ok(())
}

fn real_tensor_value(data: Vec<f64>, shape: Vec<usize>, label: &str) -> BuiltinResult<Value> {
    Tensor::new(data, shape)
        .map(Value::Tensor)
        .map_err(|err| internal(format!("lscov: failed to construct {label}: {err}")))
}

fn complex_matrix_value(
    matrix: &DMatrix<Complex64>,
    rows: usize,
    cols: usize,
    label: &str,
) -> BuiltinResult<Value> {
    let data = matrix
        .as_slice()
        .iter()
        .map(|value| (value.re, value.im))
        .collect::<Vec<_>>();
    let tensor = ComplexTensor::new(data, vec![rows, cols])
        .map_err(|err| internal(format!("lscov: failed to construct {label}: {err}")))?;
    if tensor
        .data
        .iter()
        .all(|(_, im)| im.abs() <= EPS || im.is_nan())
    {
        let real = tensor.data.iter().map(|(re, _)| *re).collect::<Vec<_>>();
        return real_tensor_value(real, vec![rows, cols], label);
    }
    Ok(complex_tensor_into_value(tensor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{CharArray, IntegerStorage};

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
    }

    fn poisoned_int_tensor(
        storage: IntegerStorage,
        rows: usize,
        cols: usize,
        poison: f64,
    ) -> Value {
        let mut tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
        tensor.data.fill(poison);
        Value::Tensor(tensor)
    }

    fn mirrorless_int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
        let mut tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
        tensor.data.clear();
        Value::Tensor(tensor)
    }

    fn complex_tensor(data: Vec<(f64, f64)>, rows: usize, cols: usize) -> Value {
        Value::ComplexTensor(ComplexTensor::new(data, vec![rows, cols]).unwrap())
    }

    fn outputs(value: Value) -> Vec<Value> {
        match value {
            Value::OutputList(values) => values,
            other => panic!("expected output list, got {other:?}"),
        }
    }

    fn tensor_ref(value: &Value) -> &Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    fn complex_ref(value: &Value) -> &ComplexTensor {
        match value {
            Value::ComplexTensor(tensor) => tensor,
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    fn numeric_matrix_shape(value: &Value) -> Vec<usize> {
        match value {
            Value::Tensor(tensor) => tensor.shape.clone(),
            Value::ComplexTensor(tensor) => tensor.shape.clone(),
            other => panic!("expected numeric matrix, got {other:?}"),
        }
    }

    fn assert_close(left: f64, right: f64) {
        assert!(
            (left - right).abs() < 1.0e-9,
            "{left:?} not close to {right:?}"
        );
    }

    #[test]
    fn lscov_solves_ordinary_least_squares_outputs() {
        let _guard = crate::output_count::push_output_count(Some(4));
        let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
        let b = tensor(vec![1.0, 3.0, 5.0], 3, 1);
        let out = outputs(block_on(lscov_builtin(a, b, Vec::new())).unwrap());
        let x = tensor_ref(&out[0]);
        assert_eq!(x.shape, vec![2, 1]);
        assert_close(x.data[0], 1.0);
        assert_close(x.data[1], 2.0);
        assert_eq!(tensor_ref(&out[1]).shape, vec![2, 1]);
        assert_close(tensor_ref(&out[2]).data[0], 0.0);
        assert_eq!(tensor_ref(&out[3]).shape, vec![2, 2]);
    }

    #[test]
    fn lscov_accepts_typed_integer_matrices_and_weights() {
        let _guard = crate::output_count::push_output_count(Some(1));
        let a = poisoned_int_tensor(IntegerStorage::I16(vec![1, 1, 1, 0, 1, 2]), 3, 2, f64::NAN);
        let b = poisoned_int_tensor(IntegerStorage::I16(vec![1, 3, 5]), 3, 1, f64::NAN);
        let weights = poisoned_int_tensor(IntegerStorage::U16(vec![1, 1, 1]), 3, 1, f64::NAN);
        let out = outputs(block_on(lscov_builtin(a, b, vec![weights])).unwrap());
        let x = tensor_ref(&out[0]);
        assert_eq!(x.shape, vec![2, 1]);
        assert_close(x.data[0], 1.0);
        assert_close(x.data[1], 2.0);
    }

    #[test]
    fn lscov_reads_typed_integer_covariance_from_exact_storage() {
        let _guard = crate::output_count::push_output_count(Some(1));
        let a = poisoned_int_tensor(IntegerStorage::I16(vec![1, 1, 1, 0, 1, 2]), 3, 2, f64::NAN);
        let b = poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 10]), 3, 1, f64::NAN);
        let v = poisoned_int_tensor(
            IntegerStorage::U16(vec![
                1, 0, 0, //
                0, 1, 0, //
                0, 0, 100,
            ]),
            3,
            3,
            f64::NAN,
        );
        let alg = Value::CharArray(CharArray::new_row("chol"));
        let out = outputs(block_on(lscov_builtin(a, b, vec![v, alg])).unwrap());
        let x = tensor_ref(&out[0]);
        assert_eq!(x.shape, vec![2, 1]);
        assert!(x.data.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn lscov_weighting_length_uses_typed_integer_storage_not_mirror() {
        let weighting = block_on(parse_weighting(
            mirrorless_int_tensor(IntegerStorage::U16(vec![1, 2, 3]), 3, 1),
            3,
            Algorithm::Orth,
        ))
        .unwrap();
        match weighting {
            Weighting::Weights(values) => assert_eq!(values, vec![1.0, 2.0, 3.0]),
            other => panic!("expected vector weighting, got {other:?}"),
        }
    }

    #[test]
    fn lscov_supports_weight_vector_weighting() {
        let _guard = crate::output_count::push_output_count(Some(3));
        let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
        let b = tensor(vec![1.0, 2.0, 10.0], 3, 1);
        let v = tensor(vec![1.0, 1.0, 100.0], 3, 1);
        let out = outputs(block_on(lscov_builtin(a, b, vec![v])).unwrap());
        let x = tensor_ref(&out[0]);
        assert_close(x.data[0], -0.3972055888223553);
        assert_close(x.data[1], 5.191616766467066);
        let stdx = tensor_ref(&out[1]);
        assert_eq!(stdx.shape, vec![2, 1]);
        assert!(stdx.data.iter().all(|value| value.is_finite()));
        assert!(tensor_ref(&out[2]).data[0].is_finite());
    }

    #[test]
    fn lscov_supports_zero_weight_observations() {
        let _guard = crate::output_count::push_output_count(Some(1));
        let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
        let b = tensor(vec![1.0, 3.0, 100.0], 3, 1);
        let w = tensor(vec![1.0, 1.0, 0.0], 3, 1);
        let out = outputs(block_on(lscov_builtin(a, b, vec![w])).unwrap());
        let x = tensor_ref(&out[0]);
        assert_eq!(x.shape, vec![2, 1]);
        assert_close(x.data[0], 1.0);
        assert_close(x.data[1], 2.0);
    }

    #[test]
    fn lscov_supports_full_covariance_and_chol_algorithm() {
        let _guard = crate::output_count::push_output_count(Some(1));
        let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
        let b = tensor(vec![1.0, 2.0, 10.0], 3, 1);
        let v = tensor(
            vec![
                1.0, 0.1, 0.0, //
                0.1, 1.0, 0.0, //
                0.0, 0.0, 100.0,
            ],
            3,
            3,
        );
        let alg = Value::CharArray(CharArray::new_row("chol"));
        let out = outputs(block_on(lscov_builtin(a, b, vec![v, alg])).unwrap());
        let x = tensor_ref(&out[0]);
        assert_eq!(x.shape, vec![2, 1]);
        assert!(x.data.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn lscov_orth_handles_singular_psd_covariance() {
        let _guard = crate::output_count::push_output_count(Some(3));
        let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
        let b = tensor(vec![1.0, 3.0, 100.0], 3, 1);
        let v = tensor(
            vec![
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0,
            ],
            3,
            3,
        );
        let alg = Value::String("orth".to_string());
        let out = outputs(block_on(lscov_builtin(a, b, vec![v, alg])).unwrap());
        let x = tensor_ref(&out[0]);
        assert_close(x.data[0], 1.0);
        assert_close(x.data[1], 2.0);
        assert_close(tensor_ref(&out[2]).data[0], 0.0);
    }

    #[test]
    fn lscov_supports_matrix_rhs() {
        let _guard = crate::output_count::push_output_count(Some(3));
        let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
        let b = tensor(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0], 3, 2);
        let out = outputs(block_on(lscov_builtin(a, b, Vec::new())).unwrap());
        let x = tensor_ref(&out[0]);
        assert_eq!(x.shape, vec![2, 2]);
        assert_close(x.data[0], 1.0);
        assert_close(x.data[1], 2.0);
        assert_close(x.data[2], 2.0);
        assert_close(x.data[3], 2.0);
        assert_eq!(tensor_ref(&out[2]).shape, vec![1, 2]);
    }

    #[test]
    fn lscov_supports_complex_design_and_response() {
        let _guard = crate::output_count::push_output_count(Some(4));
        let a = complex_tensor(
            vec![
                (1.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (1.0, 0.0),
                (2.0, -1.0),
            ],
            3,
            2,
        );
        let b = complex_tensor(vec![(1.0, 1.0), (3.0, 0.0), (5.0, -1.0)], 3, 1);
        let out = outputs(block_on(lscov_builtin(a, b, Vec::new())).unwrap());
        let x = complex_ref(&out[0]);
        assert_eq!(x.shape, vec![2, 1]);
        assert!(x
            .data
            .iter()
            .all(|(re, im)| re.is_finite() && im.is_finite()));
        assert_eq!(tensor_ref(&out[1]).shape, vec![2, 1]);
        assert_eq!(tensor_ref(&out[2]).shape, vec![1, 1]);
        assert_eq!(numeric_matrix_shape(&out[3]), vec![2, 2]);
    }

    #[test]
    fn lscov_rejects_fourth_output_for_matrix_rhs() {
        let _guard = crate::output_count::push_output_count(Some(4));
        let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
        let b = tensor(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0], 3, 2);
        let err = block_on(lscov_builtin(a, b, Vec::new())).unwrap_err();
        assert!(err.message().contains("fourth output S"));
    }

    #[test]
    fn lscov_accepts_zero_weights() {
        let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
        let b = tensor(vec![1.0, 3.0, 5.0], 3, 1);
        let v = tensor(vec![1.0, 0.0, 1.0], 3, 1);
        block_on(lscov_builtin(a, b, vec![v])).expect("zero weights are allowed");
    }

    #[test]
    fn lscov_rejects_negative_weights() {
        let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
        let b = tensor(vec![1.0, 3.0, 5.0], 3, 1);
        let v = tensor(vec![1.0, -1.0, 1.0], 3, 1);
        let err = block_on(lscov_builtin(a, b, vec![v])).unwrap_err();
        assert!(err.message().contains("finite nonnegative"));
    }

    #[test]
    fn lscov_underdetermined_mse_is_zero() {
        let _guard = crate::output_count::push_output_count(Some(3));
        let a = tensor(vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0], 2, 3);
        let b = tensor(vec![1.0, 2.0], 2, 1);
        let out = outputs(block_on(lscov_builtin(a, b, Vec::new())).unwrap());
        assert_eq!(tensor_ref(&out[0]).shape, vec![3, 1]);
        assert_close(tensor_ref(&out[2]).data[0], 0.0);
    }
}
