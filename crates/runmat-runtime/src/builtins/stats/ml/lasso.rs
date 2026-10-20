//! Lasso and elastic-net regularized linear regression.

mod admission;
mod cross_validation;
mod design;
mod model;
mod options;
mod result;
mod solver;
mod values;

use admission::*;
use cross_validation::*;
use design::*;
use model::*;
use options::*;
use result::*;
use solver::*;
use values::*;

use runmat_builtins::{
    BuiltinErrorDescriptor, LassoOption, LassoOptionRole, LASSO_ERROR_CONVERGENCE,
    LASSO_ERROR_INTERNAL, LASSO_ERROR_INVALID_ARGUMENT, LASSO_INTEGER_BOOLEAN_EXTENSION,
    LASSO_INTEGER_CONTROL_EXTENSION, LASSO_INTEGER_DATA_EXTENSION,
    LASSO_INTEGER_PARAMETER_EXTENSION, LASSO_RESIDENT_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, LogicalArray, Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::stats::ml::regularized::{
    geometric_sequence, predictor_matrix_dimensions, soft_threshold,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "lasso";
const EPS: f64 = 1.0e-12;
const MAX_NUM_LAMBDA: usize = 10_000;
const MAX_ITERATIONS: usize = 1_000_000;

fn lasso_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    lasso_error(message, &LASSO_ERROR_INVALID_ARGUMENT)
}

fn convergence_error(message: impl Into<String>) -> RuntimeError {
    lasso_error(message, &LASSO_ERROR_CONVERGENCE)
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    lasso_error(message, &LASSO_ERROR_INTERNAL)
}

#[derive(Clone, Debug)]
struct LassoOptions {
    alpha: f64,
    lambda: Option<Vec<f64>>,
    lambda_ratio: f64,
    num_lambda: usize,
    standardize: bool,
    intercept: bool,
    weights: Option<Vec<f64>>,
    rel_tol: f64,
    max_iter: usize,
    cv: CvSpec,
    predictor_names: Option<Vec<String>>,
    df_max: Option<usize>,
}

impl Default for LassoOptions {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            lambda: None,
            lambda_ratio: 1.0e-4,
            num_lambda: 100,
            standardize: true,
            intercept: true,
            weights: None,
            rel_tol: 1.0e-4,
            max_iter: 100_000,
            cv: CvSpec::Resubstitution,
            predictor_names: None,
            df_max: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CvSpec {
    Resubstitution,
    KFold(usize),
}

#[derive(Clone, Debug)]
struct PreparedData {
    x_original: Vec<f64>,
    x_work: Vec<f64>,
    y_original: Vec<f64>,
    y_work: Vec<f64>,
    weights: Vec<f64>,
    x_means: Vec<f64>,
    x_scales: Vec<f64>,
    y_mean: f64,
    rows: usize,
    cols: usize,
}

#[derive(Clone, Debug)]
struct FitPath {
    lambdas: Vec<f64>,
    coefficients: Vec<Vec<f64>>,
    intercepts: Vec<f64>,
    df: Vec<f64>,
    mse: Vec<f64>,
    iterations: Vec<f64>,
}

#[derive(Clone, Debug)]
struct FitResult {
    b: Value,
    fit_info: Value,
}

struct OptionInput {
    option: LassoOption,
    value: Value,
}

#[runtime_builtin(
    name = "lasso",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::lasso"
)]
pub(crate) async fn lasso_builtin(
    x: Value,
    y: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let rest = admit_options(rest)?;
    ensure_extensions(&x, &y, &rest)?;
    let (rows, cols) = predictor_matrix_dimensions(&x)
        .map_err(|detail| invalid_argument(format!("lasso: {detail}")))?;
    if rows == 0 || cols == 0 {
        return Err(invalid_argument(
            "lasso: X must be a nonempty numeric matrix",
        ));
    }
    let all_host_options = rest
        .iter()
        .all(|input| !crate::dispatcher::value_contains_gpu(&input.value));
    let host_options = parse_options(
        rest.iter()
            .filter(|input| !crate::dispatcher::value_contains_gpu(&input.value)),
    )?;
    let options = if all_host_options {
        host_options
    } else {
        let rest = gather_options(rest).await?;
        parse_options(rest.iter())?
    };
    let x = gathered(x).await?;
    let y = gathered(y).await?;
    let result = lasso_compute(x, y, options)?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![result.b])),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![result.b, result.fit_info],
        )),
        None => Ok(result.b),
    }
}

#[derive(Clone, Debug)]
struct CvInfo {
    mse: Vec<f64>,
    se: Vec<f64>,
    lambda_min_mse: f64,
    lambda_1se: f64,
    index_min_mse: usize,
    index_1se: usize,
}

#[cfg(test)]
#[path = "lasso/tests.rs"]
mod tests;
