//! Lasso and elastic-net regularized generalized linear models.

mod admission;
mod cross_validation;
mod design;
mod model;
mod objective;
mod options;
mod response;
mod result;
mod solver;
mod values;

use admission::*;
use cross_validation::*;
use design::*;
use model::*;
use objective::*;
use options::*;
use response::*;
use result::*;
use solver::*;
use values::*;

use runmat_builtins::{
    BuiltinErrorDescriptor, BuiltinExtensionDescriptor, LassoglmDistribution as Distribution,
    LassoglmEstimateDisp, LassoglmLink, LassoglmOption, LassoglmOptionRole, StatOptionField,
    LASSOGLM_ERROR_CONVERGENCE, LASSOGLM_ERROR_INTERNAL, LASSOGLM_ERROR_INVALID_ARGUMENT,
    LASSOGLM_INTEGER_BOOLEAN_EXTENSION, LASSOGLM_INTEGER_CONTROL_EXTENSION,
    LASSOGLM_INTEGER_DATA_EXTENSION, LASSOGLM_INTEGER_PARAMETER_EXTENSION,
    LASSOGLM_RESIDENT_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, LogicalArray, StructValue, Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::stats::ml::regularized::{
    geometric_sequence, predictor_matrix_dimensions, soft_threshold,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "lassoglm";
const EPS: f64 = 1.0e-12;
const MAX_NUM_LAMBDA: usize = 10_000;
const MAX_ITERATIONS: usize = 1_000_000;

fn error(message: impl Into<String>, descriptor: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    error(message, &LASSOGLM_ERROR_INVALID_ARGUMENT)
}

fn convergence(message: impl Into<String>) -> RuntimeError {
    error(message, &LASSOGLM_ERROR_CONVERGENCE)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    error(message, &LASSOGLM_ERROR_INTERNAL)
}

fn parse_distribution(value: &Value) -> BuiltinResult<Distribution> {
    let text = scalar_text(value, "distribution")?;
    Distribution::from_public_name(&text)
        .ok_or_else(|| invalid(format!("lassoglm: unsupported distribution '{text}'")))
}

#[derive(Clone, Debug)]
struct Options {
    alpha: f64,
    lambda: Option<Vec<f64>>,
    lambda_ratio: f64,
    num_lambda: usize,
    standardize: bool,
    intercept: bool,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    binomial_size: Option<Vec<f64>>,
    link: Option<LassoglmLink>,
    rel_tol: f64,
    max_iter: usize,
    cv: CvSpec,
    predictor_names: Option<Vec<String>>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            lambda: None,
            lambda_ratio: 1.0e-4,
            num_lambda: 100,
            standardize: true,
            intercept: true,
            weights: None,
            offset: None,
            binomial_size: None,
            link: None,
            rel_tol: 1.0e-4,
            max_iter: 1000,
            cv: CvSpec::Resubstitution,
            predictor_names: None,
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
    y: Vec<f64>,
    trials: Vec<f64>,
    weights: Vec<f64>,
    offset: Vec<f64>,
    x_means: Vec<f64>,
    x_scales: Vec<f64>,
    rows: usize,
    cols: usize,
}

#[derive(Clone, Debug)]
struct FitPath {
    lambdas: Vec<f64>,
    coefficients: Vec<Vec<f64>>,
    intercepts: Vec<f64>,
    deviance: Vec<f64>,
    df: Vec<f64>,
    iterations: Vec<f64>,
}

#[derive(Clone, Debug)]
struct CvInfo {
    deviance: Vec<f64>,
    se: Vec<f64>,
    lambda_min_deviance: f64,
    lambda_1se: f64,
    index_min_deviance: usize,
    index_1se: usize,
}

#[derive(Clone, Debug)]
struct FitResult {
    b: Value,
    fit_info: Value,
}

struct OptionInput {
    option: LassoglmOption,
    name: String,
    value: Value,
}

#[runtime_builtin(
    name = "lassoglm",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::lassoglm"
)]
pub(crate) async fn lassoglm_builtin(
    x: Value,
    y: Value,
    distr: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let distribution = parse_distribution(&distr)?;
    let rest = admit_options(rest)?;
    ensure_extensions(&x, &y, &distr, &rest)?;
    let (rows, cols) =
        predictor_matrix_dimensions(&x).map_err(|detail| invalid(format!("lassoglm: {detail}")))?;
    if rows == 0 || cols == 0 {
        return Err(invalid("lassoglm: X must be a nonempty numeric matrix"));
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
    validate_distribution_options(distribution, &options)?;
    let x = gathered(x).await?;
    let y = gathered(y).await?;
    let result = lassoglm_compute(x, y, distribution, options)?;
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

#[cfg(test)]
#[path = "lassoglm/tests.rs"]
mod tests;
