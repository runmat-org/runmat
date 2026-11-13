//! Fitted probability distribution data, evaluation, and sampling services.

mod arguments;
mod admission;
mod cdf;
mod evaluation;
mod fit;
mod fitting;
mod icdf;
mod object;
mod pdf;
mod sampling;
mod sampling_entry;
mod support;
#[cfg(test)]
mod tests;

use arguments::*;
pub(super) use cdf::evaluate_cdf_entry;
use cdf::{cdf_output_precision, is_typed_integer_value, CdfOutputPrecision};
use evaluation::*;
pub(super) use fit::fit_sample_distribution;
use fitting::*;
pub(crate) use icdf::{admit_icdf_probability_distribution, icdf_probability_distribution};
use object::*;
pub(super) use pdf::evaluate_pdf_entry;
use sampling::*;
pub(super) use sampling_entry::sample_distribution;
use support::*;

use runmat_accelerate_api::{GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{
    FittedDistributionFamily as DistributionKind, FittedFitOption, CDF_INTEGER_PARAMETER_EXTENSION,
    CDF_INTEGER_X_EXTENSION, CDF_LOGICAL_INPUT_EXTENSION, FITDIST_INTEGER_DATA_EXTENSION,
    FITDIST_INTEGER_FREQUENCY_EXTENSION, FITDIST_RESIDENT_FALLBACK_EXTENSION,
    PDF_INTEGER_PARAMETER_EXTENSION, PDF_INTEGER_X_EXTENSION, RANDOM_INTEGER_PARAMETER_EXTENSION,
    RANDOM_INTEGER_SIZE_EXTENSION,
};
use runmat_value::{NumericDType, ObjectInstance, StringArray, Tensor, Value};

use crate::builtins::common::broadcast;
use crate::builtins::common::random;
use crate::builtins::common::random_args::{extract_dims, keyword_of};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::elementwise::gamma_functions::gammaln::gammaln_nonnegative_scalar;
use crate::builtins::stats::summary::distribution_math;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const FITDIST_NAME: &str = "fitdist";
const PDF_NAME: &str = "pdf";
const CDF_NAME: &str = "cdf";
const RANDOM_NAME: &str = "random";
const MIN_POSITIVE: f64 = 1.0e-12;
#[derive(Clone, Debug)]
struct FittedDistribution {
    kind: DistributionKind,
    parameters: Vec<f64>,
    nlogl: f64,
    observations: f64,
}

#[derive(Clone, Debug)]
struct WeightedSample {
    values: Vec<f64>,
    weights: Vec<f64>,
    total_weight: f64,
}

#[derive(Default)]
struct FitOptions {
    frequency: Option<Vec<f64>>,
}

#[derive(Clone, Copy)]
enum FittedErrorKind {
    InvalidArgument,
    Numerical,
    Internal,
}

impl FittedErrorKind {
    const fn suffix(self) -> &'static str {
        match self {
            Self::InvalidArgument => "InvalidArgument",
            Self::Numerical => "Numerical",
            Self::Internal => "Internal",
        }
    }
}

fn error_for(
    builtin: &'static str,
    kind: FittedErrorKind,
    message: impl Into<String>,
) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(builtin)
        .with_identifier(format!("RunMat:{builtin}:{}", kind.suffix()))
        .build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    error_for(FITDIST_NAME, FittedErrorKind::InvalidArgument, message)
}

fn invalid_for(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    error_for(builtin, FittedErrorKind::InvalidArgument, message)
}

fn with_fitted_context(builtin: &'static str, mut error: RuntimeError) -> RuntimeError {
    if error.context.builtin.is_none() {
        error.context = error.context.with_builtin(builtin);
    }
    error
}

fn numerical(message: impl Into<String>) -> RuntimeError {
    error_for(FITDIST_NAME, FittedErrorKind::Numerical, message)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    error_for(FITDIST_NAME, FittedErrorKind::Internal, message)
}

fn internal_for(builtin: &'static str, message: impl Into<String>) -> RuntimeError {
    error_for(builtin, FittedErrorKind::Internal, message)
}
