//! Linear least-squares with observation covariance weighting.

use nalgebra::{DMatrix, SymmetricEigen};
use num_complex::Complex64;
pub use runmat_builtins::LSCOV_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, LscovAlgorithm as Algorithm, LSCOV_ERROR_INTERNAL,
    LSCOV_ERROR_INVALID_ARGUMENT, LSCOV_ERROR_NUMERICAL,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, Tensor, Value};

use crate::builtins::common::random_args::{complex_tensor_into_value, keyword_of};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "lscov";
const EPS: f64 = 1.0e-12;
const MAX_LSCOV_CELLS: usize = 50_000_000;
mod solver;
use solver::*;
mod transform;
use transform::*;
mod fit;
use fit::*;
mod geometry;
use geometry::*;
mod resources;
use resources::*;
mod output;
use output::*;

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
    lscov_error(message, &LSCOV_ERROR_INVALID_ARGUMENT)
}

fn numerical(message: impl Into<String>) -> RuntimeError {
    lscov_error(message, &LSCOV_ERROR_NUMERICAL)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    lscov_error(message, &LSCOV_ERROR_INTERNAL)
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
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::lscov"
)]
async fn lscov_builtin(
    a: Value,
    b: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if rest.len() > 2 {
        return Err(invalid("lscov: accepts at most four input arguments"));
    }
    if crate::output_count::current_output_count().is_some_and(|count| count > 4) {
        return Err(invalid("lscov: too many output arguments"));
    }
    let mut header = parse_header(rest)?;
    preflight_geometry(
        &a,
        &b,
        &mut header,
        crate::output_count::current_output_count(),
    )?;
    let a = gather_numeric_matrix(a).await?;
    let b = gather_numeric_matrix(b).await?;
    let parsed = finish_header(header, matrix_rows(&a)).await?;

    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
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

mod admission;
use admission::*;

#[cfg(test)]
mod tests;
