//! Multiple linear regression compatibility surface.

use nalgebra::{DMatrix, DVector};
pub use runmat_builtins::REGRESS_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, REGRESS_ERROR_INTERNAL, REGRESS_ERROR_INVALID_ARGUMENT,
    REGRESS_ERROR_NUMERICAL, REGRESS_INTEGER_ALPHA_EXTENSION, REGRESS_INTEGER_DATA_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::stats::summary::distribution_math::{regularized_beta, student_t_inv};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const NAME: &str = "regress";
const EPS: f64 = 1.0e-12;
const MAX_REGRESS_CELLS: usize = 50_000_000;

fn regress_error(
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
    regress_error(message, &REGRESS_ERROR_INVALID_ARGUMENT)
}

fn numerical(message: impl Into<String>) -> RuntimeError {
    regress_error(message, &REGRESS_ERROR_NUMERICAL)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    regress_error(message, &REGRESS_ERROR_INTERNAL)
}

#[derive(Clone, Debug)]
struct PreparedData {
    design: DMatrix<f64>,
    y: DVector<f64>,
}

#[derive(Clone, Debug)]
struct CoefficientSolve {
    beta: Vec<f64>,
    fitted: DVector<f64>,
    inv_xtx_active: Option<DMatrix<f64>>,
    active_columns: Vec<usize>,
    rank: usize,
}

#[runtime_builtin(
    name = "regress",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::regress"
)]
async fn regress_builtin(
    y: Value,
    x: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(invalid("regress: accepts at most one alpha argument"));
    }
    let outputs = crate::output_count::current_output_count();
    if outputs.is_some_and(|count| count > 5) {
        return Err(invalid("regress: too many output arguments"));
    }
    let alpha = rest.into_iter().next().map(prepare_alpha).transpose()?;
    preflight_data(&y, &x, outputs)?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &y,
        &REGRESS_INTEGER_DATA_EXTENSION,
        NAME,
        "response",
    )
    .await?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &x,
        &REGRESS_INTEGER_DATA_EXTENSION,
        NAME,
        "design",
    )
    .await?;
    if let Some(alpha) = &alpha {
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            alpha.source(),
            &REGRESS_INTEGER_ALPHA_EXTENSION,
            NAME,
            "alpha",
        )
        .await?;
    }
    let alpha = alpha.map(AlphaHeader::after_integer_boundary).transpose()?;
    let y = gather_tensor(y).await?;
    let x = gather_tensor(x).await?;
    let alpha = if let Some(value) = alpha {
        value.finish().await?
    } else {
        0.05
    };
    match outputs {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(out_count) => {
            let outputs = regress_compute(y, x, alpha, out_count)?;
            Ok(crate::output_count::output_list_with_padding(
                out_count, outputs,
            ))
        }
        None => Ok(regress_compute(y, x, alpha, 1)?
            .into_iter()
            .next()
            .expect("regress always returns b for scalar-output calls")),
    }
}

mod admission;
use admission::*;
mod fit;
use fit::*;
mod solver;
use solver::*;
mod resources;
use resources::*;
mod observations;
use observations::*;
mod rank;
use rank::*;
mod diagnostics;
use diagnostics::*;
mod output;
use output::*;

#[cfg(test)]
mod tests;
