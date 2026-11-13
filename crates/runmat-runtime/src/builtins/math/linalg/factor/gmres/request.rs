//! GMRES positional grammar. Matrix and vector class compatibility is checked
//! by the selected borrowed operator before workspace admission.

use runmat_value::{NumericDType, Value};

use super::super::iterative::controls::{
    coefficient_geometry, is_callable, is_empty, positive_count, real_scalar, rhs_geometry,
    GeometryError,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(super) struct GmresRequest {
    pub coefficient: Value,
    pub rhs: Value,
    pub restart: usize,
    pub max_outer: usize,
    pub tolerance: f64,
    pub first_preconditioner: Option<Value>,
    pub second_preconditioner: Option<Value>,
    pub initial: Option<Value>,
}

pub(super) fn parse(
    coefficient: Value,
    rhs: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<GmresRequest> {
    if rest.len() > 6 {
        return Err(invalid("gmres accepts at most eight inputs"));
    }
    let (n, cols, dtype) = if is_callable(&coefficient) {
        let (n, dtype) = rhs_geometry(&rhs).map_err(map_geometry)?;
        (n, n, dtype)
    } else {
        coefficient_geometry(&coefficient).map_err(map_geometry)?
    };
    if n == 0 || cols != n {
        return Err(invalid("A must be a nonempty square operator"));
    }
    let restart = match rest.first().filter(|value| !is_empty(value)) {
        Some(value) => positive_count(value)
            .filter(|&count| count <= n)
            .ok_or_else(|| invalid("restart must be an integer in 1..=size(A,1)"))?,
        None => n,
    };
    let restarted = restart < n;
    let tolerance = match rest.get(1).filter(|value| !is_empty(value)) {
        Some(value) => real_scalar(value)
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or_else(|| invalid("tol must be a positive finite real scalar"))?,
        None if dtype == NumericDType::F32 => 1e-3,
        None => 1e-6,
    };
    let max_outer = match rest.get(2).filter(|value| !is_empty(value)) {
        Some(value) => positive_count(value)
            .ok_or_else(|| invalid("maxit must be a positive integer scalar"))?,
        None if restarted => n.div_ceil(restart).min(10),
        None => n.min(10),
    };
    let mut rest = rest.into_iter();
    let _ = rest.next();
    let _ = rest.next();
    let _ = rest.next();
    Ok(GmresRequest {
        coefficient,
        rhs,
        restart: if restarted { restart } else { max_outer },
        max_outer: if restarted { max_outer } else { 1 },
        tolerance,
        first_preconditioner: rest.next().filter(|value| !is_empty(value)),
        second_preconditioner: rest.next().filter(|value| !is_empty(value)),
        initial: rest.next().filter(|value| !is_empty(value)),
    })
}

fn map_geometry(error: GeometryError) -> RuntimeError {
    match error {
        GeometryError::UnsupportedRightHandSide => {
            unsupported("callable A requires a native floating b vector")
        }
        GeometryError::InvalidRightHandSide => {
            invalid("b must be a nonempty n-by-1 floating vector")
        }
        GeometryError::UnsupportedCoefficient => {
            unsupported("A must be a dense or sparse floating matrix")
        }
        GeometryError::UnsupportedNativeClass => {
            unsupported("A must have native single or double storage")
        }
    }
}

pub(super) fn invalid(message: &str) -> RuntimeError {
    build_runtime_error(format!("gmres: {message}"))
        .with_builtin("gmres")
        .with_identifier("RunMat:gmres:InvalidInput")
        .build()
}

fn unsupported(message: &str) -> RuntimeError {
    build_runtime_error(format!("gmres: {message}"))
        .with_builtin("gmres")
        .with_identifier("RunMat:gmres:UnsupportedInput")
        .build()
}
