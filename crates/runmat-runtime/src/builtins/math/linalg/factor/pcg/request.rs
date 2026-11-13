use runmat_value::{NumericDType, Value};

use super::super::iterative::controls::{
    coefficient_geometry, is_empty, positive_count, real_scalar, rhs_geometry, GeometryError,
};
use crate::{build_runtime_error, BuiltinResult};

pub(super) struct PcgRequest {
    pub(super) coefficient: Value,
    pub(super) rhs: Value,
    pub(super) tolerance: f64,
    pub(super) maximum_iterations: usize,
    pub(super) first_preconditioner: Option<Value>,
    pub(super) second_preconditioner: Option<Value>,
    pub(super) initial: Option<Value>,
}

pub(super) fn parse(
    coefficient: Value,
    rhs: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<PcgRequest> {
    if rest.len() > 5 {
        return Err(invalid("pcg accepts at most seven inputs"));
    }
    let (rows, cols, dtype) = if is_callable(&coefficient) {
        let (rows, dtype) = rhs_geometry(&rhs).map_err(map_geometry)?;
        (rows, rows, dtype)
    } else {
        coefficient_geometry(&coefficient).map_err(map_geometry)?
    };
    if rows != cols {
        return Err(invalid("A must be square"));
    }
    if rows == 0 {
        return Err(invalid("A must be nonempty"));
    }
    let tolerance = match rest.first().filter(|value| !is_empty(value)) {
        Some(value) => {
            let tolerance = real_scalar(value).ok_or_else(|| invalid("tol must be real scalar"))?;
            if !tolerance.is_finite() || tolerance <= 0.0 {
                return Err(invalid("tol must be positive and finite"));
            }
            tolerance
        }
        None if dtype == NumericDType::F32 => 1e-3,
        None => 1e-6,
    };
    let maximum_iterations = match rest.get(1).filter(|value| !is_empty(value)) {
        Some(value) => positive_count(value)
            .ok_or_else(|| invalid("maxit must be a positive integer scalar"))?,
        None => rows.min(20),
    };
    let mut rest = rest.into_iter();
    let _ = rest.next();
    let _ = rest.next();
    let first_preconditioner = rest.next().filter(|value| !is_empty(value));
    let second_preconditioner = rest.next().filter(|value| !is_empty(value));
    let initial = rest.next().filter(|value| !is_empty(value));
    Ok(PcgRequest {
        coefficient,
        rhs,
        tolerance,
        maximum_iterations,
        first_preconditioner,
        second_preconditioner,
        initial,
    })
}

pub(super) use super::super::iterative::controls::is_callable;

fn map_geometry(error: GeometryError) -> crate::RuntimeError {
    match error {
        GeometryError::UnsupportedRightHandSide => {
            unsupported("callable A requires a native floating b vector")
        }
        GeometryError::InvalidRightHandSide => {
            invalid("b must be a nonempty n-by-1 native floating vector")
        }
        GeometryError::UnsupportedCoefficient => {
            unsupported("A must be a dense or sparse single/double matrix")
        }
        GeometryError::UnsupportedNativeClass => {
            unsupported("A must have native single or double storage")
        }
    }
}

pub(super) fn invalid(message: &str) -> crate::RuntimeError {
    build_runtime_error(format!("pcg: {message}"))
        .with_builtin("pcg")
        .with_identifier("RunMat:pcg:InvalidInput")
        .build()
}

pub(super) fn unsupported(message: &str) -> crate::RuntimeError {
    build_runtime_error(format!("pcg: {message}"))
        .with_builtin("pcg")
        .with_identifier("RunMat:pcg:UnsupportedInput")
        .build()
}
