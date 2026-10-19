use runmat_value::{NumericDType, Value};

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

pub(super) fn parse(coefficient: Value, rhs: Value, rest: Vec<Value>) -> BuiltinResult<PcgRequest> {
    if rest.len() > 5 {
        return Err(invalid("pcg accepts at most seven inputs"));
    }
    let (rows, cols, dtype) = if is_callable(&coefficient) {
        let (rows, dtype) = rhs_geometry(&rhs)?;
        (rows, rows, dtype)
    } else {
        coefficient_geometry(&coefficient)?
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

pub(super) fn is_callable(value: &Value) -> bool {
    matches!(
        value,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
    )
}

fn rhs_geometry(value: &Value) -> BuiltinResult<(usize, NumericDType)> {
    let (shape, dtype, native) = match value {
        Value::Tensor(tensor) => (
            tensor.shape(),
            tensor.numeric_dtype(),
            tensor.as_f32_slice().is_some() || tensor.as_f64_slice().is_some(),
        ),
        Value::ComplexTensor(tensor) => (
            tensor.shape(),
            tensor.numeric_dtype(),
            tensor.as_f32_slice().is_some() || tensor.as_f64_slice().is_some(),
        ),
        _ => {
            return Err(unsupported(
                "callable A requires a native floating b vector",
            ))
        }
    };
    let rows = shape.first().copied().unwrap_or(0);
    if !native || rows == 0 || shape.get(1) != Some(&1) || shape.iter().skip(2).any(|&n| n != 1) {
        return Err(invalid(
            "b must be a nonempty n-by-1 native floating vector",
        ));
    }
    Ok((rows, dtype))
}

fn coefficient_geometry(value: &Value) -> BuiltinResult<(usize, usize, NumericDType)> {
    let (rows, cols, single, double) = match value {
        Value::Tensor(matrix) => (
            matrix.rows(),
            matrix.cols(),
            matrix.as_f32_slice().is_some(),
            matrix.as_f64_slice().is_some(),
        ),
        Value::ComplexTensor(matrix) => (
            matrix.rows(),
            matrix.cols(),
            matrix.as_f32_slice().is_some(),
            matrix.as_f64_slice().is_some(),
        ),
        Value::SparseTensor(matrix) => (
            matrix.rows,
            matrix.cols,
            matrix.as_f32_slice().is_some() || matrix.as_complex_f32_slice().is_some(),
            matrix.as_f64_slice().is_some() || matrix.as_complex_f64_slice().is_some(),
        ),
        _ => {
            return Err(unsupported(
                "A must be a dense or sparse single/double matrix",
            ))
        }
    };
    let dtype = match (single, double) {
        (true, false) => NumericDType::F32,
        (false, true) => NumericDType::F64,
        _ => return Err(unsupported("A must have native single or double storage")),
    };
    Ok((rows, cols, dtype))
}

fn is_empty(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.len() == 0,
        Value::ComplexTensor(tensor) => tensor.len() == 0,
        Value::SparseTensor(tensor) => tensor.nnz() == 0 && tensor.rows == 0 && tensor.cols == 0,
        _ => false,
    }
}

fn real_scalar(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Int(value) => Some(value.to_f64()),
        Value::Tensor(tensor) if tensor.len() == 1 => {
            if let Some(values) = tensor.as_f64_slice() {
                Some(values[0])
            } else if let Some(values) = tensor.as_f32_slice() {
                Some(f64::from(values[0]))
            } else {
                tensor
                    .integer_storage()?
                    .value_at(0)
                    .map(|value| value.to_f64())
            }
        }
        _ => None,
    }
}

fn positive_count(value: &Value) -> Option<usize> {
    if let Value::Int(value) = value {
        return value.try_to_usize().filter(|&count| count > 0);
    }
    if let Value::Tensor(tensor) = value {
        if tensor.len() == 1 {
            if let Some(storage) = tensor.integer_storage() {
                return storage
                    .value_at(0)?
                    .try_to_usize()
                    .filter(|&count| count > 0);
            }
        }
    }
    let value = real_scalar(value)?;
    if !value.is_finite() || value <= 0.0 || value.fract() != 0.0 || value >= usize::MAX as f64 {
        return None;
    }
    Some(value as usize)
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
