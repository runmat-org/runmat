//! Effect-free geometry admission before any resident operand is downloaded.
//! Value-dependent finite coordinates and topology indices remain with the
//! ordinary host parsers after materialization.
use super::*;

fn numeric_shape(value: &Value, builtin: &'static str) -> BuiltinResult<(usize, usize, usize)> {
    match value {
        Value::Tensor(tensor) => {
            let (rows, cols) = coordinate_matrix_dimensions(tensor);
            Ok((rows, cols, tensor.len()))
        }
        Value::Num(_) | Value::Int(_) => Ok((1, 1, 1)),
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_is_logical(handle)
                || runmat_accelerate_api::handle_storage(handle)
                    != runmat_accelerate_api::GpuTensorStorage::Real
            {
                return Err(invalid(format!("{builtin}: expected real numeric input")));
            }
            let shape = handle.shape();
            Ok((
                shape.first().copied().unwrap_or(0),
                shape.get(1).copied().unwrap_or(0),
                shape
                    .iter()
                    .try_fold(1usize, |n, dim| n.checked_mul(*dim))
                    .ok_or_else(|| invalid(format!("{builtin}: input shape overflows")))?,
            ))
        }
        other => Err(invalid(format!(
            "{builtin}: expected numeric input, got {other:?}"
        ))),
    }
}

fn matrix(value: &Value, builtin: &'static str) -> BuiltinResult<()> {
    if !ordinary_2d_value(value) {
        return Err(invalid(format!(
            "{builtin}: points must be an N-by-2 matrix"
        )));
    }
    let (rows, columns, _) = numeric_shape(value, builtin)?;
    if columns == 3 {
        return Err(unsupported(
            "3-D DelaunayTri triangulation is not yet supported",
        ));
    }
    if columns != 2 {
        return Err(invalid(format!(
            "{builtin}: points must be an N-by-2 matrix"
        )));
    }
    finite_host_coordinates(value, rows.saturating_mul(2), builtin)?;
    Ok(())
}

fn vector_pair(x: &Value, y: &Value, builtin: &'static str) -> BuiltinResult<()> {
    if !ordinary_2d_value(x) || !ordinary_2d_value(y) {
        return Err(invalid(format!(
            "{builtin}: x and y must be vectors with the same length"
        )));
    }
    let (xr, xc, xn) = numeric_shape(x, builtin)?;
    let (yr, yc, yn) = numeric_shape(y, builtin)?;
    if (xr != 1 && xc != 1) || (yr != 1 && yc != 1) || xn != yn {
        return Err(invalid(format!(
            "{builtin}: x and y must be vectors with the same length"
        )));
    }
    finite_host_coordinates(x, xn, builtin)?;
    finite_host_coordinates(y, yn, builtin)?;
    Ok(())
}

fn ordinary_2d_value(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => has_only_trailing_singletons(tensor.shape()),
        Value::GpuTensor(handle) => has_only_trailing_singletons(handle.shape()),
        _ => true,
    }
}

fn finite_host_coordinates(
    value: &Value,
    count: usize,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let finite = match value {
        Value::Num(number) => number.is_finite(),
        Value::Tensor(tensor) => (0..count).all(|index| {
            tensor
                .numeric_value_at(index)
                .is_some_and(NumericScalar::is_finite)
        }),
        Value::Int(_) | Value::GpuTensor(_) => true,
        _ => unreachable!("numeric_shape admitted the value"),
    };
    if finite {
        Ok(())
    } else {
        Err(invalid(format!("{builtin}: points must be finite")))
    }
}

fn empty_constraints(value: &Value) -> bool {
    matches!(numeric_shape(value, BUILTIN_NAME), Ok((_, 2, 0)))
}

pub(super) fn constraint_form(first: &Value, second: &Value) -> bool {
    looks_like_point_matrix(first) && (!is_vector_value(first) || empty_constraints(second))
}

pub(super) fn constructor(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    match args.len() {
        0 => Ok(()),
        1 => matrix(&args[0], BUILTIN_NAME),
        2 if constraint_form(&args[0], &args[1]) => {
            let first = &args[0];
            let second = &args[1];
            matrix(first, BUILTIN_NAME)?;
            let (rows, columns, len) = numeric_shape(second, BUILTIN_NAME)?;
            if rows == 0 || len == 0 {
                Ok(())
            } else if columns != 2 {
                Err(invalid("DelaunayTri: constraints must be an M-by-2 matrix"))
            } else {
                Err(unsupported(
                    "constrained DelaunayTri triangulation is not yet supported",
                ))
            }
        }
        2 => vector_pair(&args[0], &args[1], BUILTIN_NAME),
        3 => {
            let x = &args[0];
            let y = &args[1];
            let third = &args[2];
            vector_pair(x, y, BUILTIN_NAME)?;
            if !looks_like_constraints(third)
                && !matches!(third, Value::GpuTensor(handle) if handle.shape().get(1) == Some(&2))
            {
                return Err(unsupported(
                    "3-D DelaunayTri triangulation is not yet supported",
                ));
            }
            let (rows, _, len) = numeric_shape(third, BUILTIN_NAME)?;
            if rows == 0 || len == 0 {
                Ok(())
            } else {
                Err(unsupported(
                    "constrained DelaunayTri triangulation is not yet supported",
                ))
            }
        }
        _ => Err(invalid(
            "expected DelaunayTri(), DelaunayTri(X), DelaunayTri(x,y), or DelaunayTri(X,C)",
        )),
    }
}

pub(super) fn query(
    args: crate::call::FunctionArgumentView<'_>,
    builtin: &'static str,
) -> BuiltinResult<()> {
    match args.len() {
        1 => matrix(&args[0], builtin),
        2 => vector_pair(&args[0], &args[1], builtin),
        _ => Err(invalid(format!(
            "{builtin}: expected query matrix or x,y vectors"
        ))),
    }
}
