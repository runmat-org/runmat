use super::*;

fn symbolic_variable(value: &Value) -> BuiltinResult<Option<String>> {
    symbolic_variable_name_from_value(value).map_err(|source| {
        let mut error = diff_invalid_argument("diff: symbolic variable must be valid UTF-16 text");
        error.source = Some(Box::new(source));
        error
    })
}

pub(super) fn diff_symbolic(
    expr: runmat_value::SymbolicExpr,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    let (variable, order) = parse_symbolic_diff_args(&expr, args)?;
    Ok(symbolic_expr_to_value(
        runmat_value::SymbolicExpr::derivative_expr(expr, variable, order),
    ))
}

pub(super) fn parse_symbolic_diff_args(
    expr: &runmat_value::SymbolicExpr,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<(String, u32)> {
    match args.len() {
        0 => Ok((infer_symbolic_diff_variable(expr)?, 1)),
        1 => {
            if let Some(variable) = symbolic_variable(&args[0])? {
                Ok((variable, 1))
            } else {
                Ok((
                    infer_symbolic_diff_variable(expr)?,
                    parse_symbolic_order(&args[0])?,
                ))
            }
        }
        2 => {
            if let Some(variable) = symbolic_variable(&args[0])? {
                Ok((variable, parse_symbolic_order(&args[1])?))
            } else if let Some(variable) = symbolic_variable(&args[1])? {
                Ok((variable, parse_symbolic_order(&args[0])?))
            } else {
                Err(diff_invalid_argument(
                    "diff: symbolic differentiation expects a variable and optional order",
                ))
            }
        }
        _ => Err(diff_invalid_argument(
            "diff: symbolic differentiation supports at most two trailing arguments",
        )),
    }
}

pub(super) fn infer_symbolic_diff_variable(
    expr: &runmat_value::SymbolicExpr,
) -> BuiltinResult<String> {
    let variables = expr.variables();
    if variables.len() == 1 {
        Ok(variables.into_iter().next().unwrap_or_default())
    } else if variables.is_empty() {
        Ok(String::new())
    } else {
        Err(diff_invalid_argument(
            "diff: symbolic differentiation variable is ambiguous",
        ))
    }
}

pub(super) fn parse_symbolic_order(value: &Value) -> BuiltinResult<u32> {
    let expr = value_to_symbolic_scalar(value).ok_or_else(|| {
        diff_invalid_argument("diff: symbolic differentiation order must be a scalar integer")
    })?;
    let Some(order) = expr.constant_value() else {
        return Err(diff_invalid_argument(
            "diff: symbolic differentiation order must be numeric",
        ));
    };
    if !order.is_finite() || order < 0.0 || (order.round() - order).abs() > 1.0e-12 {
        return Err(diff_invalid_argument(
            "diff: symbolic differentiation order must be a nonnegative integer",
        ));
    }
    if order > u32::MAX as f64 {
        return Err(diff_invalid_argument(
            "diff: symbolic differentiation order is too large",
        ));
    }
    Ok(order as u32)
}

pub(super) fn parse_arguments(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<(usize, Option<usize>)> {
    match args.len() {
        0 => Ok((1, None)),
        1 => {
            let order = parse_order(&args[0])?;
            Ok((order.unwrap_or(1), None))
        }
        2 => {
            let order = parse_order(&args[0])?.unwrap_or(1);
            let dim = parse_dimension_arg(&args[1])?;
            Ok((order, dim))
        }
        _ => Err(diff_invalid_argument("diff: unsupported arguments")),
    }
}

pub(super) fn parse_order(value: &Value) -> BuiltinResult<Option<usize>> {
    if is_empty_array(value) {
        return Ok(None);
    }
    match value {
        Value::Int(i) => i.try_to_usize().map(Some).ok_or_else(|| {
            diff_invalid_argument("diff: order must be a non-negative integer scalar")
        }),
        Value::Num(n) => parse_numeric_order(*n).map(Some),
        Value::Tensor(t) if tensor_element_len(t) == 1 => parse_tensor_order(t).map(Some),
        other => Err(diff_invalid_argument(format!(
            "diff: order must be a non-negative integer scalar, got {:?}",
            other
        ))),
    }
}

pub(super) fn parse_tensor_order(tensor: &Tensor) -> BuiltinResult<usize> {
    if let Some(storage) = tensor.integer_storage() {
        let value = storage
            .value_at(0)
            .ok_or_else(|| diff_invalid_argument("diff: integer order storage length mismatch"))?;
        return parse_integer_order(&value);
    }
    parse_numeric_order(tensor::tensor_value_f64(tensor, 0))
}

pub(super) fn parse_integer_order(value: &IntValue) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .ok_or_else(|| diff_invalid_argument("diff: order must be a non-negative integer scalar"))
}

pub(super) fn parse_numeric_order(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(diff_invalid_argument("diff: order must be finite"));
    }
    if value < 0.0 {
        return Err(diff_invalid_argument(
            "diff: order must be a non-negative integer scalar",
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(diff_invalid_argument(
            "diff: order must be a non-negative integer scalar",
        ));
    }
    runmat_types::decode_shape_dimension(value)
        .map_err(|_| diff_invalid_argument("diff: order is outside the supported platform range"))
}

pub(super) fn parse_dimension_arg(value: &Value) -> BuiltinResult<Option<usize>> {
    if is_empty_array(value) {
        return Ok(None);
    }
    match value {
        Value::Int(_) | Value::Num(_) => tensor::parse_dimension(value, "diff")
            .map(Some)
            .map_err(diff_invalid_argument),
        Value::Tensor(t) if tensor_element_len(t) == 1 => tensor::parse_dimension(value, "diff")
            .map(Some)
            .map_err(diff_invalid_argument),
        other => Err(diff_invalid_argument(format!(
            "diff: dimension must be a positive integer scalar, got {:?}",
            other
        ))),
    }
}

pub(super) fn is_empty_array(value: &Value) -> bool {
    matches!(value, Value::Tensor(t) if tensor_element_len(t) == 0)
}

pub(super) fn tensor_element_len(tensor: &Tensor) -> usize {
    tensor.len()
}
