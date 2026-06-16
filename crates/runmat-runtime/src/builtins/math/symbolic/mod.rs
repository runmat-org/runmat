pub(crate) mod limit;
pub(crate) mod sym;
pub(crate) mod syms;

use runmat_builtins::{
    symbolic::{is_valid_symbolic_identifier, SymbolicFunction},
    SymbolicExpr, Tensor, Value,
};

#[derive(Debug, Clone, Copy)]
pub(crate) enum SymbolicBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

pub(crate) fn symbolic_binary(lhs: &Value, rhs: &Value, op: SymbolicBinaryOp) -> Option<Value> {
    if !matches!(lhs, Value::Symbolic(_)) && !matches!(rhs, Value::Symbolic(_)) {
        return None;
    }
    let lhs = value_to_symbolic_scalar(lhs)?;
    let rhs = value_to_symbolic_scalar(rhs)?;
    let expr = match op {
        SymbolicBinaryOp::Add => SymbolicExpr::add_expr(lhs, rhs),
        SymbolicBinaryOp::Sub => SymbolicExpr::sub_expr(lhs, rhs),
        SymbolicBinaryOp::Mul => SymbolicExpr::mul_expr(lhs, rhs),
        SymbolicBinaryOp::Div => SymbolicExpr::div_expr(lhs, rhs),
        SymbolicBinaryOp::Pow => SymbolicExpr::pow_expr(lhs, rhs),
    };
    Some(symbolic_expr_to_value(expr))
}

pub(crate) fn symbolic_function(value: &Value, function: SymbolicFunction) -> Option<Value> {
    let expr = match value {
        Value::Symbolic(expr) => expr.clone(),
        _ => return None,
    };
    Some(symbolic_expr_to_value(SymbolicExpr::function(
        function, expr,
    )))
}

pub(crate) fn value_to_symbolic_scalar(value: &Value) -> Option<SymbolicExpr> {
    match value {
        Value::Symbolic(expr) => Some(expr.clone()),
        Value::Num(value) => Some(SymbolicExpr::constant(*value)),
        Value::Int(value) => Some(SymbolicExpr::constant(value.to_f64())),
        Value::Bool(value) => Some(SymbolicExpr::constant(if *value { 1.0 } else { 0.0 })),
        Value::Tensor(tensor) if tensor.data.len() == 1 => {
            Some(SymbolicExpr::constant(tensor.data[0]))
        }
        _ => None,
    }
}

pub(crate) fn symbolic_expr_to_value(expr: SymbolicExpr) -> Value {
    Value::Symbolic(expr)
}

pub(crate) fn symbolic_variable_name_from_value(value: &Value) -> Option<String> {
    match value {
        Value::Symbolic(expr) => expr.variable_name().map(ToOwned::to_owned),
        _ => text_scalar(value).map(|text| text.trim().to_string()),
    }
    .filter(|name| is_valid_symbolic_identifier(name))
}

pub(crate) fn empty_return_value() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

pub(crate) fn text_scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Some(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        _ => None,
    }
}

pub(crate) fn is_valid_identifier(name: &str) -> bool {
    is_valid_symbolic_identifier(name)
}
