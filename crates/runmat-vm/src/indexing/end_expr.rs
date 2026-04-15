use crate::bytecode::{EndExpr, UserFunction};
use crate::interpreter::errors::mex;
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone, Copy)]
pub struct ValueToF64Error;

pub type BuiltinEndCallback<'a> = dyn Fn(
        &'a str,
        Vec<Value>,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Value>, RuntimeError>> + 'a>>
    + 'a;

pub type UserEndCallback<'a> = dyn Fn(
        &'a str,
        Vec<Value>,
        &'a HashMap<String, UserFunction>,
        &'a [Value],
    ) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'a>>
    + 'a;

pub fn value_to_f64(v: &Value) -> Result<f64, ValueToF64Error> {
    match v {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if t.data.len() == 1 => Ok(t.data[0]),
        Value::Complex(re, im) if im.abs() < 1e-12 => Ok(*re),
        Value::ComplexTensor(ct) if ct.data.len() == 1 && ct.data[0].1.abs() < 1e-12 => {
            Ok(ct.data[0].0)
        }
        _ => Err(ValueToF64Error),
    }
}

pub fn eval_end_expr_value<'a>(
    expr: &'a EndExpr,
    end_value: f64,
    vars: &'a [Value],
    functions: &'a HashMap<String, UserFunction>,
    call_builtin: &'a BuiltinEndCallback<'a>,
    call_user: &'a UserEndCallback<'a>,
) -> Pin<Box<dyn Future<Output = Result<f64, RuntimeError>> + 'a>> {
    Box::pin(async move {
        match expr {
            EndExpr::End => Ok(end_value),
            EndExpr::Const(v) => Ok(*v),
            EndExpr::Var(i) => {
                let v = vars.get(*i).ok_or_else(|| {
                    mex("MissingNumericIndex", "missing variable for end expression")
                })?;
                value_to_f64(v)
                    .map_err(|_| mex("UnsupportedIndexType", "end expression must be numeric"))
            }
            EndExpr::Call(name, args) => {
                let mut argv: Vec<Value> = Vec::with_capacity(args.len());
                for a in args {
                    let val =
                        eval_end_expr_value(a, end_value, vars, functions, call_builtin, call_user)
                            .await?;
                    argv.push(Value::Num(val));
                }
                let v = if let Some(v) = call_builtin(name, argv.clone()).await? {
                    v
                } else if functions.contains_key(name) {
                    call_user(name, argv, functions, vars).await?
                } else {
                    return Err(mex(
                        "UndefinedFunction",
                        &format!("Undefined function in end expression: {name}"),
                    ));
                };
                value_to_f64(&v)
                    .map_err(|_| mex("UnsupportedIndexType", "end call must return scalar"))
            }
            EndExpr::Add(a, b) => {
                let lhs =
                    eval_end_expr_value(a, end_value, vars, functions, call_builtin, call_user)
                        .await?;
                let rhs =
                    eval_end_expr_value(b, end_value, vars, functions, call_builtin, call_user)
                        .await?;
                Ok(lhs + rhs)
            }
            EndExpr::Sub(a, b) => {
                let lhs =
                    eval_end_expr_value(a, end_value, vars, functions, call_builtin, call_user)
                        .await?;
                let rhs =
                    eval_end_expr_value(b, end_value, vars, functions, call_builtin, call_user)
                        .await?;
                Ok(lhs - rhs)
            }
            EndExpr::Mul(a, b) => {
                let lhs =
                    eval_end_expr_value(a, end_value, vars, functions, call_builtin, call_user)
                        .await?;
                let rhs =
                    eval_end_expr_value(b, end_value, vars, functions, call_builtin, call_user)
                        .await?;
                Ok(lhs * rhs)
            }
            EndExpr::Div(a, b) => {
                let denom =
                    eval_end_expr_value(b, end_value, vars, functions, call_builtin, call_user)
                        .await?;
                if denom == 0.0 {
                    return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                }
                let lhs =
                    eval_end_expr_value(a, end_value, vars, functions, call_builtin, call_user)
                        .await?;
                Ok(lhs / denom)
            }
            EndExpr::LeftDiv(a, b) => {
                let denom =
                    eval_end_expr_value(a, end_value, vars, functions, call_builtin, call_user)
                        .await?;
                if denom == 0.0 {
                    return Err(mex("IndexOutOfBounds", "Index out of bounds"));
                }
                let rhs =
                    eval_end_expr_value(b, end_value, vars, functions, call_builtin, call_user)
                        .await?;
                Ok(rhs / denom)
            }
            EndExpr::Pow(a, b) => {
                let lhs =
                    eval_end_expr_value(a, end_value, vars, functions, call_builtin, call_user)
                        .await?;
                let rhs =
                    eval_end_expr_value(b, end_value, vars, functions, call_builtin, call_user)
                        .await?;
                Ok(lhs.powf(rhs))
            }
            EndExpr::Neg(a) => {
                Ok(
                    -eval_end_expr_value(a, end_value, vars, functions, call_builtin, call_user)
                        .await?,
                )
            }
            EndExpr::Pos(a) => {
                Ok(
                    eval_end_expr_value(a, end_value, vars, functions, call_builtin, call_user)
                        .await?,
                )
            }
            EndExpr::Floor(a) => {
                Ok(
                    eval_end_expr_value(a, end_value, vars, functions, call_builtin, call_user)
                        .await?
                        .floor(),
                )
            }
            EndExpr::Ceil(a) => {
                Ok(
                    eval_end_expr_value(a, end_value, vars, functions, call_builtin, call_user)
                        .await?
                        .ceil(),
                )
            }
            EndExpr::Round(a) => {
                Ok(
                    eval_end_expr_value(a, end_value, vars, functions, call_builtin, call_user)
                        .await?
                        .round(),
                )
            }
            EndExpr::Fix(a) => {
                let v = eval_end_expr_value(a, end_value, vars, functions, call_builtin, call_user)
                    .await?;
                Ok(if v >= 0.0 { v.floor() } else { v.ceil() })
            }
        }
    })
}

pub async fn resolve_range_end_index<'a>(
    dim_len: usize,
    end_expr: &'a EndExpr,
    vars: &'a [Value],
    functions: &'a HashMap<String, UserFunction>,
    call_builtin: &'a BuiltinEndCallback<'a>,
    call_user: &'a UserEndCallback<'a>,
) -> Result<i64, RuntimeError> {
    let value = eval_end_expr_value(
        end_expr,
        dim_len as f64,
        vars,
        functions,
        call_builtin,
        call_user,
    )
    .await?;
    Ok(value.floor() as i64)
}
