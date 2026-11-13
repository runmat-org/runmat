//! Shared global-solver hybrid grammar. The caller supplies only its existing
//! error owner; a selector does not grant execution or capability authority.
use crate::{BuiltinResult, RuntimeError};
use runmat_value::Value;
pub(crate) struct Hybrid {
    pub(crate) kind: runmat_builtins::OptimHybridSolver,
    pub(crate) target: Value,
    pub(crate) options: Option<Value>,
}
pub(crate) fn parse(
    value: &Value,
    invalid: impl Fn(&str) -> RuntimeError,
) -> BuiltinResult<Option<Hybrid>> {
    if matches!(value,Value::Tensor(v)if v.is_empty()) {
        return Ok(None);
    }
    let (target, options) = if let Value::Cell(values) = value {
        if values.data().len() != 2 {
            return Err(invalid(
                "HybridFcn cell must contain a solver and its options",
            ));
        }
        (&values.data()[0], Some(values.data()[1].clone()))
    } else {
        (value, None)
    };
    if options.as_ref().is_some_and(|v| {
        !matches!(v, Value::Struct(_)) && !matches!(v,Value::Tensor(v)if v.is_empty())
    }) {
        return Err(invalid(
            "hybrid options must be a scalar options structure or []",
        ));
    }
    let name = match target {
        Value::FunctionHandle(v)
        | Value::ExternalFunctionHandle(v)
        | Value::MethodFunctionHandle(v) => Some(v.name().to_owned()),
        Value::BoundFunctionHandle(v) => Some(v.name.to_string()),
        _ => crate::builtins::common::validation::value_text_scalar_string(target)
            .map(|v| v.to_owned()),
    };
    let kind = name
        .and_then(|name| runmat_builtins::OptimHybridSolver::parse(&name))
        .ok_or_else(|| {
            invalid("HybridFcn must select fminsearch, fminunc, fmincon, or patternsearch")
        })?;
    Ok(Some(Hybrid {
        kind,
        target: target.clone(),
        options,
    }))
}
