use runmat_value::Value;

use crate::{gather_if_needed_async, BuiltinResult};

use super::error;

pub(super) struct Inputs {
    pub x: Value,
    pub edges_or_count: Value,
    pub rest: Vec<Value>,
}

pub(super) async fn gather(
    x: Value,
    edges_or_count: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Inputs> {
    let x = gather_if_needed_async(&x).await.map_err(error::internal)?;
    let edges_or_count = gather_if_needed_async(&edges_or_count)
        .await
        .map_err(error::internal)?;
    let mut gathered_rest = Vec::with_capacity(rest.len());
    for value in rest {
        gathered_rest.push(
            gather_if_needed_async(&value)
                .await
                .map_err(error::internal)?,
        );
    }
    Ok(Inputs {
        x,
        edges_or_count,
        rest: gathered_rest,
    })
}

pub(super) fn scalar_text(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::CharArray(value) if value.rows <= 1 => Ok(value.data.iter().collect()),
        other => Err(error::invalid(format!(
            "discretize: {context} must be text, got {other:?}"
        ))),
    }
}

pub(super) fn is_option_name(value: &Value) -> bool {
    scalar_text(value, "option name")
        .map(|name| name.eq_ignore_ascii_case("IncludedEdge"))
        .unwrap_or(false)
}
