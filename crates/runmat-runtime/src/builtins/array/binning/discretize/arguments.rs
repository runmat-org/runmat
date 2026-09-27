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
    super::admission::validate(&x, &edges_or_count, &rest)?;
    let _metadata = [&x, &edges_or_count]
        .into_iter()
        .chain(rest.iter())
        .filter_map(|value| {
            if let Value::GpuTensor(handle) = value {
                Some(crate::builtins::common::gpu_helpers::HandleMetadataRestoreGuard::new(handle))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let x = gather_if_needed_async(&x).await.map_err(error::transfer)?;
    let edges_or_count = gather_if_needed_async(&edges_or_count)
        .await
        .map_err(error::transfer)?;
    let mut gathered_rest = Vec::with_capacity(rest.len());
    for value in &rest {
        gathered_rest.push(
            gather_if_needed_async(value)
                .await
                .map_err(error::transfer)?,
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
        Value::StringArray(value) if value.data.len() == 1 => Ok(value.data[0].clone()),
        Value::CharArray(value)
            if value.rows <= 1
                && crate::builtins::common::shape::effective_rank(&value.shape) <= 2 =>
        {
            Ok(value.data.iter().collect())
        }
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
