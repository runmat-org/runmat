use runmat_value::Value;

use crate::{builtins::common::gpu_helpers, BuiltinResult};

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
    let protected = [&x, &edges_or_count]
        .into_iter()
        .chain(rest.iter())
        .collect::<Vec<_>>();
    let gathered_x = gather_one(&x, &protected).await?;
    let gathered_edges = gather_one(&edges_or_count, &protected).await?;
    let mut gathered_rest = Vec::with_capacity(rest.len());
    for value in &rest {
        gathered_rest.push(gather_one(value, &protected).await?);
    }
    Ok(Inputs {
        x: gathered_x,
        edges_or_count: gathered_edges,
        rest: gathered_rest,
    })
}

async fn gather_one(value: &Value, protected: &[&Value]) -> BuiltinResult<Value> {
    let Value::GpuTensor(handle) = value else {
        return Ok(value.clone());
    };
    // Restore every borrowed operand after each download, before the next
    // operand's metadata is decoded, and on failure or cancellation.
    let _metadata = protected
        .iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(gpu_helpers::HandleMetadataRestoreGuard::new(handle)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let owner = gpu_helpers::exact_provider_for_handle(handle).ok_or_else(|| {
        error::transfer(
            crate::build_runtime_error("discretize: no exact input provider")
                .with_identifier("RunMat:gpu:ProviderOwnershipMismatch")
                .build(),
        )
    })?;
    gpu_helpers::download_value_preserving_residency_async(owner, handle)
        .await
        .map_err(error::transfer)
}

pub(super) fn scalar_text(value: &Value, context: &str) -> BuiltinResult<String> {
    validate_scalar_text(value, context)?;
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::StringArray(value) => Ok(value.data[0].clone()),
        Value::CharArray(value) => Ok(value.data.iter().collect()),
        other => Err(error::invalid(format!(
            "discretize: {context} must be text, got {other:?}"
        ))),
    }
}

pub(super) fn validate_scalar_text(value: &Value, context: &str) -> BuiltinResult<()> {
    match value {
        Value::String(_) => Ok(()),
        Value::StringArray(value) if value.data.len() == 1 => Ok(()),
        Value::CharArray(value)
            if value.rows <= 1
                && crate::builtins::common::shape::effective_rank(&value.shape) <= 2 =>
        {
            Ok(())
        }
        _ => Err(error::invalid(format!(
            "discretize: {context} must be scalar text"
        ))),
    }
}

pub(super) fn is_option_name(value: &Value) -> bool {
    scalar_text(value, "option name")
        .map(|name| name.eq_ignore_ascii_case("IncludedEdge"))
        .unwrap_or(false)
}
