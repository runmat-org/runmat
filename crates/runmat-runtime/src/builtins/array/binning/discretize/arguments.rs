use runmat_value::Value;
use std::borrow::Cow;

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
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Inputs> {
    super::admission::validate(&x, &edges_or_count, rest.view())?;
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
    // Restore every borrowed operand's residency after each download, before
    // the next operand is decoded, and on failure or cancellation.
    let _residency = protected
        .iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(gpu_helpers::ResidencyRestoreGuard::new(handle)),
            _ => None,
        })
        .collect::<Vec<_>>();
    runmat_accelerate_api::numeric_readback_for_handle(handle).map_err(|_| {
        error::transfer(
            crate::build_runtime_error("discretize: no exact input provider")
                .with_identifier("RunMat:gpu:ProviderOwnershipMismatch")
                .build(),
        )
    })?;
    gpu_helpers::gather_value_async(value)
        .await
        .map_err(error::transfer)
}

pub(super) fn scalar_text<'a>(value: &'a Value, context: &str) -> BuiltinResult<Cow<'a, str>> {
    validate_scalar_text(value, context)?;
    match value {
        Value::String(runmat_value::StringElement::Text(value)) => {
            Ok(Cow::Borrowed(value.as_str()))
        }
        Value::StringArray(value) => Ok(Cow::Borrowed(
            value.data()[0]
                .as_str()
                .expect("validated nonmissing scalar text"),
        )),
        Value::CharArray(value) => runmat_value::decode_utf16_code_units(value.data())
            .map(Cow::Owned)
            .map_err(|source| {
                let mut error = error::invalid(format!("discretize: {context}: {source}"));
                error.source = Some(Box::new(source));
                error
            }),
        other => Err(error::invalid(format!(
            "discretize: {context} must be text, got {other:?}"
        ))),
    }
}

pub(super) fn validate_scalar_text(value: &Value, context: &str) -> BuiltinResult<()> {
    match value {
        Value::String(runmat_value::StringElement::Text(_)) => Ok(()),
        Value::StringArray(value) if value.data().len() == 1 && !value.data()[0].is_missing() => {
            Ok(())
        }
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
