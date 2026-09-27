//! Parsed representation controls for the trigonometric `like` extension.
use crate::builtins::common::gpu_helpers;
use crate::{BuiltinResult, RuntimeError};
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;
use std::future::Future;

#[cfg(test)]
mod tests;

#[derive(Clone)]
pub(crate) enum Template {
    Default,
    Like(Prototype),
}

impl Template {
    pub(crate) fn resident(&self) -> Option<&GpuTensorHandle> {
        match self {
            Self::Like(prototype) => prototype.resident(),
            Self::Default => None,
        }
    }
}

#[derive(Clone)]
pub(crate) enum Prototype {
    HostReal,
    HostComplex,
    Resident {
        handle: GpuTensorHandle,
        complex: bool,
    },
}

#[derive(Clone, Copy)]
pub(crate) enum PrototypeError {
    Unsupported,
    MissingOwner,
    InvalidMetadata,
}

impl std::fmt::Display for PrototypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Unsupported => "'like' prototype must be numeric or a gpuArray",
            Self::MissingOwner => "no provider owns the 'like' prototype",
            Self::InvalidMetadata => "'like' prototype has invalid numeric storage metadata",
        })
    }
}

impl Prototype {
    pub(crate) fn parse(value: &Value) -> Result<Self, PrototypeError> {
        Ok(match value {
            Value::GpuTensor(handle) => {
                gpu_helpers::exact_provider_for_handle(handle)
                    .ok_or(PrototypeError::MissingOwner)?;
                let storage = runmat_accelerate_api::handle_storage(handle);
                if !gpu_helpers::expected_handle_numeric_element_type(handle).is_ok_and(|element| {
                    gpu_helpers::numeric_descriptor_matches(handle, element, storage)
                }) {
                    return Err(PrototypeError::InvalidMetadata);
                }
                Self::Resident {
                    handle: handle.clone(),
                    complex: storage == GpuTensorStorage::ComplexInterleaved,
                }
            }
            Value::Num(_)
            | Value::Tensor(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::LogicalArray(_) => Self::HostReal,
            Value::Complex(_, _) | Value::ComplexTensor(_) => Self::HostComplex,
            _ => return Err(PrototypeError::Unsupported),
        })
    }

    pub(crate) fn resident(&self) -> Option<&GpuTensorHandle> {
        match self {
            Self::Resident { handle, .. } => Some(handle),
            _ => None,
        }
    }
}

/// Own the computed resident intermediate until adaptation publishes it or a
/// replacement. Both the original and prepared input, plus the prototype, are
/// borrowed even when a provider legally aliases one as its result.
pub(crate) async fn evaluate<C, CF, A, AF>(
    value: Value,
    original: Option<&GpuTensorHandle>,
    template: &Template,
    compute: C,
    adapt: A,
    error: impl FnOnce(String) -> RuntimeError,
) -> BuiltinResult<Value>
where
    C: FnOnce(Value) -> CF,
    CF: Future<Output = BuiltinResult<Value>>,
    A: FnOnce(Value, Vec<GpuTensorHandle>) -> AF,
    AF: Future<Output = BuiltinResult<Value>>,
{
    let prepared = match &value {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };
    let protected: Vec<_> = original
        .into_iter()
        .chain(prepared.as_ref())
        .chain(template.resident())
        .cloned()
        .collect();
    let result = compute(value).await?;
    let mut adaptation_protected = protected.clone();
    if let Value::GpuTensor(handle) = &result {
        adaptation_protected.push(handle.clone());
    }
    let borrowed: Vec<_> = protected.iter().collect();
    let guard =
        crate::builtins::common::resident_output::ResidentResultGuard::new(&result, &borrowed)
            .map_err(error)?;
    let output = adapt(result, adaptation_protected).await?;
    guard.publish(&output);
    Ok(output)
}
