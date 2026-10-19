//! Parsed representation controls for the trigonometric `like` extension.
use crate::builtins::common::gpu_helpers;
use crate::execution::{RootedValue, RootedValueLease};
use crate::BuiltinResult;
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;
use std::future::Future;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod ownership_tests;

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
) -> BuiltinResult<RootedValue>
where
    C: FnOnce(Value, Vec<GpuTensorHandle>) -> CF,
    CF: Future<Output = BuiltinResult<RootedValue>>,
    A: FnOnce(Value, Vec<GpuTensorHandle>) -> AF,
    AF: Future<Output = BuiltinResult<RootedValue>>,
{
    let _input = RootedValueLease::for_values(std::iter::once(&value))
        .map_err(|cause| cause.into_runtime_error())?;
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
    let protected_values: Vec<_> = protected.iter().cloned().map(Value::GpuTensor).collect();
    let _protected_roots = RootedValueLease::for_values(protected_values.iter())
        .map_err(|cause| cause.into_runtime_error())?;
    let result = compute(value, protected.clone()).await?;
    let mut adaptation_protected = protected.clone();
    if let Value::GpuTensor(handle) = result.value() {
        adaptation_protected.push(handle.clone());
    }
    adapt(result.value().clone(), adaptation_protected).await
}
