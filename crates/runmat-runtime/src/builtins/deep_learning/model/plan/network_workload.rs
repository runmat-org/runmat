//! Borrowed, admitted network work facts for the shared placement boundary.
//!
//! This is not a provider capability or a cost estimate. Physical stage
//! lowering, scratch demand, and execution costs belong to the provider and
//! shared placement service after this semantic projection.

use runmat_accelerate_api::{
    AllocationIdentity, GpuHandleProvenance, GpuTensorLayout, ProviderPrecision,
};
use runmat_value::{NumericDType, Value};

use super::{axes::AxisPlan, host::HostNetworkPlan};
use crate::builtins::deep_learning::deep_learning_error;
use crate::builtins::deep_learning::model::{
    admission::forward_data,
    parameters::{ParameterKey, ParameterView},
    state::{StateKey, StateView},
};
use crate::BuiltinResult;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) enum PayloadClass {
    Numeric(NumericDType),
    Logical,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) enum PayloadStorage {
    Real,
    Complex,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) enum PayloadOrigin {
    Host,
    Provider {
        allocation: AllocationIdentity,
        provenance: Option<GpuHandleProvenance>,
        layout: GpuTensorLayout,
        /// Full allocation request, not the view's logical byte count.
        requested_bytes: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) struct PayloadFacts {
    pub class: PayloadClass,
    pub storage: PayloadStorage,
    pub shape: Vec<usize>,
    pub logical_bytes: u64,
    pub origin: PayloadOrigin,
}

fn logical_bytes(shape: &[usize], bytes_per_element: usize, lanes: usize) -> Option<u64> {
    shape
        .iter()
        .try_fold(1_u64, |total, extent| {
            total.checked_mul(u64::try_from(*extent).ok()?)
        })?
        .checked_mul(u64::try_from(bytes_per_element).ok()?)?
        .checked_mul(u64::try_from(lanes).ok()?)
}

impl PayloadFacts {
    fn from_admitted(value: &Value, function: &'static str) -> BuiltinResult<Self> {
        let data = forward_data(value);
        let (class, storage, shape, origin) = match data {
            Value::Tensor(tensor) => (
                PayloadClass::Numeric(tensor.numeric_dtype()),
                PayloadStorage::Real,
                tensor.shape.clone(),
                PayloadOrigin::Host,
            ),
            Value::ComplexTensor(tensor) => (
                PayloadClass::Numeric(tensor.numeric_dtype()),
                PayloadStorage::Complex,
                tensor.shape.clone(),
                PayloadOrigin::Host,
            ),
            Value::LogicalArray(array) => (
                PayloadClass::Logical,
                PayloadStorage::Real,
                array.shape.clone(),
                PayloadOrigin::Host,
            ),
            Value::GpuTensor(handle) => {
                let dtype = match runmat_accelerate_api::handle_precision(handle) {
                    Some(ProviderPrecision::F32) => NumericDType::F32,
                    Some(ProviderPrecision::F64) => NumericDType::F64,
                    None => {
                        return Err(deep_learning_error(
                            function,
                            "unsupported resident numeric class",
                        ))
                    }
                };
                (
                    PayloadClass::Numeric(dtype),
                    PayloadStorage::Real,
                    handle.shape().to_vec(),
                    PayloadOrigin::Provider {
                        allocation: handle.allocation_identity(),
                        provenance: runmat_accelerate_api::handle_provenance(handle),
                        layout: handle.view().layout(),
                        requested_bytes: handle.physical().requested_bytes(),
                    },
                )
            }
            Value::Num(_) => (
                PayloadClass::Numeric(NumericDType::F64),
                PayloadStorage::Real,
                vec![1, 1],
                PayloadOrigin::Host,
            ),
            Value::Int(value) => (
                PayloadClass::Numeric(value.numeric_dtype()),
                PayloadStorage::Real,
                vec![1, 1],
                PayloadOrigin::Host,
            ),
            Value::Bool(_) => (
                PayloadClass::Logical,
                PayloadStorage::Real,
                vec![1, 1],
                PayloadOrigin::Host,
            ),
            _ => {
                return Err(deep_learning_error(
                    function,
                    "admitted network payload has no numeric placement facts",
                ))
            }
        };
        let element_bytes = match class {
            PayloadClass::Numeric(dtype) => dtype.byte_size(),
            PayloadClass::Logical => 1,
        };
        let lanes = if storage == PayloadStorage::Complex {
            2
        } else {
            1
        };
        let logical_bytes = logical_bytes(&shape, element_bytes, lanes)
            .ok_or_else(|| deep_learning_error(function, "network payload byte size overflows"))?;
        Ok(Self {
            class,
            storage,
            shape,
            logical_bytes,
            origin,
        })
    }
}

pub(in crate::builtins::deep_learning) struct NetworkResource<'a, Key> {
    pub key: &'a Key,
    pub payload: PayloadFacts,
}

/// A borrowed view of the already-admitted semantic plan. `steps` retain the
/// canonical typed controls and per-layer axes; no parallel layer schema is
/// introduced here.
pub(in crate::builtins::deep_learning) struct NetworkWorkload<'a> {
    pub steps: &'a [super::host::StepDescriptor],
    pub input_axes: &'a AxisPlan,
    pub source_output_axes: &'a AxisPlan,
    pub target_output_axes: &'a AxisPlan,
    pub input: PayloadFacts,
    pub parameters: Vec<NetworkResource<'a, ParameterKey>>,
    pub state: Vec<NetworkResource<'a, StateKey>>,
    pub execution_precision: NumericDType,
    pub output_elements: u64,
    /// Logical output bytes only if the selected route produces a real array
    /// in the admitted execution precision. This is not device allocation
    /// demand and does not claim a complex/host/device output contract.
    pub output_bytes_if_real: u64,
    /// Upper-level semantic geometry does not establish provider scratch or
    /// an exact output representation. The provider must supply those facts.
    pub scratch_bytes: Option<u64>,
}

impl<'a> NetworkWorkload<'a> {
    pub fn from_admitted(
        plan: &'a HostNetworkPlan<'_>,
        input: &Value,
        target_output_axes: &'a AxisPlan,
    ) -> BuiltinResult<Self> {
        let function = plan.function();
        let parameters = plan
            .parameter_views()
            .iter()
            .map(|row: &ParameterView<'_>| {
                let value = row.value.ok_or_else(|| {
                    deep_learning_error(function, "admitted network has an unfilled parameter")
                })?;
                Ok(NetworkResource {
                    key: &row.key,
                    payload: PayloadFacts::from_admitted(value, function)?,
                })
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        let state = plan
            .state_views()
            .iter()
            .map(|row: &StateView<'_>| {
                Ok(NetworkResource {
                    key: &row.key,
                    payload: PayloadFacts::from_admitted(row.value, function)?,
                })
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        let output_elements = u64::try_from(target_output_axes.element_count())
            .map_err(|_| deep_learning_error(function, "network output size overflows"))?;
        let output_bytes_if_real = output_elements
            .checked_mul(
                u64::try_from(plan.execution_precision().byte_size()).map_err(|_| {
                    deep_learning_error(function, "network output element size overflows")
                })?,
            )
            .ok_or_else(|| deep_learning_error(function, "network output byte size overflows"))?;
        Ok(Self {
            steps: plan.steps(),
            input_axes: plan.input_axes(),
            source_output_axes: plan.output_axes(),
            target_output_axes,
            input: PayloadFacts::from_admitted(input, function)?,
            parameters,
            state,
            execution_precision: plan.execution_precision(),
            output_elements,
            output_bytes_if_real,
            scratch_bytes: None,
        })
    }
}

#[cfg(test)]
mod tests;
