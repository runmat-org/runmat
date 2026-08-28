use std::fmt;

use runmat_execution::PoolBackend;
use runmat_types::{EffectKind, ParallelVariableRole};
use runmat_value::Value;

use crate::BytecodeParforRegion;

/// Values and frame updates prepared for a portable parallel worker boundary.
///
/// The task inputs are host-transferable values. Sliced destinations that were
/// automatically resident are replaced in the coordinator frame with the same
/// gathered host value, so result assembly uses the representation sent to the
/// workers. Reduction accumulators stay unchanged in the coordinator frame;
/// only their worker input is replaced by the operator identity.
pub(crate) struct PreparedParallelInputs {
    pub(crate) task_inputs: Vec<Value>,
    pub(crate) sliced_destinations: Vec<(usize, Value)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ParallelPlacementRejection {
    NoWorkerCapacity,
    UnsafeEffect(EffectKind),
    StaticValueNotTransferable { slot: usize },
    ExplicitDeviceIntent { slot: usize },
    AutomaticGatherFailed { slot: usize, reason: String },
    UnsupportedSlicedStorage { slot: usize, family: &'static str },
    RuntimeValueNotTransferable { slot: usize, reason: String },
}

impl fmt::Display for ParallelPlacementRejection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoWorkerCapacity => {
                formatter.write_str("the active pool has no parallel worker budget")
            }
            Self::UnsafeEffect(effect) => write!(
                formatter,
                "the region contains the non-parallel effect {effect:?}"
            ),
            Self::StaticValueNotTransferable { slot } => {
                write!(formatter, "compiled slot {slot} is not transferable")
            }
            Self::ExplicitDeviceIntent { slot } => write!(
                formatter,
                "compiled slot {slot} carries explicit device-residency intent"
            ),
            Self::AutomaticGatherFailed { slot, reason } => write!(
                formatter,
                "automatic residency for compiled slot {slot} could not be gathered: {reason}"
            ),
            Self::UnsupportedSlicedStorage { slot, family } => write!(
                formatter,
                "compiled sliced slot {slot} uses unsupported {family} storage"
            ),
            Self::RuntimeValueNotTransferable { slot, reason } => write!(
                formatter,
                "compiled slot {slot} failed runtime transfer validation: {reason}"
            ),
        }
    }
}

pub(crate) async fn prepare(
    executable: &BytecodeParforRegion,
    inputs: &[Value],
    backend: PoolBackend,
    worker_budget: u32,
) -> Result<PreparedParallelInputs, ParallelPlacementRejection> {
    if backend == PoolBackend::Serial || worker_budget <= 1 {
        return Err(ParallelPlacementRejection::NoWorkerCapacity);
    }
    if let Some(effect) = executable
        .contract
        .effects
        .0
        .iter()
        .copied()
        .find(|effect| !matches!(effect, EffectKind::MayThrow | EffectKind::Randomness))
    {
        return Err(ParallelPlacementRejection::UnsafeEffect(effect));
    }

    let mut task_inputs = Vec::with_capacity(inputs.len());
    let mut sliced_destinations = Vec::new();
    for (variable, value) in executable.input_variables().zip(inputs) {
        let slot = variable.slot;
        if !variable.contract.transferable {
            return Err(ParallelPlacementRejection::StaticValueNotTransferable { slot });
        }
        if runmat_runtime::dispatcher::value_contains_explicit_gpu(value) {
            return Err(ParallelPlacementRejection::ExplicitDeviceIntent { slot });
        }

        let host_value = runmat_runtime::dispatcher::gather_if_needed_async(value)
            .await
            .map_err(|error| ParallelPlacementRejection::AutomaticGatherFailed {
                slot,
                reason: error.to_string(),
            })?;
        if matches!(variable.contract.role, ParallelVariableRole::Sliced { .. })
            && !runmat_runtime::parallel::assembly::supports_sliced_assembly(&host_value)
        {
            return Err(ParallelPlacementRejection::UnsupportedSlicedStorage {
                slot,
                family: value_family(&host_value),
            });
        }
        runmat_runtime::execution::validate_spawn_capture(&host_value).map_err(|error| {
            ParallelPlacementRejection::RuntimeValueNotTransferable {
                slot,
                reason: error.to_string(),
            }
        })?;

        match variable.contract.role {
            ParallelVariableRole::Reduction { operator } => {
                let identity = runmat_runtime::parallel::reduction::identity(operator, &host_value)
                    .await
                    .map_err(
                        |error| ParallelPlacementRejection::RuntimeValueNotTransferable {
                            slot,
                            reason: error.to_string(),
                        },
                    )?;
                task_inputs.push(identity);
            }
            ParallelVariableRole::Sliced { .. } => {
                sliced_destinations.push((slot, host_value.clone()));
                task_inputs.push(host_value);
            }
            _ => task_inputs.push(host_value),
        }
    }

    Ok(PreparedParallelInputs {
        task_inputs,
        sliced_destinations,
    })
}

fn value_family(value: &Value) -> &'static str {
    match value {
        Value::Num(_) => "double scalar",
        Value::Int(_) => "integer scalar",
        Value::Tensor(_) => "numeric array",
        Value::Complex(_, _) => "complex scalar",
        Value::ComplexTensor(_) => "complex array",
        Value::LogicalArray(_) => "logical array",
        Value::SparseTensor(_) => "sparse array",
        Value::Cell(_) => "cell array",
        Value::GpuTensor(_) => "resident array",
        _ => "other",
    }
}

#[cfg(all(test, feature = "native-accel"))]
mod tests {
    use runmat_accelerate_api::{GpuHandleProvenance, GpuTensorHandle};

    use super::*;

    #[test]
    fn explicit_device_intent_has_a_distinct_placement_rejection() {
        let value = Value::GpuTensor(
            GpuTensorHandle::new(vec![2, 2], 1, 7).with_provenance(GpuHandleProvenance::Explicit),
        );
        assert!(runmat_runtime::dispatcher::value_contains_explicit_gpu(
            &value
        ));
        assert!(!runmat_runtime::dispatcher::value_contains_explicit_gpu(
            &Value::GpuTensor(
                GpuTensorHandle::new(vec![2, 2], 1, 8)
                    .with_provenance(GpuHandleProvenance::Automatic),
            )
        ));
    }
}
