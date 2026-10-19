use std::fmt;

use runmat_execution::PoolBackend;
use runmat_runtime::execution::{RootedValueLease, RootedValuesBuilder};
use runmat_types::{EffectKind, EffectSet, ParallelTransferability, ParallelVariableRole};
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
    pub(crate) task_roots: RootedValueLease,
    pub(crate) sliced_destinations: Vec<(usize, Value)>,
}

#[derive(Debug)]
pub(crate) enum ParallelPlacementRejection {
    NoWorkerCapacity,
    UnsafeEffect(EffectKind),
    StaticValueNotTransferable {
        slot: usize,
    },
    ExplicitDeviceIntent {
        slot: usize,
    },
    AutomaticGatherFailed {
        slot: usize,
        error: runmat_runtime::runtime_error::RuntimeErrorOwner,
    },
    UnsupportedSlicedStorage {
        slot: usize,
        family: &'static str,
    },
    RuntimeValueNotTransferable {
        slot: usize,
        error: runmat_runtime::runtime_error::RuntimeErrorOwner,
    },
    ReductionIdentityFailed {
        slot: usize,
        error: runmat_runtime::runtime_error::RuntimeErrorOwner,
    },
    InputRootingFailed {
        error: runmat_runtime::execution::ExecutionServiceError,
    },
    InputMaterializationFailed {
        error: runmat_runtime::runtime_error::RuntimeErrorOwner,
    },
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
            Self::AutomaticGatherFailed { slot, error } => write!(
                formatter,
                "automatic residency for compiled slot {slot} could not be gathered: {error}"
            ),
            Self::UnsupportedSlicedStorage { slot, family } => write!(
                formatter,
                "compiled sliced slot {slot} uses unsupported {family} storage"
            ),
            Self::RuntimeValueNotTransferable { slot, error } => write!(
                formatter,
                "compiled slot {slot} failed runtime transfer validation: {error}"
            ),
            Self::ReductionIdentityFailed { slot, error } => write!(
                formatter,
                "parallel reduction identity for compiled slot {slot} failed: {error}"
            ),
            Self::InputRootingFailed { error } => {
                write!(formatter, "parallel task input ownership failed: {error}")
            }
            Self::InputMaterializationFailed { error } => {
                write!(
                    formatter,
                    "parallel task input materialization failed: {error}"
                )
            }
        }
    }
}

pub(crate) async fn prepare(
    runtime: &runmat_runtime::context::RuntimeContext,
    executable: &BytecodeParforRegion,
    inputs: &[Value],
    backend: PoolBackend,
    worker_budget: u32,
) -> Result<PreparedParallelInputs, ParallelPlacementRejection> {
    if backend == PoolBackend::Serial || worker_budget <= 1 {
        return Err(ParallelPlacementRejection::NoWorkerCapacity);
    }
    if let Some(effect) = unsupported_worker_effect(backend, &executable.contract.effects) {
        return Err(ParallelPlacementRejection::UnsafeEffect(effect));
    }

    let mut task_inputs = RootedValuesBuilder::with_capacity(inputs.len());
    let mut sliced_destinations = Vec::new();
    for (variable, value) in executable.input_variables().zip(inputs) {
        let slot = variable.slot;
        if variable.contract.transferability == ParallelTransferability::NonTransferable {
            return Err(ParallelPlacementRejection::StaticValueNotTransferable { slot });
        }
        if runmat_runtime::dispatcher::value_contains_explicit_gpu(value) {
            return Err(ParallelPlacementRejection::ExplicitDeviceIntent { slot });
        }

        let host_value = runmat_runtime::dispatcher::gather_if_needed_async(value)
            .await
            .map_err(|error| ParallelPlacementRejection::AutomaticGatherFailed {
                slot,
                error: error.into(),
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
                error: error.into(),
            }
        })?;

        match variable.contract.role {
            ParallelVariableRole::Reduction { operator } => {
                let identity = runmat_runtime::parallel::reduction::identity_in_context(
                    Some(runtime),
                    operator,
                    &host_value,
                )
                .await
                .map_err(|error| {
                    ParallelPlacementRejection::ReductionIdentityFailed {
                        slot,
                        error: error.into(),
                    }
                })?;
                identity
                    .with_owned(|value| task_inputs.push(value))
                    .map_err(|error| ParallelPlacementRejection::InputRootingFailed { error })?;
            }
            ParallelVariableRole::Sliced { .. } => {
                sliced_destinations.push((slot, host_value.clone()));
                task_inputs
                    .push(host_value)
                    .map_err(|error| ParallelPlacementRejection::InputRootingFailed { error })?;
            }
            _ => task_inputs
                .push(host_value)
                .map_err(|error| ParallelPlacementRejection::InputRootingFailed { error })?,
        }
    }

    let task_inputs = task_inputs
        .finish()
        .map_err(|error| ParallelPlacementRejection::InputRootingFailed { error })?;
    let (task_inputs, task_roots) = task_inputs
        .with_unbounded_values(|values| {
            RootedValueLease::for_values(values.iter()).map(|roots| (values, roots))
        })
        .map_err(
            |error| ParallelPlacementRejection::InputMaterializationFailed {
                error: error.into(),
            },
        )?
        .map_err(|error| ParallelPlacementRejection::InputRootingFailed { error })?;

    Ok(PreparedParallelInputs {
        task_inputs,
        task_roots,
        sliced_destinations,
    })
}

/// A suspended builtin call is portable through the native process and browser
/// worker program executors: both await the VM region interpreter. This is not
/// authority for ambient effects (I/O, workspace mutation, callbacks, etc.),
/// nor proof that an arbitrary remote backend provides the same async contract.
fn unsupported_worker_effect(backend: PoolBackend, effects: &EffectSet) -> Option<EffectKind> {
    effects.0.iter().copied().find(|effect| match effect {
        EffectKind::MayThrow | EffectKind::Randomness => false,
        EffectKind::MaySuspend => !matches!(
            backend,
            PoolBackend::LocalProcesses | PoolBackend::BrowserWorkers
        ),
        _ => true,
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

#[cfg(test)]
mod tests {
    #[cfg(feature = "native-accel")]
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, GpuHandleProvenance, GpuTensorStorage,
        NumericElementType,
    };

    use super::*;

    #[test]
    fn worker_async_admission_is_backend_and_effect_specific() {
        let effects = EffectSet(
            [EffectKind::MayThrow, EffectKind::MaySuspend]
                .into_iter()
                .collect(),
        );
        assert_eq!(
            unsupported_worker_effect(PoolBackend::LocalProcesses, &effects),
            None
        );
        assert_eq!(
            unsupported_worker_effect(PoolBackend::BrowserWorkers, &effects),
            None
        );
        assert_eq!(
            unsupported_worker_effect(PoolBackend::Remote, &effects),
            Some(EffectKind::MaySuspend)
        );

        for unsafe_effect in [
            EffectKind::WorkspaceWrite,
            EffectKind::FilesystemRead,
            EffectKind::HostCallback,
            EffectKind::Unknown,
        ] {
            let effects = EffectSet([unsafe_effect].into_iter().collect());
            assert_eq!(
                unsupported_worker_effect(PoolBackend::LocalProcesses, &effects),
                Some(unsafe_effect)
            );
        }
        let randomness = EffectSet([EffectKind::Randomness].into_iter().collect());
        assert_eq!(
            unsupported_worker_effect(PoolBackend::Remote, &randomness),
            None,
            "randomness admission remains governed by the existing policy"
        );
    }

    #[test]
    fn ordinary_numeric_parfor_retains_suspend_effect_and_passes_native_admission() {
        let ast =
            runmat_parser::parse("out = cell(1, 2); parfor i = 1:2; out{i} = sum(sin(i)); end")
                .expect("parse numeric parfor");
        let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty())
            .expect("lower numeric parfor HIR");
        let mir =
            runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower numeric parfor MIR");
        let analysis = runmat_mir::analysis::analyze_assembly(&mir);
        let region = analysis
            .parallel
            .parfor_regions
            .first()
            .expect("classified numeric parfor");
        assert!(region.effects.0.contains(&EffectKind::MaySuspend));
        assert_eq!(
            unsupported_worker_effect(PoolBackend::LocalProcesses, &region.effects),
            None
        );
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn explicit_device_intent_has_a_distinct_placement_rejection() {
        let owner = TestAllocationOwner::new();
        let allocation = owner
            .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let value = Value::GpuTensor(
            allocation
                .handle
                .with_provenance(GpuHandleProvenance::Explicit)
                .unwrap(),
        );
        assert!(runmat_runtime::dispatcher::value_contains_explicit_gpu(
            &value
        ));
        assert!(!runmat_runtime::dispatcher::value_contains_explicit_gpu(
            &Value::GpuTensor(
                allocation
                    .handle
                    .with_provenance(GpuHandleProvenance::Automatic)
                    .unwrap(),
            )
        ));
        drop(value);
        assert_eq!(allocation.probe.retirements(), 0);
        drop(allocation.handle);
        assert_eq!(allocation.probe.retirements(), 1);
    }
}
