//! The selected host computation's allocation route and workspace quote.
//! Both routes fill one final native-class output; only finite managed work
//! needs the scratch claim owners in addition to its numeric stage payload.

use crate::context::host_allocation::HostAllocationPlan;
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult};
use runmat_execution::ResourceByteBound;
use runmat_value::NumericDType;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExactHostRandomMode {
    Managed,
    Unbounded,
}

pub(crate) fn host_random_mode(runtime: Option<&RuntimeContext>) -> ExactHostRandomMode {
    if runtime.is_some_and(|runtime| {
        runtime.service_ports().parallel().is_some_and(|parallel| {
            parallel
                .placement_resources()
                .memory_available_bytes
                .is_some()
        })
    }) {
        ExactHostRandomMode::Managed
    } else {
        ExactHostRandomMode::Unbounded
    }
}

pub(crate) enum ExactHostRandomWorkspace {
    DirectOutput,
    IntegerOffsets { count: usize },
    Permutation { population: usize, count: usize },
}

pub(crate) fn exact_host_workspace(
    label: &'static str,
    work: ExactHostRandomWorkspace,
    mode: ExactHostRandomMode,
) -> BuiltinResult<ResourceByteBound> {
    let numeric = |dtype, count| match mode {
        ExactHostRandomMode::Managed => HostAllocationPlan::numeric_workspace_bytes(dtype, count)
            .map_err(|source| {
                build_runtime_error("random exact host workspace quote failed")
                    .with_builtin(label)
                    .with_source(source)
                    .build()
            }),
        ExactHostRandomMode::Unbounded => {
            let bytes = match dtype {
                NumericDType::F64 => std::alloc::Layout::array::<f64>(count),
                NumericDType::U64 => std::alloc::Layout::array::<u64>(count),
                _ => unreachable!("random native workspace has only F64/U64 stages"),
            }
            .map_err(|source| {
                build_runtime_error("random native workspace layout overflow")
                    .with_builtin(label)
                    .with_source(source)
                    .build()
            })?
            .size();
            u64::try_from(bytes).map_err(|source| {
                build_runtime_error("random native workspace byte count overflow")
                    .with_builtin(label)
                    .with_source(source)
                    .build()
            })
        }
    };
    let bytes = match work {
        ExactHostRandomWorkspace::DirectOutput => 0,
        ExactHostRandomWorkspace::IntegerOffsets { count } => numeric(NumericDType::U64, count)?,
        ExactHostRandomWorkspace::Permutation { population, count } => {
            numeric(NumericDType::F64, population)?
                .checked_add(numeric(NumericDType::U64, count)?)
                .ok_or_else(|| {
                    build_runtime_error("random exact host workspace quote overflow")
                        .with_builtin(label)
                        .build()
                })?
        }
    };
    Ok(ResourceByteBound::Known(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_host_quotes_match_selected_workspace_owners() {
        assert_eq!(
            exact_host_workspace(
                "rand",
                ExactHostRandomWorkspace::DirectOutput,
                ExactHostRandomMode::Managed
            )
            .unwrap(),
            ResourceByteBound::Known(0)
        );
        assert_eq!(
            exact_host_workspace(
                "randn",
                ExactHostRandomWorkspace::DirectOutput,
                ExactHostRandomMode::Unbounded
            )
            .unwrap(),
            ResourceByteBound::Known(0)
        );
        let managed_offsets =
            HostAllocationPlan::numeric_workspace_bytes(NumericDType::U64, 7).unwrap();
        assert_eq!(
            exact_host_workspace(
                "randi",
                ExactHostRandomWorkspace::IntegerOffsets { count: 7 },
                ExactHostRandomMode::Managed
            )
            .unwrap(),
            ResourceByteBound::Known(managed_offsets)
        );
        let managed_permutation =
            HostAllocationPlan::numeric_workspace_bytes(NumericDType::F64, 11)
                .unwrap()
                .checked_add(
                    HostAllocationPlan::numeric_workspace_bytes(NumericDType::U64, 7).unwrap(),
                )
                .unwrap();
        assert_eq!(
            exact_host_workspace(
                "randperm",
                ExactHostRandomWorkspace::Permutation {
                    population: 11,
                    count: 7
                },
                ExactHostRandomMode::Managed
            )
            .unwrap(),
            ResourceByteBound::Known(managed_permutation)
        );
        assert_eq!(
            exact_host_workspace(
                "randi",
                ExactHostRandomWorkspace::IntegerOffsets { count: 7 },
                ExactHostRandomMode::Unbounded
            )
            .unwrap(),
            ResourceByteBound::Known(7 * std::mem::size_of::<u64>() as u64)
        );
        assert_eq!(
            exact_host_workspace(
                "randperm",
                ExactHostRandomWorkspace::Permutation {
                    population: 11,
                    count: 7
                },
                ExactHostRandomMode::Unbounded
            )
            .unwrap(),
            ResourceByteBound::Known(
                11 * std::mem::size_of::<f64>() as u64 + 7 * std::mem::size_of::<u64>() as u64
            )
        );
    }
}
