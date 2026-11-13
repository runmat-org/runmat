//! Shared, admitted source-name metadata for one Native workspace frame.
//!
//! Slot and name indexes share each immutable name owner. Neither index owns
//! language values, and service handles do not copy the frame's name table.

use allocator_api2::vec::Vec as AllocatedVec;
use hashbrown::HashMap;
use runmat_execution::resource::ExactShared;
use runmat_native_codegen::{NativeFunction, NativeLocalKind};
use runmat_runtime::context::host_allocation::{
    collection_block_retained_bytes, frame_hash_table_retained_bytes, FrameCollectionGate,
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt, OperationCollectionAllocator,
};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::workspace::WorkspaceName;
use runmat_value::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostNumericAllocationError,
};
use std::alloc::Layout;
use std::collections::hash_map::RandomState;

use super::NativeWorkspaceInput;
use crate::{NativeExecutorError, NativeExecutorResult};

type SlotNames = AllocatedVec<(usize, WorkspaceName), OperationCollectionAllocator>;
type NameSlots = HashMap<WorkspaceName, usize, RandomState, OperationCollectionAllocator>;

#[derive(Debug)]
struct Names {
    slots: SlotNames,
    by_name: NameSlots,
    // All physical collection blocks and their name owners retire first.
    _claim: Option<ExactHostAllocationClaim>,
}

#[derive(Clone, Debug)]
pub(super) struct NativeWorkspaceNames(Option<ExactShared<Names>>);

impl NativeWorkspaceNames {
    pub(super) fn prepare(
        function: &NativeFunction,
        input: Option<&NativeWorkspaceInput>,
        context: Option<&RuntimeContext>,
    ) -> NativeExecutorResult<Self> {
        let count = source_names(function, input).count();
        if count == 0 {
            return Ok(Self(None));
        }
        let header = ExactShared::<Names>::requested_bytes();
        let mut retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(header).map_err(|_| error(context, HostAllocationError::Overflow))?,
            1,
        )
        .map_err(|cause| error(context, cause))?;
        let mut add = |bytes: u64| -> NativeExecutorResult<()> {
            retained = retained
                .checked_add(bytes)
                .ok_or_else(|| error(context, HostAllocationError::Overflow))?;
            Ok(())
        };
        add(FrameCollectionGate::retained_bytes().map_err(|cause| error(context, cause))?)?;
        let layout = Layout::array::<(usize, WorkspaceName)>(count)
            .map_err(|_| error(context, HostAllocationError::Overflow))?;
        add(collection_block_retained_bytes(layout).map_err(|cause| error(context, cause))?)?;
        add(
            frame_hash_table_retained_bytes::<WorkspaceName, usize>(count)
                .map_err(|cause| error(context, cause))?,
        )?;
        for (_, name) in source_names(function, input) {
            let physical = WorkspaceName::requested_bytes(name)
                .map_err(|cause| error(context, HostAllocationError::Numeric(cause)))?;
            add(HostAllocationPlan::retained_exact_output_bytes(
                u64::try_from(physical)
                    .map_err(|_| error(context, HostAllocationError::Overflow))?,
                1,
            )
            .map_err(|cause| error(context, cause))?)?;
        }
        let plan = HostAllocationPlan::new(0, retained).map_err(|cause| error(context, cause))?;
        let receipt =
            HostAllocationReceipt::admit(context, plan).map_err(|cause| error(context, cause))?;
        let gate = match (context, &receipt) {
            (Some(context), Some(receipt)) => FrameCollectionGate::prepare(context, receipt),
            _ => Ok(FrameCollectionGate::unbounded()),
        }
        .map_err(|cause| error(context, cause))?;
        let _installed = match (context, &receipt) {
            (Some(context), Some(receipt)) => gate.install(context, receipt),
            _ => gate.install_unbounded(),
        }
        .map_err(|cause| error(context, cause))?;
        let build = |permit: Option<&dyn ExactHostAllocationPermit>| {
            let claim = permit
                .map(|permit| permit.reserve_exact_claim(header))
                .transpose()?;
            let allocator = gate.allocator();
            let mut slots = SlotNames::new_in(allocator.clone());
            let mut by_name = NameSlots::with_hasher_in(RandomState::new(), allocator);
            slots
                .try_reserve_exact(count)
                .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?;
            by_name
                .try_reserve(count)
                .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)?;
            for (slot, text) in source_names(function, input) {
                let name = match permit {
                    Some(permit) => WorkspaceName::prepare(text, permit)?,
                    None => WorkspaceName::unbounded(text),
                };
                slots.push((slot, name.clone()));
                by_name.insert(name, slot);
            }
            slots.sort_unstable_by_key(|(slot, _)| *slot);
            ExactShared::try_new(Names {
                slots,
                by_name,
                _claim: claim,
            })
            .map(|names| Self(Some(names)))
            .map_err(|_| ExactHostNumericAllocationError::AllocationFailed)
        };
        let names = match (context, &receipt) {
            (Some(context), Some(receipt)) => receipt
                .with_output_allocation(context, |permit| build(Some(permit)))
                .map_err(|cause| error(Some(context), cause))?,
            _ => {
                build(None).map_err(|cause| error(context, HostAllocationError::Numeric(cause)))?
            }
        };
        let payload = names.0.as_ref().expect("nonempty source-name layout").get();
        if payload.slots.len() != payload.by_name.len() {
            return Err(NativeExecutorError::Host(
                "native workspace layout contains duplicate source names".into(),
            ));
        }
        Ok(names)
    }

    pub(super) fn name(&self, slot: usize) -> Option<&str> {
        self.name_owner(slot).map(WorkspaceName::as_str)
    }

    pub(super) fn name_owner(&self, slot: usize) -> Option<&WorkspaceName> {
        let slots = &self.0.as_ref()?.get().slots;
        slots
            .binary_search_by_key(&slot, |(slot, _)| *slot)
            .ok()
            .map(|index| &slots[index].1)
    }

    pub(super) fn slot(&self, name: &str) -> Option<usize> {
        self.0.as_ref()?.get().by_name.get(name).copied()
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (usize, &str)> {
        self.0
            .iter()
            .flat_map(|names| names.get().slots.iter())
            .map(|(slot, name)| (*slot, name.as_str()))
    }
}

fn source_names<'a>(
    function: &'a NativeFunction,
    input: Option<&'a NativeWorkspaceInput>,
) -> impl Iterator<Item = (usize, &'a str)> {
    function.locals.iter().filter_map(move |local| {
        let name = match input {
            Some(input) if local.kind == NativeLocalKind::Binding => local
                .binding
                .and_then(|binding| input.local_names.get(&binding).map(String::as_str))
                .or(local.name.as_deref()),
            None if local.kind != NativeLocalKind::Temporary => local.name.as_deref(),
            _ => None,
        }?;
        Some((local.id.0 as usize, name))
    })
}

fn error(context: Option<&RuntimeContext>, cause: HostAllocationError) -> NativeExecutorError {
    NativeExecutorError::from(crate::error::allocation_error(
        context,
        format!("native workspace name metadata admission failed: {cause}"),
        cause,
    ))
}
