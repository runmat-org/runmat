use super::*;
use crate::context::host_allocation::{hashbrown_table_layout, AdmittedCollectionAllocator};
use std::alloc::Layout;

#[derive(Clone, Copy)]
pub(super) struct GenerationWritePlan {
    pub(super) named_layout: Option<Layout>,
    pub(super) slot_layout: Option<Layout>,
    pub(super) named_capacity: usize,
    pub(super) slot_capacity: usize,
    pub(super) replace_generation: bool,
    pub(super) new_name: bool,
    pub(super) retained_bytes: u64,
}

impl SessionGlobals {
    pub(super) fn plan(
        &self,
        target: GlobalTarget<'_>,
        canonical_empty: bool,
    ) -> Result<GenerationWritePlan, HostAllocationError> {
        let current = self.generation.as_ref().map(ExactShared::get);
        let (new_name, named_len, slot_len) = match target {
            GlobalTarget::Named(name) => (
                current.is_none_or(|generation| !generation.named.contains_key(name)),
                current.map_or(0, |generation| generation.named.len()),
                current.map_or(0, |generation| generation.slots.len()),
            ),
            GlobalTarget::Slot(_) => (
                false,
                current.map_or(0, |generation| generation.named.len()),
                current.map_or(0, |generation| generation.slots.len()),
            ),
        };
        let replace_generation = self
            .generation
            .as_ref()
            .is_none_or(|owner| !owner.is_unique());
        let named_target = named_len
            .checked_add(usize::from(
                matches!(target, GlobalTarget::Named(_)) && new_name,
            ))
            .ok_or(HostAllocationError::Overflow)?;
        let slot_target = slot_len
            .checked_add(usize::from(
                matches!(target, GlobalTarget::Slot(slot) if current.is_none_or(|generation| !generation.slots.contains_key(&slot))),
            ))
            .ok_or(HostAllocationError::Overflow)?;
        let named_layout = if replace_generation
            || current.is_some_and(|generation| named_target > generation.named.capacity())
        {
            hashbrown_table_layout::<WorkspaceName, WorkspaceValueLease>(named_target)?
        } else {
            None
        };
        let slot_layout = if replace_generation
            || current.is_some_and(|generation| slot_target > generation.slots.capacity())
        {
            hashbrown_table_layout::<usize, WorkspaceValueLease>(slot_target)?
        } else {
            None
        };
        let mut retained_bytes = retained_owner_bytes(WorkspaceRetainedValue::requested_bytes())?;
        if canonical_empty {
            let raw = ExactHostTensorAllocation::requested_bytes(NumericDType::F64, &[0, 0])?;
            retained_bytes = checked_add(
                retained_bytes,
                HostAllocationPlan::retained_exact_output_bytes(
                    u64::try_from(raw).map_err(|_| HostAllocationError::Overflow)?,
                    ExactHostTensorAllocation::retained_claim_count(),
                )?,
            )?;
        }
        if new_name {
            if let GlobalTarget::Named(name) = target {
                retained_bytes = checked_add(
                    retained_bytes,
                    retained_owner_bytes(WorkspaceName::requested_bytes(name)?)?,
                )?;
            }
        }
        if replace_generation {
            retained_bytes = checked_add(
                retained_bytes,
                retained_owner_bytes(ExactShared::<GlobalGeneration>::requested_bytes())?,
            )?;
        }
        if current.is_none() {
            retained_bytes = checked_add(retained_bytes, OperationPermitGate::retained_bytes()?)?;
        }
        for layout in [named_layout, slot_layout].into_iter().flatten() {
            let raw = AdmittedCollectionAllocator::requested_bytes(layout)
                .map_err(|_| HostAllocationError::Overflow)?;
            retained_bytes = checked_add(retained_bytes, retained_owner_bytes(raw)?)?;
        }
        Ok(GenerationWritePlan {
            named_layout,
            slot_layout,
            named_capacity: named_target,
            slot_capacity: slot_target,
            replace_generation,
            new_name,
            retained_bytes,
        })
    }

    pub(super) fn prepare_generation(
        &self,
        plan: GenerationWritePlan,
        gate: &OperationPermitGate,
        permit: Option<&HostOutputPermit>,
        context: &RuntimeContext,
    ) -> Result<ExactShared<GlobalGeneration>, crate::RuntimeError> {
        let mut named = new_named_map(gate, plan.named_capacity, plan.named_layout, context)?;
        let mut slots = new_slot_map(gate, plan.slot_capacity, plan.slot_layout, context)?;
        if let Some(existing) = &self.generation {
            for (key, value) in &existing.get().named {
                named.insert(key.clone(), value.clone());
            }
            for (key, value) in &existing.get().slots {
                slots.insert(*key, value.clone());
            }
        }
        let claim = permit
            .map(|permit| {
                permit.reserve_exact_claim(ExactShared::<GlobalGeneration>::requested_bytes())
            })
            .transpose()
            .map_err(|cause| session_allocation_error(context, cause.into()))?;
        ExactShared::try_new(GlobalGeneration {
            named,
            slots,
            gate: gate.clone(),
            _claim: claim,
        })
        .map_err(|_| {
            session_allocation_error(
                context,
                HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                ),
            )
        })
    }

    pub(super) fn prepare_unique_growth(
        &self,
        plan: GenerationWritePlan,
        gate: &OperationPermitGate,
        context: &RuntimeContext,
    ) -> Result<Option<PreparedGrowth>, crate::RuntimeError> {
        if self.generation.is_none() {
            return Ok(None);
        }
        if plan.named_layout.is_some() {
            return new_named_map(gate, plan.named_capacity, plan.named_layout, context)
                .map(|map| Some(PreparedGrowth::Named(map)));
        }
        if plan.slot_layout.is_some() {
            return new_slot_map(gate, plan.slot_capacity, plan.slot_layout, context)
                .map(|map| Some(PreparedGrowth::Slot(map)));
        }
        Ok(None)
    }
}

pub(super) enum PreparedGrowth {
    Named(NamedMap),
    Slot(SlotMap),
}

pub(super) fn new_named_map(
    gate: &OperationPermitGate,
    capacity: usize,
    layout: Option<Layout>,
    context: &RuntimeContext,
) -> Result<NamedMap, crate::RuntimeError> {
    let mut map = HashMap::with_hasher_in(
        RandomState::new(),
        OperationCollectionAllocator::new(gate.clone()),
    );
    map.try_reserve(capacity)
        .map_err(|_| session_capacity_error(context))?;
    if map.allocation_size() != layout.map_or(0, |layout| layout.size()) {
        return Err(session_capacity_error(context));
    }
    Ok(map)
}

pub(super) fn new_slot_map(
    gate: &OperationPermitGate,
    capacity: usize,
    layout: Option<Layout>,
    context: &RuntimeContext,
) -> Result<SlotMap, crate::RuntimeError> {
    let mut map = HashMap::with_hasher_in(
        RandomState::new(),
        OperationCollectionAllocator::new(gate.clone()),
    );
    map.try_reserve(capacity)
        .map_err(|_| session_capacity_error(context))?;
    if map.allocation_size() != layout.map_or(0, |layout| layout.size()) {
        return Err(session_capacity_error(context));
    }
    Ok(map)
}
