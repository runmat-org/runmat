//! Session global storage with operation-bound collection growth.
//!
//! A context fork shares one immutable generation. Its first write detaches
//! only name/value owner handles; ordinary unique writes mutate the admitted
//! hash tables in place. No RuntimeContext is retained by the generation.

use hashbrown::HashMap;
use runmat_execution::resource::ExactShared;
use runmat_value::{ExactHostAllocationPermit, ExactHostTensorAllocation, NumericDType, Value};
use std::collections::hash_map::RandomState;

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt, HostOutputPermit,
    OperationCollectionAllocator, OperationPermitGate,
};
use crate::context::RuntimeContext;
use crate::workspace::retained::{WorkspaceName, WorkspaceRetainedValue, WorkspaceValueLease};

#[path = "globals/owners.rs"]
mod owners;
pub use owners::SessionGlobalGeneration;
pub(super) use owners::SessionGlobals;
use owners::{GlobalGeneration, NamedMap, SlotMap};

#[derive(Clone, Copy)]
enum GlobalTarget<'a> {
    Named(&'a str),
    Slot(usize),
}

enum GlobalPayloadSource<'a> {
    Owned(Value),
    Borrowed(&'a Value),
    CanonicalEmpty,
}

struct GlobalValueSource<'a> {
    payload: GlobalPayloadSource<'a>,
    roots: crate::execution::RootedValueHandles,
}

impl GlobalValueSource<'_> {
    fn owned(value: Value, roots: crate::execution::RootedValueHandles) -> Self {
        Self {
            payload: GlobalPayloadSource::Owned(value),
            roots,
        }
    }

    fn canonical_empty() -> Self {
        Self {
            payload: GlobalPayloadSource::CanonicalEmpty,
            roots: crate::execution::RootedValueHandles::empty(),
        }
    }

    fn borrowed(value: &Value) -> GlobalValueSource<'_> {
        GlobalValueSource {
            payload: GlobalPayloadSource::Borrowed(value),
            roots: crate::execution::RootedValueHandles::empty(),
        }
    }
}

enum PreparedTarget {
    Named(WorkspaceName),
    Slot(usize),
}

/// Every allocation and capacity check for one session write has completed.
/// The caller's reservation prevents another global mutation before commit.
pub(super) struct PreparedGlobalWrite {
    kind: PreparedGlobalWriteKind,
    // Keep the selected allocation authority through publication, after the
    // prepared value/table owners in `kind` retire on abort.
    receipt: Option<HostAllocationReceipt>,
}

enum PreparedGlobalWriteKind {
    /// A declaration of an already-established global only binds the caller's
    /// workspace. Its session value must not be replaced with canonical empty.
    Existing { value: WorkspaceValueLease },
    Mutation {
        target: PreparedTarget,
        value: WorkspaceValueLease,
        detached: Option<ExactShared<GlobalGeneration>>,
        growth: Option<PreparedGrowth>,
    },
}

/// Owners displaced by an allocation-free session commit. These must retire
/// after the session table's mutable borrow is released: a Value may own a
/// foreign release callback which reenters the session/workspace.
#[derive(Default)]
pub(super) struct SessionGlobalRetirement {
    _value: Option<WorkspaceValueLease>,
    _existing: Option<WorkspaceValueLease>,
    _name: Option<WorkspaceName>,
    _named: Option<NamedMap>,
    _slots: Option<SlotMap>,
    _generation: Option<ExactShared<GlobalGeneration>>,
    _receipt: Option<HostAllocationReceipt>,
}

impl PreparedGlobalWrite {
    pub(super) fn value(&self) -> &Value {
        match &self.kind {
            PreparedGlobalWriteKind::Existing { value }
            | PreparedGlobalWriteKind::Mutation { value, .. } => value.value(),
        }
    }

    pub(super) fn checkpoint(&self, context: &RuntimeContext) -> Result<(), crate::RuntimeError> {
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(context)
                .map_err(|cause| session_allocation_error(context, cause))?;
        }
        Ok(())
    }

    pub(super) fn changes_session_value(&self) -> bool {
        matches!(&self.kind, PreparedGlobalWriteKind::Mutation { .. })
    }
}

#[path = "globals/planning.rs"]
mod planning;
use planning::PreparedGrowth;

impl SessionGlobals {
    pub(super) fn store_named_unbounded(
        &mut self,
        name: &str,
        value: Value,
    ) -> Result<(), crate::RuntimeError> {
        self.store_unbounded(GlobalTarget::Named(name), value)
    }

    pub(super) fn store_slot_unbounded(
        &mut self,
        slot: usize,
        value: Value,
    ) -> Result<(), crate::RuntimeError> {
        self.store_unbounded(GlobalTarget::Slot(slot), value)
    }

    fn store_unbounded(
        &mut self,
        target: GlobalTarget<'_>,
        value: Value,
    ) -> Result<(), crate::RuntimeError> {
        crate::execution::validate_storable_value(&value)?;
        let plan = self
            .plan(target, false)
            .expect("session global geometry overflow");
        let gate = self
            .generation
            .as_ref()
            .map(|generation| generation.get().gate.clone())
            .unwrap_or_else(OperationPermitGate::unbounded);
        let _installed = gate
            .install_unbounded()
            .expect("standalone session mutation is reentrant");
        let prepared_name = match target {
            GlobalTarget::Named(name) if plan.new_name => Some(WorkspaceName::unbounded(name)),
            _ => None,
        };
        let roots = crate::execution::RootedValueHandles::for_traced(&value)
            .map_err(|cause| session_root_error(None, cause))?;
        let prepared_value = WorkspaceValueLease::unbounded_rooted(value, roots);
        if plan.replace_generation {
            let mut named = unbounded_named_map(&gate, plan.named_capacity);
            let mut slots = unbounded_slot_map(&gate, plan.slot_capacity);
            if let Some(existing) = &self.generation {
                for (name, value) in &existing.get().named {
                    named.insert(name.clone(), value.clone());
                }
                for (slot, value) in &existing.get().slots {
                    slots.insert(*slot, value.clone());
                }
            }
            self.generation = Some(ExactShared::new(GlobalGeneration {
                named,
                slots,
                gate: gate.clone(),
                _claim: None,
            }));
        } else {
            let generation = self.generation.as_mut().expect("unique generation exists");
            let current = generation.get_mut_unique();
            if plan.named_layout.is_some() {
                let mut replacement = unbounded_named_map(&gate, plan.named_capacity);
                for (name, value) in current.named.drain() {
                    replacement.insert(name, value);
                }
                current.named = replacement;
            }
            if plan.slot_layout.is_some() {
                let mut replacement = unbounded_slot_map(&gate, plan.slot_capacity);
                for (slot, value) in current.slots.drain() {
                    replacement.insert(slot, value);
                }
                current.slots = replacement;
            }
        }
        let generation = self
            .generation
            .as_mut()
            .expect("global generation exists")
            .get_mut_unique();
        match target {
            GlobalTarget::Named(name) => {
                let key = prepared_name.unwrap_or_else(|| {
                    generation
                        .named
                        .get_key_value(name)
                        .expect("existing name")
                        .0
                        .clone()
                });
                generation.named.insert(key, prepared_value);
            }
            GlobalTarget::Slot(slot) => {
                generation.slots.insert(slot, prepared_value);
            }
        }
        Ok(())
    }

    pub(super) fn named(&self, name: &str) -> Option<&Value> {
        self.generation
            .as_ref()?
            .get()
            .named
            .get(name)
            .map(WorkspaceValueLease::value)
    }

    pub(super) fn slot(&self, slot: usize) -> Option<&Value> {
        self.generation
            .as_ref()?
            .get()
            .slots
            .get(&slot)
            .map(WorkspaceValueLease::value)
    }

    pub(super) fn names(&self) -> Vec<String> {
        let mut names = self
            .generation
            .as_ref()
            .map(|generation| {
                generation
                    .get()
                    .named
                    .keys()
                    .map(|name| name.as_str().to_owned())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        names.sort();
        names
    }

    pub(super) fn values(&self) -> impl Iterator<Item = &Value> {
        self.generation
            .as_ref()
            .into_iter()
            .flat_map(|generation| {
                generation
                    .get()
                    .named
                    .values()
                    .chain(generation.get().slots.values())
            })
            .map(WorkspaceValueLease::value)
    }

    pub(super) fn prepare_named(
        &self,
        context: &RuntimeContext,
        name: &str,
        value: Value,
        roots: crate::execution::RootedValueHandles,
    ) -> Result<PreparedGlobalWrite, crate::RuntimeError> {
        self.prepare(
            context,
            GlobalTarget::Named(name),
            GlobalValueSource::owned(value, roots),
        )
    }

    pub(super) fn prepare_slot(
        &self,
        context: &RuntimeContext,
        slot: usize,
        value: Value,
        roots: crate::execution::RootedValueHandles,
    ) -> Result<PreparedGlobalWrite, crate::RuntimeError> {
        self.prepare(
            context,
            GlobalTarget::Slot(slot),
            GlobalValueSource::owned(value, roots),
        )
    }

    pub(super) fn prepare_named_ref(
        &self,
        context: &RuntimeContext,
        name: &str,
        value: &Value,
    ) -> Result<PreparedGlobalWrite, crate::RuntimeError> {
        self.prepare(
            context,
            GlobalTarget::Named(name),
            GlobalValueSource::borrowed(value),
        )
    }

    pub(super) fn prepare_slot_ref(
        &self,
        context: &RuntimeContext,
        slot: usize,
        value: &Value,
    ) -> Result<PreparedGlobalWrite, crate::RuntimeError> {
        self.prepare(
            context,
            GlobalTarget::Slot(slot),
            GlobalValueSource::borrowed(value),
        )
    }

    pub(super) fn prepare_declaration(
        &self,
        context: &RuntimeContext,
        name: &str,
    ) -> Result<PreparedGlobalWrite, crate::RuntimeError> {
        if let Some(value) = self
            .generation
            .as_ref()
            .and_then(|generation| generation.get().named.get(name))
        {
            return Ok(PreparedGlobalWrite {
                kind: PreparedGlobalWriteKind::Existing {
                    value: value.clone(),
                },
                receipt: None,
            });
        }
        self.prepare(
            context,
            GlobalTarget::Named(name),
            GlobalValueSource::canonical_empty(),
        )
    }

    pub(super) fn prepare_slot_declaration(
        &self,
        context: &RuntimeContext,
        slot: usize,
    ) -> Result<PreparedGlobalWrite, crate::RuntimeError> {
        if let Some(value) = self
            .generation
            .as_ref()
            .and_then(|generation| generation.get().slots.get(&slot))
        {
            return Ok(PreparedGlobalWrite {
                kind: PreparedGlobalWriteKind::Existing {
                    value: value.clone(),
                },
                receipt: None,
            });
        }
        self.prepare(
            context,
            GlobalTarget::Slot(slot),
            GlobalValueSource::canonical_empty(),
        )
    }

    fn prepare(
        &self,
        context: &RuntimeContext,
        target: GlobalTarget<'_>,
        source: GlobalValueSource<'_>,
    ) -> Result<PreparedGlobalWrite, crate::RuntimeError> {
        let GlobalValueSource { payload, roots } = source;
        match &payload {
            GlobalPayloadSource::Owned(value) => {
                crate::execution::validate_storable_value(value)?;
            }
            GlobalPayloadSource::Borrowed(value) => {
                crate::execution::validate_storable_value(value)?;
            }
            GlobalPayloadSource::CanonicalEmpty => {}
        }
        let empty = matches!(&payload, GlobalPayloadSource::CanonicalEmpty);
        let plan = self
            .plan(target, empty)
            .map_err(|cause| session_allocation_error(context, cause))?;
        let receipt = HostAllocationReceipt::admit(
            Some(context),
            HostAllocationPlan::new(0, plan.retained_bytes)
                .map_err(|cause| session_allocation_error(context, cause))?,
        )
        .map_err(|cause| session_allocation_error(context, cause))?;
        let permit = receipt
            .as_ref()
            .map(|receipt| receipt.output_permit(context))
            .transpose()
            .map_err(|cause| session_allocation_error(context, cause))?;
        let prepared_value = match payload {
            GlobalPayloadSource::Owned(value) => match &permit {
                Some(permit) => WorkspaceValueLease::from_retained(
                    WorkspaceRetainedValue::prepare_owned(value, roots, permit)
                        .map_err(|cause| session_allocation_error(context, cause.into()))?,
                ),
                None => WorkspaceValueLease::unbounded_rooted(value, roots),
            },
            GlobalPayloadSource::Borrowed(value) => match &permit {
                Some(permit) => WorkspaceValueLease::from_retained(
                    WorkspaceRetainedValue::prepare(value, permit)
                        .map_err(|cause| session_allocation_error(context, cause.into()))?
                        .ok_or_else(|| {
                            session_allocation_error(
                                context,
                                HostAllocationError::Resource(
                                    crate::context::host_effect::HostEffectAllocationError::Unavailable,
                                ),
                            )
                        })?,
                ),
                None => WorkspaceValueLease::unbounded(value.clone()),
            },
            GlobalPayloadSource::CanonicalEmpty => match &permit {
                Some(permit) => WorkspaceValueLease::from_retained(
                    WorkspaceRetainedValue::prepare_owned(
                        Value::Tensor(
                    ExactHostTensorAllocation::new(NumericDType::F64, &[0, 0], permit)
                        .and_then(ExactHostTensorAllocation::into_tensor)
                        .map_err(|cause| session_allocation_error(context, cause.into()))?,
                        ),
                        roots,
                        permit,
                    )
                    .map_err(|cause| session_allocation_error(context, cause.into()))?,
                ),
                None => WorkspaceValueLease::unbounded_rooted(
                    super::canonical_empty_workspace_value(),
                    roots,
                ),
            },
        };
        let prepared_name = match target {
            GlobalTarget::Named(name) if plan.new_name => Some(match &permit {
                Some(permit) => WorkspaceName::prepare(name, permit)
                    .map_err(|cause| session_allocation_error(context, cause.into()))?,
                None => WorkspaceName::unbounded(name),
            }),
            _ => None,
        };
        let gate = self
            .generation
            .as_ref()
            .map(|generation| generation.get().gate.clone())
            .map(Ok)
            .unwrap_or_else(|| match &permit {
                Some(permit) => {
                    OperationPermitGate::prepare(permit).map_err(HostAllocationError::from)
                }
                None => Ok(OperationPermitGate::unbounded()),
            })
            .map_err(|cause| session_allocation_error(context, cause))?;
        let _installed = match &permit {
            Some(permit) => gate.install(permit.clone()),
            None => gate.install_unbounded(),
        }
        .map_err(|_| session_gate_error(context))?;

        // All fallible table allocations happen before the current binding is
        // changed. A detached generation contains only shared owner handles.
        let detached = if plan.replace_generation {
            Some(self.prepare_generation(plan, &gate, permit.as_ref(), context)?)
        } else {
            None
        };
        let growth = if detached.is_none() {
            self.prepare_unique_growth(plan, &gate, context)?
        } else {
            None
        };
        if let Some(receipt) = &receipt {
            receipt
                .checkpoint(context)
                .map_err(|cause| session_allocation_error(context, cause))?;
        }
        let key = match target {
            GlobalTarget::Named(name) => Some(prepared_name.unwrap_or_else(|| {
                self.generation
                    .as_ref()
                    .and_then(|generation| generation.get().named.get_key_value(name))
                    .expect("existing global name was authenticated by the plan")
                    .0
                    .clone()
            })),
            GlobalTarget::Slot(_) => None,
        };
        Ok(PreparedGlobalWrite {
            kind: PreparedGlobalWriteKind::Mutation {
                target: match target {
                    GlobalTarget::Named(_) => {
                        PreparedTarget::Named(key.expect("named target has a key"))
                    }
                    GlobalTarget::Slot(slot) => PreparedTarget::Slot(slot),
                },
                value: prepared_value,
                detached,
                growth,
            },
            receipt,
        })
    }

    /// Insertion cannot allocate: the prepared table has the selected spare
    /// slot, and a reservation excluded intervening session-global writes.
    pub(super) fn commit(&mut self, prepared: PreparedGlobalWrite) -> SessionGlobalRetirement {
        let mut retired = SessionGlobalRetirement::default();
        let PreparedGlobalWrite { kind, receipt } = prepared;
        retired._receipt = receipt;
        let (target, value, detached, growth) = match kind {
            PreparedGlobalWriteKind::Mutation {
                target,
                value,
                detached,
                growth,
            } => (target, value, detached, growth),
            PreparedGlobalWriteKind::Existing { value } => {
                retired._existing = Some(value);
                return retired;
            }
        };
        if let Some(detached) = detached {
            retired._generation = self.generation.replace(detached);
        }
        let generation = self
            .generation
            .as_mut()
            .expect("prepared generation exists")
            .get_mut_unique();
        if let Some(growth) = growth {
            match growth {
                PreparedGrowth::Named(mut replacement) => {
                    for (name, value) in generation.named.drain() {
                        replacement.insert(name, value);
                    }
                    retired._named = Some(std::mem::replace(&mut generation.named, replacement));
                }
                PreparedGrowth::Slot(mut replacement) => {
                    for (slot, value) in generation.slots.drain() {
                        replacement.insert(slot, value);
                    }
                    retired._slots = Some(std::mem::replace(&mut generation.slots, replacement));
                }
            }
        }
        match target {
            PreparedTarget::Named(key) => {
                if let Some(existing) = generation.named.get_mut(key.as_str()) {
                    retired._value = Some(std::mem::replace(existing, value));
                    retired._name = Some(key);
                } else {
                    generation.named.insert(key, value);
                }
            }
            PreparedTarget::Slot(slot) => {
                if let Some(existing) = generation.slots.get_mut(&slot) {
                    retired._value = Some(std::mem::replace(existing, value));
                } else {
                    generation.slots.insert(slot, value);
                }
            }
        }
        retired
    }
}

fn unbounded_named_map(gate: &OperationPermitGate, capacity: usize) -> NamedMap {
    let mut map = HashMap::with_hasher_in(
        RandomState::new(),
        OperationCollectionAllocator::new(gate.clone()),
    );
    map.try_reserve(capacity)
        .expect("standalone session global table allocation failed");
    map
}

fn unbounded_slot_map(gate: &OperationPermitGate, capacity: usize) -> SlotMap {
    let mut map = HashMap::with_hasher_in(
        RandomState::new(),
        OperationCollectionAllocator::new(gate.clone()),
    );
    map.try_reserve(capacity)
        .expect("standalone session global table allocation failed");
    map
}

fn retained_owner_bytes(raw: usize) -> Result<u64, HostAllocationError> {
    HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(raw).map_err(|_| HostAllocationError::Overflow)?,
        1,
    )
}

fn checked_add(left: u64, right: u64) -> Result<u64, HostAllocationError> {
    left.checked_add(right).ok_or(HostAllocationError::Overflow)
}

fn session_capacity_error(context: &RuntimeContext) -> crate::RuntimeError {
    session_allocation_error(
        context,
        HostAllocationError::Resource(
            crate::context::host_effect::HostEffectAllocationError::ExceedsCapacity,
        ),
    )
}

fn session_gate_error(context: &RuntimeContext) -> crate::RuntimeError {
    crate::runtime_error::semantic_error(
        &format!("{}:WorkspaceSessionMutation", context.error_namespace()),
        "session-global mutation is already active",
    )
}

fn session_root_error(
    context: Option<&RuntimeContext>,
    cause: runmat_gc::GcError,
) -> crate::RuntimeError {
    let namespace = context.map_or_else(|| "RunMat".to_owned(), RuntimeContext::error_namespace);
    crate::build_runtime_error("session-global value rooting failed")
        .with_identifier(format!("{namespace}:Workspace:Rooting"))
        .with_source(cause)
        .build()
}

pub(super) fn session_allocation_error(
    context: &RuntimeContext,
    cause: HostAllocationError,
) -> crate::RuntimeError {
    crate::context::host_allocation::host_allocation_runtime_error(
        Some(context),
        &format!(
            "{}:{}",
            context.error_namespace(),
            cause.runtime_identifier_suffix()
        ),
        format!("session-global publication admission failed: {cause}"),
        cause,
    )
}
