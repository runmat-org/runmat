//! The private typed GC activation is the sole lexical binding authority.
//! Its immutable slot generations can be leased without copying a Value.

use runmat_gc::{
    gc_allocate_lexical_activation_rooted, gc_record_handle_write, gc_root,
    gc_with_lexical_activation, gc_with_lexical_activation_mut, ExplicitRoot, LexicalActivation,
    LexicalReservationOutcome, LexicalSlotState, LexicalValueOwner, PreparedLexicalTable,
};
use runmat_types::BindingId;
use runmat_value::{LexicalEnvironment, Value};

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::execution::{RootedValueAdmissionError, RootedValueLease};
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

/// A stable old binding generation. Its root lease is external to the GC
/// activation, so a closure-to-activation cycle remains collectible when no
/// read lease escapes.
#[derive(Clone, Debug)]
pub struct LexicalBindingReadLease {
    owner: LexicalValueOwner,
    _roots: RootedValueLease,
}

impl LexicalBindingReadLease {
    pub fn value(&self) -> &Value {
        self.owner.value()
    }

    pub(crate) fn shared_generation(&self) -> LexicalValueOwner {
        self.owner.clone()
    }

    pub(crate) fn shared_roots(&self) -> crate::execution::RootedValueHandles {
        self._roots.shared_handles()
    }
}

/// Allocate a distinct private activation. The returned root belongs to the
/// invoking frame; escaped closures trace the handle through their Value edge.
pub fn allocate_environment(
    parent: Option<LexicalEnvironment>,
) -> Result<(LexicalEnvironment, Vec<ExplicitRoot>), RuntimeError> {
    let mut roots = parent
        .as_ref()
        .map(root_environment)
        .transpose()?
        .unwrap_or_default();
    let root = gc_allocate_lexical_activation_rooted(LexicalActivation::empty())
        .map_err(environment_gc_error)?;
    let environment = LexicalEnvironment::new(root.handle(), parent);
    roots.push(root);
    Ok((environment, roots))
}

pub fn root_environment(
    environment: &LexicalEnvironment,
) -> Result<Vec<ExplicitRoot>, RuntimeError> {
    let mut roots = Vec::new();
    let mut error = None;
    environment.for_each_handle(|handle| {
        if error.is_none() {
            match gc_root(handle) {
                Ok(root) => roots.push(root),
                Err(source) => error = Some(environment_gc_error(source)),
            }
        }
    });
    error.map_or(Ok(roots), Err)
}

/// Reserve one owned BindingId in the sorted activation table. The legacy
/// context-free form is explicitly unbounded; finite execution uses the
/// current-operation form below.
pub fn reserve_owned_binding(
    environment: &LexicalEnvironment,
    binding: BindingId,
) -> Result<bool, RuntimeError> {
    reserve_owned_binding_in_operation(environment, binding, None)
}

pub fn reserve_owned_binding_with_context(
    environment: &LexicalEnvironment,
    binding: BindingId,
    runtime: &RuntimeContext,
) -> Result<bool, RuntimeError> {
    reserve_owned_binding_in_operation(environment, binding, Some(runtime))
}

fn reserve_owned_binding_in_operation(
    environment: &LexicalEnvironment,
    binding: BindingId,
    runtime: Option<&RuntimeContext>,
) -> Result<bool, RuntimeError> {
    let plan = gc_with_lexical_activation(&environment.handle(), |activation| {
        if activation.owns_binding(binding) {
            Ok(None)
        } else {
            activation.growth_plan().map(Some)
        }
    })
    .map_err(environment_gc_error)?
    .map_err(|error| environment_error(&error.to_string()))?;
    let Some(growth) = plan else {
        validate_operation(runtime)?;
        return Ok(false);
    };
    let (prepared, receipt) = if let Some((capacity, revision)) = growth {
        let raw = PreparedLexicalTable::requested_bytes(capacity)
            .map_err(|error| environment_error(&error.to_string()))?;
        let receipt = admit_exact(runtime, raw, 1)?;
        let permit = receipt
            .as_ref()
            .map(|(receipt, runtime)| receipt.output_permit(runtime))
            .transpose()
            .map_err(environment_host_error)?;
        let prepared = PreparedLexicalTable::prepare(
            capacity,
            revision,
            permit
                .as_ref()
                .map(|permit| permit as &dyn runmat_value::ExactHostAllocationPermit),
        )
        .map_err(|error| environment_error(&error.to_string()))?;
        (Some(prepared), receipt)
    } else {
        validate_operation(runtime)?;
        (None, None)
    };
    checkpoint(&receipt)?;
    let outcome = gc_with_lexical_activation_mut(&environment.handle(), |activation| {
        activation.reserve_owned(binding, prepared)
    })
    .map_err(environment_gc_error)?;
    match outcome {
        LexicalReservationOutcome::Existing(unused) => {
            drop(unused);
            Ok(false)
        }
        LexicalReservationOutcome::Applied(retired) => {
            drop(retired);
            Ok(true)
        }
        LexicalReservationOutcome::Stale(unused) => {
            drop(unused);
            Err(environment_error(
                "lexical activation changed during reservation",
            ))
        }
    }
}

pub fn read_binding_lease(
    environment: &LexicalEnvironment,
    binding: BindingId,
) -> Result<Option<LexicalBindingReadLease>, RuntimeError> {
    read_binding_lease_in_operation(environment, binding, None)
}

pub fn read_binding_lease_with_context(
    environment: &LexicalEnvironment,
    binding: BindingId,
    runtime: &RuntimeContext,
) -> Result<Option<LexicalBindingReadLease>, RuntimeError> {
    read_binding_lease_in_operation(environment, binding, Some(runtime))
}

fn read_binding_lease_in_operation(
    environment: &LexicalEnvironment,
    binding: BindingId,
    runtime: Option<&RuntimeContext>,
) -> Result<Option<LexicalBindingReadLease>, RuntimeError> {
    let mut current = Some(environment.clone());
    while let Some(environment) = current {
        let observed = gc_with_lexical_activation(
            &environment.handle(),
            |activation| -> Result<_, RuntimeError> {
                match activation.slot_state(binding) {
                    Some(LexicalSlotState::Assigned) => {
                        let owner = activation.read(binding).ok_or_else(|| {
                            environment_error("assigned lexical binding has no value owner")
                        })?;
                        // Root while the GC borrow still protects the old slot,
                        // before a writer can replace it or collection can run.
                        let roots = match runtime {
                            Some(runtime) => {
                                RootedValueLease::for_value_in_context(owner.value(), runtime)
                                    .map_err(root_admission_error)?
                            }
                            None => RootedValueLease::for_traced(owner.value())
                                .map_err(|error| environment_error(&error.to_string()))?,
                        };
                        Ok(Some(Some(LexicalBindingReadLease {
                            owner,
                            _roots: roots,
                        })))
                    }
                    Some(LexicalSlotState::Uninitialized | LexicalSlotState::Vacant) => {
                        Ok(Some(None))
                    }
                    None => Ok(None),
                }
            },
        )
        .map_err(environment_gc_error)??;
        if let Some(lease) = observed {
            return Ok(lease);
        }
        current = environment.parent();
    }
    validate_operation(runtime)?;
    Ok(None)
}

/// Detached compatibility result. Consumers requiring finite all-class
/// admission must keep the lease and borrow its Value instead.
pub fn read_binding(
    environment: &LexicalEnvironment,
    binding: BindingId,
) -> Result<Option<Value>, RuntimeError> {
    read_binding_lease(environment, binding).map(|lease| lease.map(|lease| lease.value().clone()))
}

pub fn write_binding(
    environment: &LexicalEnvironment,
    binding: BindingId,
    value: Value,
) -> Result<(), RuntimeError> {
    write_binding_in_operation(environment, binding, value, None)
}

pub fn write_binding_with_context(
    environment: &LexicalEnvironment,
    binding: BindingId,
    value: Value,
    runtime: &RuntimeContext,
) -> Result<(), RuntimeError> {
    write_binding_in_operation(environment, binding, value, Some(runtime))
}

fn write_binding_in_operation(
    environment: &LexicalEnvironment,
    binding: BindingId,
    value: Value,
    runtime: Option<&RuntimeContext>,
) -> Result<(), RuntimeError> {
    let mut current = Some(environment.clone());
    while let Some(owner) = current {
        let owns = gc_with_lexical_activation(&owner.handle(), |activation| {
            activation.owns_binding(binding)
        })
        .map_err(environment_gc_error)?;
        if owns {
            return write_owned_binding_in_operation(&owner, binding, value, runtime);
        }
        current = owner.parent();
    }
    Err(environment_error(
        "inherited shared binding has no typed owner",
    ))
}

pub fn write_owned_binding(
    environment: &LexicalEnvironment,
    binding: BindingId,
    value: Value,
) -> Result<(), RuntimeError> {
    write_owned_binding_in_operation(environment, binding, value, None)
}

pub fn write_owned_binding_with_context(
    environment: &LexicalEnvironment,
    binding: BindingId,
    value: Value,
    runtime: &RuntimeContext,
) -> Result<(), RuntimeError> {
    write_owned_binding_in_operation(environment, binding, value, Some(runtime))
}

fn write_owned_binding_in_operation(
    environment: &LexicalEnvironment,
    binding: BindingId,
    value: Value,
    runtime: Option<&RuntimeContext>,
) -> Result<(), RuntimeError> {
    let _input_root = match runtime {
        Some(runtime) => {
            RootedValueLease::for_value_in_context(&value, runtime).map_err(root_admission_error)?
        }
        None => RootedValueLease::for_traced(&value)
            .map_err(|error| environment_error(&error.to_string()))?,
    };
    runmat_value::validate_no_transient_sequence(&value)
        .map_err(|error| environment_error(&error.to_string()))?;
    let receipt = admit_exact(runtime, LexicalValueOwner::requested_bytes(), 1)?;
    let permit = receipt
        .as_ref()
        .map(|(receipt, runtime)| receipt.output_permit(runtime))
        .transpose()
        .map_err(environment_host_error)?;
    let owner = match permit.as_ref() {
        Some(permit) => LexicalValueOwner::prepare(value, permit)
            .map_err(|error| environment_error(&error.to_string()))?,
        None => LexicalValueOwner::unbounded(value),
    };
    checkpoint(&receipt)?;
    gc_record_handle_write(&environment.handle(), owner.value());
    let outcome = gc_with_lexical_activation_mut(&environment.handle(), |activation| {
        activation.assign(binding, owner)
    })
    .map_err(environment_gc_error)?;
    match outcome {
        Ok(retired) => {
            drop(retired);
            Ok(())
        }
        Err(unused) => {
            drop(unused);
            Err(environment_error("owned shared binding was not reserved"))
        }
    }
}

pub fn remove_binding(
    environment: &LexicalEnvironment,
    binding: BindingId,
) -> Result<(), RuntimeError> {
    remove_binding_in_operation(environment, binding, None)
}

pub fn remove_binding_with_context(
    environment: &LexicalEnvironment,
    binding: BindingId,
    runtime: &RuntimeContext,
) -> Result<(), RuntimeError> {
    remove_binding_in_operation(environment, binding, Some(runtime))
}

fn remove_binding_in_operation(
    environment: &LexicalEnvironment,
    binding: BindingId,
    runtime: Option<&RuntimeContext>,
) -> Result<(), RuntimeError> {
    validate_operation(runtime)?;
    let mut current = Some(environment.clone());
    while let Some(owner) = current {
        let owns = gc_with_lexical_activation(&owner.handle(), |activation| {
            activation.owns_binding(binding)
        })
        .map_err(environment_gc_error)?;
        if owns {
            return remove_owned_binding_in_operation(&owner, binding, runtime);
        }
        current = owner.parent();
    }
    Ok(())
}

pub fn remove_owned_binding(
    environment: &LexicalEnvironment,
    binding: BindingId,
) -> Result<(), RuntimeError> {
    remove_owned_binding_in_operation(environment, binding, None)
}

pub fn remove_owned_binding_with_context(
    environment: &LexicalEnvironment,
    binding: BindingId,
    runtime: &RuntimeContext,
) -> Result<(), RuntimeError> {
    remove_owned_binding_in_operation(environment, binding, Some(runtime))
}

fn remove_owned_binding_in_operation(
    environment: &LexicalEnvironment,
    binding: BindingId,
    runtime: Option<&RuntimeContext>,
) -> Result<(), RuntimeError> {
    validate_operation(runtime)?;
    let retired = gc_with_lexical_activation_mut(&environment.handle(), |activation| {
        if activation.slot_state(binding) != Some(LexicalSlotState::Assigned) {
            return Ok(None);
        }
        activation.remove(binding)
    })
    .map_err(environment_gc_error)?
    .map_err(|source| {
        crate::runtime_error::semantic_error_with_source(
            "LexicalEnvironment",
            source.to_string(),
            source,
        )
    })?;
    drop(retired);
    Ok(())
}

/// Detached compatibility metadata for graph encoding. A finite snapshot
/// caller must separately admit its result collection.
pub fn owned_binding_ids(environment: &LexicalEnvironment) -> Result<Vec<BindingId>, RuntimeError> {
    gc_with_lexical_activation(&environment.handle(), |activation| {
        activation.owned_bindings().collect()
    })
    .map_err(environment_gc_error)
}

fn validate_operation(runtime: Option<&RuntimeContext>) -> Result<(), RuntimeError> {
    HostAllocationReceipt::policy(runtime)
        .map(|_| ())
        .map_err(environment_host_error)
}

fn admit_exact(
    runtime: Option<&RuntimeContext>,
    bytes: usize,
    claims: usize,
) -> Result<Option<(HostAllocationReceipt, &RuntimeContext)>, RuntimeError> {
    let Some(runtime) = runtime else {
        return Ok(None);
    };
    if HostAllocationReceipt::policy(Some(runtime)).map_err(environment_host_error)?
        == HostAllocationPolicy::Unbounded
    {
        return Ok(None);
    }
    let raw = u64::try_from(bytes).map_err(|_| environment_error("lexical owner size overflow"))?;
    let output = HostAllocationPlan::retained_exact_output_bytes(raw, claims)
        .map_err(environment_host_error)?;
    let plan = HostAllocationPlan::new(0, output).map_err(environment_host_error)?;
    let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(environment_host_error)?
        .ok_or_else(|| environment_error("managed lexical admission has no receipt"))?;
    Ok(Some((receipt, runtime)))
}

fn checkpoint(
    receipt: &Option<(HostAllocationReceipt, &RuntimeContext)>,
) -> Result<(), RuntimeError> {
    if let Some((receipt, runtime)) = receipt {
        receipt
            .checkpoint(runtime)
            .map_err(environment_host_error)?;
    }
    Ok(())
}

pub(super) fn root_admission_error(error: RootedValueAdmissionError) -> RuntimeError {
    match error {
        RootedValueAdmissionError::Host(error) => environment_host_error(error),
        RootedValueAdmissionError::Gc(error) => environment_gc_error(error),
    }
}

fn environment_error(message: &str) -> RuntimeError {
    semantic_error("LexicalEnvironment", message)
}

pub(super) fn environment_gc_error(source: runmat_gc::GcError) -> RuntimeError {
    let mut error = environment_error(&source.to_string());
    error.source = Some(Box::new(source));
    error
}

pub(super) fn environment_host_error(source: HostAllocationError) -> RuntimeError {
    let mut error = environment_error(&source.to_string());
    error.source = Some(Box::new(source));
    error
}

#[cfg(test)]
#[path = "environment/tests.rs"]
mod tests;
