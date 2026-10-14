use runmat_gc::{gc_allocate_rooted, gc_root, gc_with_value, gc_with_value_mut, ExplicitRoot};
use runmat_types::BindingId;
use runmat_value::{CellArray, LexicalEnvironment, Value};

use crate::runtime_error::semantic_error;
use crate::RuntimeError;

/// Allocate a distinct activation. The returned root belongs to the invoking
/// frame; escaped closures retain the handle through their Trace edge.
pub fn allocate_environment(
    parent: Option<LexicalEnvironment>,
) -> Result<(LexicalEnvironment, Vec<ExplicitRoot>), RuntimeError> {
    // Root every ancestor before allocating the child; the allocation may
    // collect, and the non-GC Rc metadata is not itself a tracing root.
    let mut roots = parent
        .as_ref()
        .map(root_environment)
        .transpose()?
        .unwrap_or_default();
    let cell = CellArray::new(Vec::new(), 1, 0).map_err(|error| environment_error(&error))?;
    let root = gc_allocate_rooted(Value::Cell(cell)).map_err(environment_gc_error)?;
    let environment = LexicalEnvironment::new(root.handle(), parent);
    roots.push(root);
    Ok((environment, roots))
}

/// Root a copied closure environment at call admission, before its async
/// invoker or callback can be queued and the source closure can be dropped.
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

pub fn read_binding(
    environment: &LexicalEnvironment,
    binding: BindingId,
) -> Result<Option<Value>, RuntimeError> {
    let Some((handle, slot)) = environment.resolve_slot(binding) else {
        return Ok(None);
    };
    gc_with_value(&handle, |value| match value {
        Value::Cell(cell) => cell
            .data
            .get(slot)
            .cloned()
            .ok_or_else(|| environment_error("shared binding slot is outside its cell")),
        _ => Err(environment_error("shared binding owner is not a cell")),
    })
    .map_err(environment_gc_error)?
    .map(Some)
}

pub fn write_binding(
    environment: &LexicalEnvironment,
    binding: BindingId,
    value: Value,
) -> Result<(), RuntimeError> {
    let _input_root = crate::execution::RootedValueLease::for_traced(&value)
        .map_err(|error| error.into_runtime_error())?;
    runmat_value::validate_no_transient_sequence(&value)
        .map_err(|error| environment_error(&error.to_string()))?;
    if let Some(owner) = environment.owner_for_binding(binding) {
        if let Some(slot) = owner.local_slot(binding) {
            return replace_binding_slot(owner.handle(), slot, value);
        }
        return write_owned_binding(&owner, binding, value);
    }
    Err(environment_error(
        "inherited shared binding has no typed owner",
    ))
}

/// Creation of a binding owned by this activation must never resolve a slot
/// in its recursive caller's parent node, even with the same BindingId.
pub fn write_owned_binding(
    environment: &LexicalEnvironment,
    binding: BindingId,
    value: Value,
) -> Result<(), RuntimeError> {
    let _input_root = crate::execution::RootedValueLease::for_traced(&value)
        .map_err(|error| error.into_runtime_error())?;
    runmat_value::validate_no_transient_sequence(&value)
        .map_err(|error| environment_error(&error.to_string()))?;
    if !environment.owns_binding(binding) {
        return Err(environment_error("owned shared binding was not reserved"));
    }
    if let Some(slot) = environment.local_slot(binding) {
        return replace_binding_slot(environment.handle(), slot, value);
    }
    if let Some(slot) = environment.reusable_local_slot(binding) {
        replace_binding_slot(environment.handle(), slot, value)?;
        if !environment.register_slot(binding, slot) {
            return Err(environment_error(
                "vacant binding was concurrently assigned",
            ));
        }
        return Ok(());
    }
    append_owned_binding(environment, binding, value)
}

fn replace_binding_slot(
    handle: runmat_gc::GcHandle,
    slot: usize,
    value: Value,
) -> Result<(), RuntimeError> {
    gc_with_value_mut(&handle, |stored| match stored {
        Value::Cell(cell) => {
            let target = cell
                .data
                .get_mut(slot)
                .ok_or_else(|| environment_error("shared binding slot is outside its cell"))?;
            *target = value;
            Ok(())
        }
        _ => Err(environment_error("shared binding owner is not a cell")),
    })
    .map_err(environment_gc_error)?
}

fn append_owned_binding(
    environment: &LexicalEnvironment,
    binding: BindingId,
    value: Value,
) -> Result<(), RuntimeError> {
    let slot = gc_with_value_mut(&environment.handle(), |stored| match stored {
        Value::Cell(cell) => {
            let slot = cell.data.len();
            cell.push_row_element(value)
                .map_err(|error| environment_error(&error))?;
            Ok(slot)
        }
        _ => Err(environment_error("shared binding owner is not a cell")),
    })
    .map_err(environment_gc_error)??;
    if !environment.register_slot(binding, slot) {
        return Err(environment_error(
            "shared binding was concurrently registered",
        ));
    }
    Ok(())
}

/// An unassigned shared local must not retain its prior managed value through
/// an escaped sibling handle. Clear the cell before removing its typed slot.
pub fn remove_binding(
    environment: &LexicalEnvironment,
    binding: BindingId,
) -> Result<(), RuntimeError> {
    let Some((handle, slot)) = environment.resolve_slot(binding) else {
        return Ok(());
    };
    clear_binding_slot(handle, slot)?;
    if !environment.unregister_slot(binding) {
        return Err(environment_error(
            "shared binding disappeared during removal",
        ));
    }
    Ok(())
}

pub fn remove_owned_binding(
    environment: &LexicalEnvironment,
    binding: BindingId,
) -> Result<(), RuntimeError> {
    let Some(slot) = environment.local_slot(binding) else {
        return Ok(());
    };
    clear_binding_slot(environment.handle(), slot)?;
    if !environment.unregister_local_slot(binding) {
        return Err(environment_error(
            "owned binding disappeared during removal",
        ));
    }
    Ok(())
}

fn clear_binding_slot(handle: runmat_gc::GcHandle, slot: usize) -> Result<(), RuntimeError> {
    gc_with_value_mut(&handle, |stored| match stored {
        Value::Cell(cell) => {
            let target = cell
                .data
                .get_mut(slot)
                .ok_or_else(|| environment_error("shared binding slot is outside its cell"))?;
            *target = Value::Num(0.0);
            Ok(())
        }
        _ => Err(environment_error("shared binding owner is not a cell")),
    })
    .map_err(environment_gc_error)?
}

fn environment_error(message: &str) -> RuntimeError {
    semantic_error("LexicalEnvironment", message)
}

fn environment_gc_error(source: runmat_gc::GcError) -> RuntimeError {
    let mut error = environment_error(&source.to_string());
    error.source = Some(Box::new(source));
    error
}

#[cfg(test)]
#[path = "environment/tests.rs"]
mod tests;
