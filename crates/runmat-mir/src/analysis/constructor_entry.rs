use crate::{MirAssembly, MirBody, MirLocalId};

/// The instance that a constructor receives at invocation is its first fixed
/// output, not an assignment performed by a statement in the body.
pub(super) fn instance_output_local(assembly: &MirAssembly, body: &MirBody) -> Option<MirLocalId> {
    let owner = assembly
        .functions
        .get(&body.function)?
        .class_method_owner
        .as_ref()?;
    if owner.role != runmat_types::ClassMethodRole::Constructor || owner.is_static {
        return None;
    }
    let binding = body.abi.fixed_output_bindings().next()?;
    body.locals
        .iter()
        .find(|local| local.binding == Some(*binding))
        .map(|local| local.id)
}

#[cfg(test)]
#[path = "constructor_entry/tests.rs"]
mod tests;
