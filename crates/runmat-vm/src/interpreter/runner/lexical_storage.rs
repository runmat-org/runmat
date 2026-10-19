use std::collections::BTreeSet;

use runmat_hir::FunctionId;
use runmat_runtime::RuntimeError;

use crate::bytecode::{FunctionBytecode, FunctionRegistry, InheritedStorageKind, Instr};

/// Validate the complete typed ancestor and declaration before a nested frame
/// may bind its local slot to session-owned global or persistent storage.
pub(super) fn validate_inherited_storage(
    function: &FunctionBytecode,
    registry: &FunctionRegistry,
) -> Result<(), RuntimeError> {
    let mut slots = BTreeSet::new();
    let mut bindings = BTreeSet::new();
    for binding in &function.inherited_storage_slots {
        if binding.slot >= function.var_count
            || !slots.insert(binding.slot)
            || !bindings.insert(binding.binding)
            || function
                .lexical_binding_slots
                .iter()
                .any(|shared| shared.slot == binding.slot || shared.binding == binding.binding)
        {
            return Err(super::mex(
                "LexicalStorage",
                "canonical storage cannot overlap another frame binding",
            ));
        }
        let mut cursor = function.lexical_parent;
        let mut seen = BTreeSet::<FunctionId>::new();
        let mut admitted_owner = None;
        while let Some(parent) = cursor {
            if !seen.insert(parent) {
                return Err(super::mex(
                    "LexicalStorage",
                    "cyclic canonical storage ancestry",
                ));
            }
            let ancestor = registry.get(parent).ok_or_else(|| {
                super::mex(
                    "LexicalStorage",
                    "canonical storage ancestor is unavailable",
                )
            })?;
            if ancestor.source_id != function.source_id {
                return Err(super::mex(
                    "LexicalStorage",
                    "canonical storage crossed a source owner",
                ));
            }
            if parent == binding.owner {
                admitted_owner = Some(ancestor);
                break;
            }
            cursor = ancestor.lexical_parent;
        }
        let owner = admitted_owner.ok_or_else(|| {
            super::mex(
                "LexicalStorage",
                "canonical storage owner is not an ancestor",
            )
        })?;
        if owner.portable_function != binding.portable_owner
            || owner.display_name != binding.owner_function
        {
            return Err(super::mex(
                "LexicalStorage",
                "canonical storage owner identity changed",
            ));
        }
        let declared = owner.instructions.iter().any(|instruction| {
            match (binding.kind, instruction) {
                (InheritedStorageKind::Global, Instr::DeclareGlobalNamed(_, names))
                | (InheritedStorageKind::Persistent, Instr::DeclarePersistentNamed(_, names)) => {
                    names.iter().any(|name| name == &binding.name)
                }
                _ => false,
            }
        });
        if !declared {
            return Err(super::mex(
                "LexicalStorage",
                "canonical storage was not declared by its owner",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::program::{InheritedStorageSlot, LexicalBindingKind, LexicalBindingSlot};
    use runmat_types::{BindingId, ProgramFunctionId};

    #[test]
    fn canonical_storage_and_shared_cell_cannot_claim_the_same_frame_slot() {
        let function = FunctionBytecode {
            var_count: 1,
            lexical_binding_slots: vec![LexicalBindingSlot {
                binding: BindingId(1),
                slot: 0,
                kind: LexicalBindingKind::Inherited,
            }],
            inherited_storage_slots: vec![InheritedStorageSlot {
                binding: BindingId(1),
                slot: 0,
                owner: FunctionId(2),
                portable_owner: ProgramFunctionId(2),
                name: "P".into(),
                owner_function: "outer".into(),
                kind: InheritedStorageKind::Persistent,
            }],
            ..Default::default()
        };
        let error = validate_inherited_storage(&function, &FunctionRegistry::default())
            .expect_err("a forged dual owner must fail before frame admission");
        assert!(error.to_string().contains("overlap"), "{error}");
    }
}
