//! Admit explicitly declared cross-product static references without treating
//! arbitrary missing session IDs as external functions.

use std::collections::{HashMap, HashSet};

use runmat_execution::executable::{
    decode_linked_native_imports, decode_static_imports, ExecutableStaticCaller,
    ExecutableStaticImport, ExecutableStaticImportKind, LinkedNativeStaticImport,
    LINKED_NATIVE_IMPORT_SECTION, STATIC_IMPORT_SECTION,
};
use runmat_execution::value::GraphStaticReferenceSite;
use runmat_execution::ExecutableUnitEnvelope;

use super::ExecutableUnitAdmissionError;
use crate::bytecode::program::function_remap::{
    instruction_reference_sites, FunctionReferenceKind,
};
use crate::{Bytecode, FunctionRegistry};

pub(super) fn admit(
    envelope: &ExecutableUnitEnvelope,
    bytecode: &Bytecode,
    registry: &FunctionRegistry,
) -> Result<
    (Vec<ExecutableStaticImport>, Vec<LinkedNativeStaticImport>),
    ExecutableUnitAdmissionError,
> {
    let imports = envelope
        .manifest
        .optional_sections
        .iter()
        .find(|section| section.name == STATIC_IMPORT_SECTION)
        .map(decode_static_imports)
        .transpose()
        .map_err(ExecutableUnitAdmissionError::InvalidEnvelope)?
        .unwrap_or_default();
    let linked_native = envelope
        .manifest
        .optional_sections
        .iter()
        .find(|section| section.name == LINKED_NATIVE_IMPORT_SECTION)
        .map(decode_linked_native_imports)
        .transpose()
        .map_err(ExecutableUnitAdmissionError::InvalidEnvelope)?
        .unwrap_or_default();
    let mut declared = HashMap::new();
    for import in &imports {
        if let ExecutableStaticCaller::Function(caller) = import.caller {
            let program = envelope
                .manifest
                .identity
                .program
                .identity_digest()
                .map_err(ExecutableUnitAdmissionError::InvalidEnvelope)?;
            if caller.program != program {
                return Err(invalid("static caller belongs to another program"));
            }
        }
        if declared
            .insert((import.caller, import.site), (import.target, import.kind))
            .is_some()
        {
            return Err(invalid("static import repeats one instruction site"));
        }
    }
    for import in &linked_native {
        if let ExecutableStaticCaller::Function(caller) = import.caller {
            let program = envelope
                .manifest
                .identity
                .program
                .identity_digest()
                .map_err(ExecutableUnitAdmissionError::InvalidEnvelope)?;
            if caller.program != program {
                return Err(invalid("linked-native caller belongs to another program"));
            }
        }
        if declared
            .insert(
                (import.caller, import.site),
                (import.target, ExecutableStaticImportKind::ExplicitExternal),
            )
            .is_some()
        {
            return Err(invalid("linked-native import repeats a VM or native site"));
        }
    }
    let mut observed = HashSet::new();
    validate_stream(
        ExecutableStaticCaller::Script,
        &bytecode.instructions,
        registry,
        &declared,
        &mut observed,
    )?;
    for body in registry.functions.values() {
        let origin = body
            .binding_origin
            .ok_or_else(|| invalid("static-import caller has no portable origin"))?;
        validate_stream(
            ExecutableStaticCaller::Function(origin),
            &body.instructions,
            registry,
            &declared,
            &mut observed,
        )?;
    }
    if observed.len() != declared.len() {
        return Err(invalid(
            "static-import section has an unused reference site",
        ));
    }
    Ok((imports, linked_native))
}

fn validate_stream(
    caller: ExecutableStaticCaller,
    instructions: &[crate::bytecode::Instr],
    registry: &FunctionRegistry,
    declared: &HashMap<
        (ExecutableStaticCaller, GraphStaticReferenceSite),
        (
            runmat_execution::FunctionBindingOrigin,
            ExecutableStaticImportKind,
        ),
    >,
    observed: &mut HashSet<(ExecutableStaticCaller, GraphStaticReferenceSite)>,
) -> Result<(), ExecutableUnitAdmissionError> {
    for (pc, instruction) in instructions.iter().enumerate() {
        let sites = instruction_reference_sites(instruction, pc).map_err(invalid)?;
        for (site, function, kind) in sites {
            let key = (caller, site);
            match (kind, registry.functions.get(&function), declared.get(&key)) {
                // An explicitly external reference can share a numeric ID
                // with a local body. The site declaration, not that number,
                // selects the external identity domain.
                (
                    FunctionReferenceKind::External,
                    _,
                    Some((_, ExecutableStaticImportKind::ExplicitExternal)),
                ) => {
                    observed.insert(key);
                }
                // A compiler-bound direct identity without a local body is
                // also an external dependency. The producer must seal its
                // exact inherited origin at this site.
                (
                    FunctionReferenceKind::Local,
                    None,
                    Some((_, ExecutableStaticImportKind::PartitionedDefinition)),
                ) => {
                    observed.insert(key);
                }
                (_, _, Some(_)) => {
                    return Err(invalid(
                        "static-import provenance differs from its instruction",
                    ));
                }
                (FunctionReferenceKind::External, _, None) => {
                    return Err(invalid("external reference has no admitted static import"));
                }
                (_, Some(_), None) => {}
                (FunctionReferenceKind::Local, None, None) => {
                    return Err(invalid("local reference is outside its admitted unit"));
                }
            }
        }
    }
    Ok(())
}

fn invalid(message: &'static str) -> ExecutableUnitAdmissionError {
    ExecutableUnitAdmissionError::SourceBinding(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::{FunctionBytecode, Instr};
    use runmat_execution::Digest;
    use runmat_hir::FunctionId;
    use runmat_types::{FunctionSourceRole, ProgramFunctionId};

    #[test]
    fn external_site_is_authoritative_even_when_numeric_id_collides() {
        let local = FunctionId(7);
        let registry = FunctionRegistry::new(HashMap::from([(
            local,
            FunctionBytecode {
                function: local,
                ..FunctionBytecode::default()
            },
        )]));
        let site = GraphStaticReferenceSite {
            instruction: 0,
            occurrence: 0,
        };
        let caller = ExecutableStaticCaller::Script;
        let target = runmat_execution::FunctionBindingOrigin {
            program: Digest::sha256(b"selected parent"),
            function: ProgramFunctionId(2),
        };
        let declared = HashMap::from([(
            (caller, site),
            (target, ExecutableStaticImportKind::ExplicitExternal),
        )]);
        let external = [Instr::CreateExternalBoundFunctionHandle(
            local,
            "step".into(),
            FunctionSourceRole::Local,
        )];
        let mut observed = HashSet::new();
        validate_stream(caller, &external, &registry, &declared, &mut observed)
            .expect("typed external site wins over colliding local number");
        assert!(observed.contains(&(caller, site)));
        assert!(validate_stream(
            caller,
            &external,
            &registry,
            &HashMap::new(),
            &mut HashSet::new(),
        )
        .is_err());
        let local_instruction = [Instr::CreateBoundFunctionHandle(
            local,
            "step".into(),
            FunctionSourceRole::Local,
        )];
        assert!(validate_stream(
            caller,
            &local_instruction,
            &registry,
            &declared,
            &mut HashSet::new(),
        )
        .is_err());
    }

    #[test]
    fn direct_site_without_local_layout_requires_selected_import() {
        let caller = ExecutableStaticCaller::Script;
        let site = GraphStaticReferenceSite {
            instruction: 0,
            occurrence: 0,
        };
        let target = runmat_execution::FunctionBindingOrigin {
            program: Digest::sha256(b"private parent"),
            function: ProgramFunctionId(2),
        };
        let instruction = [Instr::CreateBoundFunctionHandle(
            FunctionId(2),
            "step".into(),
            FunctionSourceRole::Local,
        )];
        let empty = FunctionRegistry::new(HashMap::new());
        assert!(validate_stream(
            caller,
            &instruction,
            &empty,
            &HashMap::new(),
            &mut HashSet::new(),
        )
        .is_err());
        let declared = HashMap::from([(
            (caller, site),
            (target, ExecutableStaticImportKind::PartitionedDefinition),
        )]);
        let mut observed = HashSet::new();
        validate_stream(caller, &instruction, &empty, &declared, &mut observed)
            .expect("typed selected parent import");
        assert!(observed.contains(&(caller, site)));
    }
}
