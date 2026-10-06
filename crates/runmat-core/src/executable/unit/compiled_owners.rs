use std::collections::BTreeSet;

/// Select only bodies authored by this immutable product. MIR retains bodies
/// and declarations imported for whole-program analysis, but those inherited
/// definitions keep their earlier region and parallel compilation metadata.
pub(super) fn compiled_function_owners(
    current_program: runmat_execution::Digest,
    mir: &runmat_mir::MirAssembly,
    bytecode: &runmat_vm::Bytecode,
) -> Result<BTreeSet<runmat_hir::FunctionId>, String> {
    let entrypoints = mir.entrypoints.iter().copied().collect::<BTreeSet<_>>();
    if let Some(missing) = entrypoints
        .iter()
        .find(|function| !mir.bodies.contains_key(function))
    {
        return Err(format!("MIR entrypoint {missing:?} has no executable body"));
    }
    let mut compiled = BTreeSet::new();
    for function in mir.bodies.keys().copied() {
        let Some(body) = bytecode.function_registry.functions.get(&function) else {
            if entrypoints.contains(&function) {
                let synthetic = mir.functions.get(&function).is_some_and(|metadata| {
                    metadata.kind == runmat_hir::FunctionKind::SyntheticEntrypoint
                });
                if !synthetic {
                    return Err(format!(
                        "unregistered MIR entrypoint {function:?} is not synthetic"
                    ));
                }
                compiled.insert(function);
                continue;
            }
            return Err(format!("MIR function {function:?} has no executable body"));
        };
        let origin = body
            .binding_origin
            .ok_or_else(|| format!("MIR function {function:?} has no immutable binding origin"))?;
        if origin.program == current_program {
            compiled.insert(function);
        } else if entrypoints.contains(&function) {
            return Err(format!(
                "MIR entrypoint {function:?} belongs to a different immutable program"
            ));
        }
    }
    Ok(compiled)
}
