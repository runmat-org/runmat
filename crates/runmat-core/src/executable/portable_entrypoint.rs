use super::ExecutableUnit;

pub(super) struct PortableEntrypoint {
    pub(super) name: String,
    pub(super) function: runmat_types::ProgramFunctionId,
    /// Session/compiler identity of the same body. This is intentionally not
    /// serialized into the portable executable identity.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) compiled_function: runmat_hir::FunctionId,
    pub(super) kind: runmat_execution::ExecutableEntrypointKind,
}

pub(super) fn resolve(
    unit: &ExecutableUnit,
    preferred_function: Option<&str>,
) -> Result<PortableEntrypoint, String> {
    if let Some(name) = preferred_function {
        let function = unit
            .native_function_id(name)
            .ok_or_else(|| format!("source does not define requested function '{name}'"))?;
        return function_entrypoint(unit, name, function);
    }

    let source_stem = std::path::Path::new(&unit.source().relative_path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("script");
    if let Some(function) = unit.native_function_id(source_stem) {
        return function_entrypoint(unit, source_stem, function);
    }
    script_entrypoint(unit)
}

/// An interactive definition-only source has no script entrypoint. Select an
/// exact compiled function for indexing its callable artifact without
/// changing the ordinary source-name/script executable selection contract.
pub(super) fn resolve_compiled_function(
    unit: &ExecutableUnit,
    function: runmat_hir::FunctionId,
) -> Result<PortableEntrypoint, String> {
    let metadata = unit
        .mir()
        .functions
        .get(&function)
        .ok_or_else(|| "callable artifact function has no MIR metadata".to_string())?;
    if metadata.kind == runmat_hir::FunctionKind::SyntheticEntrypoint {
        return Err("synthetic script entrypoint is not a callable definition".into());
    }
    function_entrypoint(unit, &metadata.name.0, function)
}

pub(super) fn resolve_current_definition(
    unit: &ExecutableUnit,
    current: &std::collections::HashSet<runmat_hir::FunctionId>,
    primary_source: runmat_types::ProgramSourceId,
) -> Result<PortableEntrypoint, String> {
    use runmat_types::FunctionSourceRole;

    let mut candidates = current
        .iter()
        .filter_map(|function| {
            let metadata = unit.mir().functions.get(function)?;
            let body = unit.functions().get(*function)?;
            if metadata.kind == runmat_hir::FunctionKind::SyntheticEntrypoint
                || body.binding_origin.is_none()
            {
                return None;
            }
            let role = match metadata.source_role {
                FunctionSourceRole::Primary => 0,
                FunctionSourceRole::Local => 1,
                FunctionSourceRole::Method => 2,
                FunctionSourceRole::Nested => 3,
                FunctionSourceRole::Anonymous => 4,
                FunctionSourceRole::Unknown | FunctionSourceRole::Synthetic => 5,
            };
            Some((
                metadata.source != primary_source,
                role,
                metadata.source.0,
                function.0,
                *function,
            ))
        })
        .collect::<Vec<_>>();
    candidates.sort_unstable();
    let function = candidates
        .first()
        .map(|candidate| candidate.4)
        .ok_or_else(|| "definition-only executable has no current callable body".to_string())?;
    resolve_compiled_function(unit, function)
}

fn function_entrypoint(
    unit: &ExecutableUnit,
    name: &str,
    function: runmat_hir::FunctionId,
) -> Result<PortableEntrypoint, String> {
    let body = unit
        .functions()
        .get(function)
        .ok_or_else(|| "executable entrypoint has no compiled function body".to_string())?;
    Ok(PortableEntrypoint {
        name: name.to_string(),
        function: body.portable_function,
        #[cfg(not(target_arch = "wasm32"))]
        compiled_function: function,
        kind: runmat_execution::ExecutableEntrypointKind::Function,
    })
}

fn script_entrypoint(unit: &ExecutableUnit) -> Result<PortableEntrypoint, String> {
    if unit.bytecode().instructions.is_empty() {
        return Err("executable has no callable function or script entrypoint".into());
    }
    let function = unit
        .script_entrypoint()
        .ok_or_else(|| "executable script has no synthetic entrypoint".to_string())?;
    let metadata = unit
        .mir()
        .functions
        .get(&function)
        .ok_or_else(|| "executable script entrypoint has no MIR metadata".to_string())?;
    if metadata.kind != runmat_hir::FunctionKind::SyntheticEntrypoint {
        return Err("executable script entrypoint is not synthetic MIR".into());
    }
    if !unit.mir().bodies.contains_key(&function) || !unit.mir().entrypoints.contains(&function) {
        return Err("executable script entrypoint has no exact MIR body".into());
    }
    let layout = unit.vm_layout();
    if layout
        .functions
        .get(&function)
        .is_none_or(|function_layout| function_layout.function != function)
        || !layout
            .entrypoints
            .values()
            .any(|entrypoint| entrypoint.target == function)
    {
        return Err("executable script entrypoint disagrees with its VM layout".into());
    }
    let primary_source = unit
        .source_map()
        .entries()
        .first()
        .ok_or_else(|| "executable script has no primary source".to_string())?
        .source_id;
    if metadata.source != primary_source
        || unit
            .bytecode()
            .source_id
            .and_then(runmat_vm::BytecodeSourceRef::program)
            != Some(primary_source)
    {
        return Err("executable script bytecode does not identify its primary MIR source".into());
    }
    let portable = u32::try_from(function.0)
        .map(runmat_types::ProgramFunctionId)
        .map_err(|_| "script entrypoint exceeds its portable representation".to_string())?;
    Ok(PortableEntrypoint {
        name: "script".into(),
        function: portable,
        #[cfg(not(target_arch = "wasm32"))]
        compiled_function: function,
        kind: runmat_execution::ExecutableEntrypointKind::Script,
    })
}
