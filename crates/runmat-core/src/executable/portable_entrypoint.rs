use super::ExecutableUnit;

pub(super) struct PortableEntrypoint {
    pub(super) name: String,
    pub(super) function: runmat_types::ProgramFunctionId,
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
        kind: runmat_execution::ExecutableEntrypointKind::Script,
    })
}
