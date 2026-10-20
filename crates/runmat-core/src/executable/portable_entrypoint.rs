use super::ExecutableUnit;

pub(super) enum PortableEntrypoint {
    Callable {
        name: String,
        compiled_function: runmat_hir::FunctionId,
        function: runmat_types::ProgramFunctionId,
        kind: runmat_execution::ExecutableEntrypointKind,
    },
    DeclarationOnly(runmat_types::ClassIdentity),
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

pub(super) fn resolve_constructor_source(
    unit: &ExecutableUnit,
    class: &runmat_types::ClassIdentity,
    primary_source: runmat_types::ProgramSourceId,
) -> Result<PortableEntrypoint, String> {
    let function = unit
        .source_constructor_function_id(class, primary_source)
        .ok_or_else(|| "selected source has no unique declared class constructor".to_string())?;
    resolve_compiled_function(unit, function)
}

/// A class with no declared constructor still has an exact source product.
/// It is metadata only and cannot be invoked as a callable entrypoint.
pub(super) fn resolve_declaration_source(
    unit: &ExecutableUnit,
    class: &runmat_types::ClassIdentity,
    primary_source: runmat_types::ProgramSourceId,
) -> Result<PortableEntrypoint, String> {
    if unit.mir().functions.values().any(|metadata| {
        metadata.source == primary_source
            && metadata.class_method_owner.as_ref().is_some_and(|owner| {
                owner.role == runmat_types::ClassMethodRole::Constructor
                    && !owner.is_static
                    && &owner.declaring_class == class
            })
    }) {
        return Err("declaration-only product has a constructor body".into());
    }
    let primary = unit
        .source_map()
        .entries()
        .first()
        .ok_or("declaration-only product has no primary source")?;
    let primary_registration = (unit
        .bytecode()
        .source_id
        .and_then(runmat_vm::BytecodeSourceRef::program)
        == Some(primary_source)
        && unit.bytecode().instructions.iter().any(|instruction| {
            matches!(instruction, runmat_vm::Instr::RegisterClass { name, .. } if name == class)
        }))
        || unit.functions().functions.values().any(|body| {
            body.source_id.and_then(runmat_vm::BytecodeSourceRef::program)
                == Some(primary_source)
                && body.instructions.iter().any(|instruction| {
                    matches!(instruction, runmat_vm::Instr::RegisterClass { name, .. } if name == class)
                })
        });
    if primary.source_id != primary_source
        || !unit.mir().classes.iter().any(|declaration| {
            declaration.name.display_name().as_deref() == Some(class.display_name())
        })
        || !primary_registration
    {
        return Err("declaration-only product lacks its primary class declaration".into());
    }
    Ok(PortableEntrypoint::DeclarationOnly(class.clone()))
}

/// Identify a genuine primary-source class declaration when no new callable
/// entrypoint is selected. A copied prior function body can still carry a
/// RegisterClass prologue; only typed MIR and the declaration validator count.
fn primary_declaration_only_class(
    unit: &ExecutableUnit,
    primary_source: runmat_types::ProgramSourceId,
) -> Result<Option<runmat_types::ClassIdentity>, String> {
    let declares = |instruction: &runmat_vm::Instr| match instruction {
        runmat_vm::Instr::RegisterClass { name, .. } => Some(name.clone()),
        _ => None,
    };
    let mut candidates = unit
        .bytecode()
        .source_id
        .and_then(runmat_vm::BytecodeSourceRef::program)
        .filter(|source| *source == primary_source)
        .into_iter()
        .flat_map(|_| unit.bytecode().instructions.iter().filter_map(declares))
        .collect::<std::collections::BTreeSet<_>>();
    for body in unit.functions().functions.values().filter(|body| {
        body.source_id
            .and_then(runmat_vm::BytecodeSourceRef::program)
            == Some(primary_source)
    }) {
        candidates.extend(body.instructions.iter().filter_map(declares));
    }
    let mut selected = None;
    for class in candidates {
        let has_constructor = unit.mir().functions.values().any(|metadata| {
            metadata.source == primary_source
                && metadata.class_method_owner.as_ref().is_some_and(|owner| {
                    owner.role == runmat_types::ClassMethodRole::Constructor
                        && !owner.is_static
                        && owner.declaring_class == class
                })
        });
        if has_constructor {
            continue;
        }
        resolve_declaration_source(unit, &class, primary_source)?;
        if selected.replace(class).is_some() {
            return Err("primary source has ambiguous declaration-only classes".into());
        }
    }
    Ok(selected)
}

impl ExecutableUnit {
    pub(crate) fn primary_declaration_only_class(
        &self,
        primary_source: runmat_types::ProgramSourceId,
    ) -> Result<Option<runmat_types::ClassIdentity>, String> {
        primary_declaration_only_class(self, primary_source)
    }
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
    if body.function != function {
        return Err("executable entrypoint disagrees with its compiled function identity".into());
    }
    // The preferred source name can be a class name while the registry owns
    // its constructor under the compiled method name. Keep source selection
    // separate from the name that the admitted registry can actually resolve.
    let registered_name = if unit.functions().resolve_name(name) == Some(function) {
        name
    } else if unit.functions().resolve_name(&body.display_name) == Some(function) {
        &body.display_name
    } else {
        return Err("executable entrypoint has no exact admitted registry name".into());
    };
    let expected_origin = runmat_execution::FunctionBindingOrigin::from_revision(
        &unit.revision().program_revision,
        body.portable_function,
    )
    .map_err(|error| error.to_string())?;
    if body
        .binding_origin
        .is_some_and(|origin| origin != expected_origin)
    {
        return Err("executable entrypoint disagrees with its immutable origin".into());
    }
    Ok(PortableEntrypoint::Callable {
        name: registered_name.to_string(),
        compiled_function: function,
        function: body.portable_function,
        kind: runmat_execution::ExecutableEntrypointKind::Function,
    })
}

pub(super) fn script_entrypoint(unit: &ExecutableUnit) -> Result<PortableEntrypoint, String> {
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
    let portable = unit.portable_function_for_local(function)?;
    Ok(PortableEntrypoint::Callable {
        name: "script".into(),
        compiled_function: function,
        function: portable,
        kind: runmat_execution::ExecutableEntrypointKind::Script,
    })
}
