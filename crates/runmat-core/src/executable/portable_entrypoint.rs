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
    let primary = unit.primary_source_id()?;
    let current = unit
        .compiled_current_body_ids()?
        .into_iter()
        .filter(|function| {
            unit.mir()
                .functions
                .get(function)
                .is_some_and(|metadata| metadata.source == primary && metadata.parent.is_none())
        })
        .collect::<std::collections::HashSet<_>>();
    if let Some(name) = preferred_function {
        let mut named = unit
            .mir()
            .functions
            .iter()
            .filter_map(|(function, metadata)| {
                (current.contains(function)
                    && metadata.source == primary
                    && metadata.parent.is_none()
                    && matches!(
                        metadata.kind,
                        runmat_hir::FunctionKind::Named
                            | runmat_hir::FunctionKind::ClassMethod { .. }
                    )
                    && metadata.name.0 == name)
                    .then_some(*function)
            });
        let exact = named.next();
        if named.next().is_some() {
            return Err(format!(
                "primary source defines requested function '{name}' more than once"
            ));
        }
        let function = exact
            .or_else(|| {
                let class = runmat_types::ClassIdentity::new(name).ok()?;
                unit.source_constructor_function_id(&class, primary)
                    .filter(|function| current.contains(function))
            })
            .ok_or_else(|| format!("source does not define requested function '{name}'"))?;
        return function_entrypoint(unit, name, function);
    }
    match super::SourceProductSelection::classify(unit, &current, Some(primary))? {
        super::SourceProductSelection::Script => script_entrypoint(unit),
        super::SourceProductSelection::Definitions => {
            resolve_current_definition(unit, &current, primary)
        }
        super::SourceProductSelection::Constructor { class, source } => {
            resolve_constructor_source(unit, &class, source)
        }
        super::SourceProductSelection::Declaration { class, source } => {
            resolve_declaration_source(unit, &class, source)
        }
    }
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
    compiled_function_entrypoint(unit, function)
}

/// Preserve an already-authenticated product label without allowing that text
/// to select a body. Registered aliases must resolve to the exact selected
/// compiler coordinate; nested/anonymous products may use only their body's
/// descriptive label after origin-based selection.
pub(super) fn resolve_compiled_function_with_label(
    unit: &ExecutableUnit,
    function: runmat_hir::FunctionId,
    label: &str,
) -> Result<PortableEntrypoint, String> {
    let selected = resolve_compiled_function(unit, function)?;
    let PortableEntrypoint::Callable {
        name,
        compiled_function,
        function: portable_function,
        kind,
    } = selected
    else {
        unreachable!("compiled function selection is callable")
    };
    if label != name && unit.functions().resolve_name(label) != Some(compiled_function) {
        return Err("portable entrypoint label does not identify its selected body".into());
    }
    Ok(PortableEntrypoint::Callable {
        name: label.to_string(),
        compiled_function,
        function: portable_function,
        kind,
    })
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
    validate_declaration_source(unit, class, primary_source)?;
    Ok(PortableEntrypoint::DeclarationOnly(class.clone()))
}

fn validate_declaration_source(
    unit: &ExecutableUnit,
    class: &runmat_types::ClassIdentity,
    primary_source: runmat_types::ProgramSourceId,
) -> Result<(), String> {
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
    let declaration = unit
        .mir()
        .classes
        .iter()
        .find(|declaration| {
            declaration.source == primary_source && class.matches_qualified_name(&declaration.name)
        })
        .ok_or("declaration-only product lacks its primary class declaration")?;
    if unit.primary_source_id()? != primary_source
        || primary_registration(unit, &declaration.name, primary_source) != Some(class)
    {
        return Err("declaration-only product lacks its primary class declaration".into());
    }
    Ok(())
}

/// Registration is required evidence, not declaration-source authority: a
/// companion registration can appear in a primary function prologue.
fn primary_registration<'a>(
    unit: &'a ExecutableUnit,
    class: &runmat_types::QualifiedName,
    primary_source: runmat_types::ProgramSourceId,
) -> Option<&'a runmat_types::ClassIdentity> {
    let declared = |instruction: &'a runmat_vm::Instr| match instruction {
        runmat_vm::Instr::RegisterClass { name, .. } if name.matches_qualified_name(class) => {
            Some(name)
        }
        _ => None,
    };
    if unit
        .bytecode()
        .source_id
        .and_then(runmat_vm::BytecodeSourceRef::program)
        == Some(primary_source)
    {
        if let Some(name) = unit.bytecode().instructions.iter().find_map(declared) {
            return Some(name);
        }
    }
    for body in unit.functions().functions.values().filter(|body| {
        body.source_id
            .and_then(runmat_vm::BytecodeSourceRef::program)
            == Some(primary_source)
    }) {
        if let Some(name) = body.instructions.iter().find_map(declared) {
            return Some(name);
        }
    }
    None
}

/// Only the actual original declaration source can make a primary class
/// candidate. Copied companion registration prologues grant no such identity.
fn primary_declaration_only_class(
    unit: &ExecutableUnit,
    primary_source: runmat_types::ProgramSourceId,
) -> Result<Option<runmat_types::ClassIdentity>, String> {
    let mut selected = None;
    for declaration in unit
        .mir()
        .classes
        .iter()
        .filter(|declaration| declaration.source == primary_source)
    {
        // The declaration's methods have already passed the shared exact MIR
        // source/owner consistency check. No name-based body lookup is needed.
        if declaration.methods.iter().any(|method| {
            method.role == runmat_types::ClassMethodRole::Constructor && !method.is_static
        }) {
            continue;
        }
        if selected.replace(declaration).is_some() {
            return Err("primary source has ambiguous declaration-only classes".into());
        }
    }
    let Some(declaration) = selected else {
        return Ok(None);
    };
    let class = primary_registration(unit, &declaration.name, primary_source)
        .ok_or("primary class declaration has no matching registration")?;
    validate_declaration_source(unit, class, primary_source)?;
    Ok(Some(class.clone()))
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
    } else if let Some(alias) = unit
        .functions()
        .names
        .keys()
        .filter(|alias| unit.functions().resolve_name(alias) == Some(function))
        .min()
    {
        // Text never selects another body: this reverse lookup runs only after
        // MIR selection and admits an exact registered alias of that coordinate.
        alias
    } else {
        return Err("executable entrypoint has no exact admitted registry name".into());
    };
    let selected = compiled_function_entrypoint(unit, function)?;
    let PortableEntrypoint::Callable {
        compiled_function,
        function,
        kind,
        ..
    } = selected
    else {
        unreachable!("compiled function selection is callable")
    };
    Ok(PortableEntrypoint::Callable {
        name: registered_name.to_string(),
        compiled_function,
        function,
        kind,
    })
}

/// Select an already-authenticated body by its compiler coordinate. This is
/// intentionally separate from textual lookup: nested and anonymous bodies
/// are callable only through an immutable binding origin, never by name.
fn compiled_function_entrypoint(
    unit: &ExecutableUnit,
    function: runmat_hir::FunctionId,
) -> Result<PortableEntrypoint, String> {
    let body = unit
        .functions()
        .get(function)
        .ok_or_else(|| "executable entrypoint has no compiled function body".to_string())?;
    if body.function != function {
        return Err("executable entrypoint disagrees with its compiled function identity".into());
    }
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
        name: body.display_name.clone(),
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
    let primary_source = unit.primary_source_id()?;
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
