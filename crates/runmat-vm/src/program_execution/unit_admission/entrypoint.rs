use super::ExecutableUnitAdmissionError;

#[derive(Clone, Debug)]
pub(super) enum AdmittedEntrypoint {
    Function {
        origin: runmat_execution::FunctionBindingOrigin,
        local_function: runmat_hir::FunctionId,
    },
    Script {
        origin: runmat_execution::FunctionBindingOrigin,
    },
    DeclarationOnly {
        class: runmat_types::ClassIdentity,
    },
}

pub(super) fn admit(
    envelope: &runmat_execution::ExecutableUnitEnvelope,
    mir: &runmat_mir::MirAssembly,
    bytecode: &crate::Bytecode,
    registry: &crate::FunctionRegistry,
    layout: &crate::VmAssemblyLayout,
    source_map: &runmat_execution::ExecutableSourceMapV2,
) -> Result<AdmittedEntrypoint, ExecutableUnitAdmissionError> {
    let selected = envelope
        .manifest
        .identity
        .validated_entrypoint()
        .map_err(|error| source_binding(error.to_string()))?;
    match selected {
        runmat_execution::ValidatedExecutableEntrypoint::Function(function) => {
            function_origin(envelope, registry, function)
        }
        runmat_execution::ValidatedExecutableEntrypoint::Script(function) => {
            Ok(AdmittedEntrypoint::Script {
                origin: script_origin(envelope, mir, bytecode, layout, source_map, function)?,
            })
        }
        runmat_execution::ValidatedExecutableEntrypoint::DeclarationOnly(class) => {
            declaration_origin(mir, bytecode, registry, source_map, class)
        }
    }
}

fn function_origin(
    envelope: &runmat_execution::ExecutableUnitEnvelope,
    registry: &crate::FunctionRegistry,
    portable_function: runmat_types::ProgramFunctionId,
) -> Result<AdmittedEntrypoint, ExecutableUnitAdmissionError> {
    let identity = &envelope.manifest.identity;
    let local_function = registry.resolve_name(&identity.entrypoint).ok_or_else(|| {
        source_binding("executable entrypoint is absent from its admitted function registry")
    })?;
    let function = registry.get(local_function).ok_or_else(|| {
        source_binding("executable entrypoint is absent from its admitted function registry")
    })?;
    if function.portable_function != portable_function {
        return Err(source_binding(
            "executable entrypoint identity disagrees with its admitted function body",
        ));
    }
    let expected = expected_origin(envelope, portable_function)?;
    if function.binding_origin != Some(expected) {
        return Err(source_binding(
            "executable entrypoint lies outside its admitted revision",
        ));
    }
    Ok(AdmittedEntrypoint::Function {
        origin: expected,
        local_function,
    })
}

fn script_origin(
    envelope: &runmat_execution::ExecutableUnitEnvelope,
    mir: &runmat_mir::MirAssembly,
    bytecode: &crate::Bytecode,
    layout: &crate::VmAssemblyLayout,
    source_map: &runmat_execution::ExecutableSourceMapV2,
    portable_function: runmat_types::ProgramFunctionId,
) -> Result<runmat_execution::FunctionBindingOrigin, ExecutableUnitAdmissionError> {
    let identity = &envelope.manifest.identity;
    if identity.entrypoint != "script" {
        return Err(source_binding(
            "script executable has a non-script entrypoint name",
        ));
    }
    let function = runmat_hir::FunctionId(
        usize::try_from(portable_function.0)
            .map_err(|_| source_binding("script entrypoint identity exceeds this host"))?,
    );
    let metadata = mir.functions.get(&function).ok_or_else(|| {
        source_binding("script entrypoint is absent from its admitted MIR metadata")
    })?;
    if metadata.kind != runmat_hir::FunctionKind::SyntheticEntrypoint {
        return Err(source_binding(
            "script entrypoint does not identify synthetic MIR",
        ));
    }
    if !mir.entrypoints.contains(&function) || !mir.bodies.contains_key(&function) {
        return Err(source_binding(
            "script entrypoint has no exact admitted MIR body",
        ));
    }
    let function_layout = layout
        .functions
        .get(&function)
        .ok_or_else(|| source_binding("script entrypoint is absent from its admitted VM layout"))?;
    if function_layout.function != function
        || !layout
            .entrypoints
            .values()
            .any(|entrypoint| entrypoint.target == function)
    {
        return Err(source_binding(
            "script entrypoint identity disagrees with its admitted VM layout",
        ));
    }
    if bytecode.instructions.is_empty() {
        return Err(source_binding(
            "script entrypoint has no primary top-level bytecode",
        ));
    }
    let primary_source = source_map
        .entries
        .first()
        .ok_or_else(|| source_binding("script executable has no primary source"))?
        .source_id;
    if metadata.source != primary_source
        || bytecode
            .source_id
            .and_then(crate::BytecodeSourceRef::program)
            != Some(primary_source)
    {
        return Err(source_binding(
            "script bytecode does not identify its primary MIR source",
        ));
    }
    expected_origin(envelope, portable_function)
}

fn declaration_origin(
    mir: &runmat_mir::MirAssembly,
    bytecode: &crate::Bytecode,
    registry: &crate::FunctionRegistry,
    source_map: &runmat_execution::ExecutableSourceMapV2,
    class: runmat_types::ClassIdentity,
) -> Result<AdmittedEntrypoint, ExecutableUnitAdmissionError> {
    let primary = source_map
        .entries
        .first()
        .ok_or_else(|| source_binding("declaration-only executable has no primary source"))?;
    if mir.functions.values().any(|metadata| {
        metadata.source == primary.source_id
            && metadata.class_method_owner.as_ref().is_some_and(|owner| {
                owner.role == runmat_types::ClassMethodRole::Constructor
                    && !owner.is_static
                    && owner.declaring_class == class
            })
    }) {
        return Err(source_binding(
            "declaration-only executable contains a constructor body",
        ));
    }
    let primary_registration = (bytecode.source_id.and_then(crate::BytecodeSourceRef::program)
        == Some(primary.source_id)
        && bytecode.instructions.iter().any(|instruction| {
            matches!(instruction, crate::Instr::RegisterClass { name, .. } if name == &class)
        }))
        || registry.functions.values().any(|body| {
            body.source_id.and_then(crate::BytecodeSourceRef::program)
                == Some(primary.source_id)
                && body.instructions.iter().any(|instruction| {
                    matches!(instruction, crate::Instr::RegisterClass { name, .. } if name == &class)
                })
        });
    if !mir
        .classes
        .iter()
        .any(|declaration| declaration.name.display_name().as_deref() == Some(class.display_name()))
        || !primary_registration
    {
        return Err(source_binding(
            "declaration-only executable lacks its primary admitted class",
        ));
    }
    Ok(AdmittedEntrypoint::DeclarationOnly { class })
}

fn expected_origin(
    envelope: &runmat_execution::ExecutableUnitEnvelope,
    portable_function: runmat_types::ProgramFunctionId,
) -> Result<runmat_execution::FunctionBindingOrigin, ExecutableUnitAdmissionError> {
    let identity = &envelope.manifest.identity;
    runmat_execution::FunctionBindingOrigin::from_revision(&identity.program, portable_function)
        .map_err(|error| {
            source_binding(format!(
                "executable entrypoint has an invalid immutable origin: {error}"
            ))
        })
}

fn source_binding(message: impl Into<String>) -> ExecutableUnitAdmissionError {
    ExecutableUnitAdmissionError::SourceBinding(message.into())
}
