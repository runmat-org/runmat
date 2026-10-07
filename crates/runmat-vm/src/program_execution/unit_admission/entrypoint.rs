use super::ExecutableUnitAdmissionError;

pub(super) struct AdmittedEntrypoint {
    pub origin: runmat_execution::FunctionBindingOrigin,
    /// Registry-local identity selected by the validated manifest entrypoint.
    /// The manifest's portable ordinal is not a registry lookup key.
    pub local_function: Option<runmat_hir::FunctionId>,
}

pub(super) fn admit(
    envelope: &runmat_execution::ExecutableUnitEnvelope,
    mir: &runmat_mir::MirAssembly,
    bytecode: &crate::Bytecode,
    registry: &crate::FunctionRegistry,
    layout: &crate::VmAssemblyLayout,
    source_map: &runmat_execution::ExecutableSourceMapV2,
) -> Result<AdmittedEntrypoint, ExecutableUnitAdmissionError> {
    match envelope.manifest.identity.entrypoint_kind {
        runmat_execution::ExecutableEntrypointKind::Function => function_origin(envelope, registry),
        runmat_execution::ExecutableEntrypointKind::Script => Ok(AdmittedEntrypoint {
            origin: script_origin(envelope, mir, bytecode, layout, source_map)?,
            local_function: None,
        }),
    }
}

fn function_origin(
    envelope: &runmat_execution::ExecutableUnitEnvelope,
    registry: &crate::FunctionRegistry,
) -> Result<AdmittedEntrypoint, ExecutableUnitAdmissionError> {
    let identity = &envelope.manifest.identity;
    let local_function = registry.resolve_name(&identity.entrypoint).ok_or_else(|| {
        source_binding("executable entrypoint is absent from its admitted function registry")
    })?;
    let function = registry.get(local_function).ok_or_else(|| {
        source_binding("executable entrypoint is absent from its admitted function registry")
    })?;
    if function.portable_function != identity.entrypoint_function {
        return Err(source_binding(
            "executable entrypoint identity disagrees with its admitted function body",
        ));
    }
    let expected = expected_origin(envelope)?;
    if function.binding_origin != Some(expected) {
        return Err(source_binding(
            "executable entrypoint lies outside its admitted revision",
        ));
    }
    Ok(AdmittedEntrypoint {
        origin: expected,
        local_function: Some(local_function),
    })
}

fn script_origin(
    envelope: &runmat_execution::ExecutableUnitEnvelope,
    mir: &runmat_mir::MirAssembly,
    bytecode: &crate::Bytecode,
    layout: &crate::VmAssemblyLayout,
    source_map: &runmat_execution::ExecutableSourceMapV2,
) -> Result<runmat_execution::FunctionBindingOrigin, ExecutableUnitAdmissionError> {
    let identity = &envelope.manifest.identity;
    if identity.entrypoint != "script" {
        return Err(source_binding(
            "script executable has a non-script entrypoint name",
        ));
    }
    let function = runmat_hir::FunctionId(
        usize::try_from(identity.entrypoint_function.0)
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
    expected_origin(envelope)
}

fn expected_origin(
    envelope: &runmat_execution::ExecutableUnitEnvelope,
) -> Result<runmat_execution::FunctionBindingOrigin, ExecutableUnitAdmissionError> {
    let identity = &envelope.manifest.identity;
    runmat_execution::FunctionBindingOrigin::from_revision(
        &identity.program,
        identity.entrypoint_function,
    )
    .map_err(|error| {
        source_binding(format!(
            "executable entrypoint has an invalid immutable origin: {error}"
        ))
    })
}

fn source_binding(message: impl Into<String>) -> ExecutableUnitAdmissionError {
    ExecutableUnitAdmissionError::SourceBinding(message.into())
}
