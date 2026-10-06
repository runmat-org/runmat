use crate::{Bytecode, BytecodeSourceRef, FunctionRegistry};
use runmat_execution::{ExecutableComponentKind, ExecutableSourceMapV2, ExecutableUnitEnvelope};

/// Import is a materialization boundary: validate all ordinal references,
/// issue fresh session leases, then project the executable body and selected
/// registry together. No serialized ordinal is ever used as a runtime ID.
pub(super) fn bind_unit_sources(
    envelope: &ExecutableUnitEnvelope,
    bytecode: &mut Bytecode,
    registry: &mut FunctionRegistry,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<runmat_runtime::context::RuntimeContext, String> {
    let payload = envelope
        .component(ExecutableComponentKind::SourceMap)
        .ok_or_else(|| "worker rejected executable unit without a source map".to_string())?;
    let source_map = ExecutableSourceMapV2::decode(&payload.bytes)
        .map_err(|error| format!("worker rejected invalid executable source map: {error}"))?;
    bytecode
        .validate_portable_source_origins()
        .map_err(str::to_string)?;
    registry
        .validate_portable_source_origins()
        .map_err(str::to_string)?;
    let in_map = |source: BytecodeSourceRef| {
        source
            .program()
            .and_then(|ordinal| usize::try_from(ordinal.0).ok())
            .is_some_and(|ordinal| ordinal < source_map.entries.len())
    };
    if bytecode.source_id.is_some_and(|source| !in_map(source))
        || registry
            .functions
            .values()
            .any(|function| function.source_id.is_some_and(|source| !in_map(source)))
    {
        return Err("worker rejected executable source ordinal outside its source map".into());
    }
    let runtime = runmat_runtime::source_context::bind_portable_source_map_v2(runtime, &source_map)
        .map_err(|error| format!("worker cannot bind executable source map: {error}"))?;
    let project = |source: BytecodeSourceRef| {
        source
            .program()
            .and_then(|ordinal| runtime.program_source_id(ordinal))
            .map(BytecodeSourceRef::Runtime)
    };
    if let Some(source) = bytecode.source_id {
        bytecode.source_id = Some(project(source).ok_or_else(|| {
            format!(
                "worker cannot project entrypoint source ordinal {}",
                source.program().map_or(0, |ordinal| ordinal.0)
            )
        })?);
    }
    registry
        .project_current_source_ids(project)
        .map_err(|source| format!("worker cannot project source {source:?}"))?;
    bytecode.bound_functions = registry.functions.clone();
    bytecode.function_registry = registry.clone();
    Ok(runtime)
}
