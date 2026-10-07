use crate::{Bytecode, BytecodeSourceRef, FunctionRegistry};
use runmat_execution::{ExecutableSourceMapV2, ExecutableUnitEnvelope};

/// Import is a materialization boundary: validate all ordinal references,
/// issue fresh session leases, then project the executable body and selected
/// registry together. No serialized ordinal is ever used as a runtime ID.
pub(super) fn bind_unit_sources(
    envelope: &ExecutableUnitEnvelope,
    source_map: &ExecutableSourceMapV2,
    bytecode: &mut Bytecode,
    registry: &mut FunctionRegistry,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<runmat_runtime::context::RuntimeContext, String> {
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
    validate_function_components(source_map, &envelope.manifest.identity.program, registry)?;
    let runtime = runmat_runtime::source_context::bind_portable_source_map_v2(runtime, source_map)
        .map_err(|error| format!("worker cannot bind executable source map: {error}"))?;
    let project = |source: BytecodeSourceRef| {
        source
            .program()
            .and_then(|ordinal| runtime.program_source_ref(ordinal))
            .map(BytecodeSourceRef::Runtime)
    };
    let mut projected = bytecode.clone();
    projected.function_registry = registry.clone();
    projected.bound_functions = registry.functions.clone();
    projected.project_source_ids(project)?;
    *registry = projected.function_registry.clone();
    *bytecode = projected;
    Ok(runtime)
}

fn validate_function_components(
    source_map: &ExecutableSourceMapV2,
    revision: &runmat_execution::ProgramRevision,
    registry: &FunctionRegistry,
) -> Result<(), String> {
    for (key, function) in &registry.functions {
        if *key != function.function {
            return Err("worker rejected function registry key/body identity mismatch".into());
        }
        validate_function_component(source_map, revision, function)?;
    }
    Ok(())
}

pub(super) fn validate_function_component(
    source_map: &ExecutableSourceMapV2,
    revision: &runmat_execution::ProgramRevision,
    function: &crate::FunctionBytecode,
) -> Result<(), String> {
    let expected_origin = runmat_execution::FunctionBindingOrigin::from_revision(
        revision,
        function.portable_function,
    )
    .map_err(|error| format!("worker rejected invalid function origin: {error}"))?;
    if function.binding_origin != Some(expected_origin) {
        return Err(
            "worker rejected function binding origin outside its executable revision".into(),
        );
    }
    let expected_file = function
        .source_id
        .and_then(BytecodeSourceRef::program)
        .and_then(|ordinal| source_map.entries.get(usize::try_from(ordinal.0).ok()?))
        .map(|source| source.relative_path.as_str());
    if function.source_file.as_deref() != expected_file {
        return Err("worker rejected function source path outside its validated source map".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FunctionBytecode;
    use runmat_hir::FunctionId;
    use runmat_types::{ProgramFunctionId, ProgramSourceId};
    use std::collections::HashMap;

    #[test]
    fn registry_path_and_origin_must_match_selected_map_before_lease_issuance() {
        let revision = runmat_execution::ProgramRevision::new(
            runmat_execution::Digest::sha256(b"graph"),
            runmat_execution::Digest::sha256(b"source"),
            runmat_execution::ProgramEnvironment {
                semantic_schema: 1,
                compiler_schema: 1,
                runtime_fingerprint: runmat_execution::Digest::sha256(b"runtime"),
                catalog_fingerprint: runmat_execution::Digest::sha256(b"catalog"),
                compatibility_mode: runmat_execution::LanguageCompatibilityMode::RunMat,
            },
        )
        .expect("revision");
        let map = ExecutableSourceMapV2::new(vec![runmat_execution::ExecutableSourceRecordV2 {
            source_id: ProgramSourceId(0),
            owner_identity: "fixture".into(),
            relative_path: "helper.m".into(),
            content_revision: runmat_execution::Digest::sha256(b"function y=helper; y=1; end"),
            display_name: "helper.m".into(),
            text: "function y=helper; y=1; end".into(),
        }])
        .expect("map");
        let origin =
            runmat_execution::FunctionBindingOrigin::from_revision(&revision, ProgramFunctionId(2))
                .expect("origin");
        let body = FunctionBytecode {
            function: FunctionId(2),
            portable_function: ProgramFunctionId(2),
            source_id: Some(BytecodeSourceRef::Program(ProgramSourceId(0))),
            source_file: Some("helper.m".into()),
            binding_origin: Some(origin),
            ..Default::default()
        };
        let registry = FunctionRegistry::new(HashMap::from([(FunctionId(2), body)]));
        validate_function_components(&map, &revision, &registry).expect("matched components");

        let mut wrong_path = registry.clone();
        wrong_path
            .functions
            .get_mut(&FunctionId(2))
            .unwrap()
            .source_file = Some("/another/checkout/helper.m".into());
        assert!(validate_function_components(&map, &revision, &wrong_path)
            .unwrap_err()
            .contains("source path"));

        let mut wrong_origin = registry;
        wrong_origin
            .functions
            .get_mut(&FunctionId(2))
            .unwrap()
            .binding_origin = Some(
            runmat_execution::FunctionBindingOrigin::from_revision(&revision, ProgramFunctionId(3))
                .expect("different function origin"),
        );
        assert!(validate_function_components(&map, &revision, &wrong_origin)
            .unwrap_err()
            .contains("binding origin"));

        let mut wrong_key = wrong_origin;
        let mut body = wrong_key.functions.remove(&FunctionId(2)).unwrap();
        body.binding_origin = Some(origin);
        wrong_key.functions.insert(FunctionId(3), body);
        assert!(validate_function_components(&map, &revision, &wrong_key)
            .unwrap_err()
            .contains("key/body"));
    }
}
