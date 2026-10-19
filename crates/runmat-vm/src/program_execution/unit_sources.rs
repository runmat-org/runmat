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
    records: Option<Vec<runmat_runtime::source_context::ProgramSourceRecord>>,
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
    let runtime = match records {
        Some(records) => runmat_runtime::source_context::bind_portable_source_map_v2_with_records(
            runtime, source_map, records,
        ),
        None => runmat_runtime::source_context::bind_portable_source_map_v2(runtime, source_map),
    }
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
    let authorized = authorized_private_aliases(source_map);
    for (key, function) in &registry.functions {
        if *key != function.function {
            return Err("worker rejected function registry key/body identity mismatch".into());
        }
        validate_function_component_with_aliases(source_map, revision, function, &authorized)?;
    }
    validate_private_partitions(registry.functions.values())
}

pub(super) fn validate_private_partitions<'a>(
    functions: impl Iterator<Item = &'a crate::FunctionBytecode>,
) -> Result<(), String> {
    let mut partition_by_alias = std::collections::HashMap::new();
    let mut partition_by_source = std::collections::HashMap::new();
    for function in functions {
        let partition = function
            .private_source_owner
            .as_ref()
            .map(|owner| owner.aliases());
        if let Some(source) = function.source_id.and_then(BytecodeSourceRef::program) {
            if let Some(previous) = partition_by_source.insert(source, partition) {
                if previous != partition {
                    return Err(
                        "worker rejected conflicting private owner partitions for one source"
                            .into(),
                    );
                }
            }
        }
        if let Some(aliases) = partition {
            for alias in aliases {
                if let Some(previous) = partition_by_alias.insert(alias, aliases) {
                    if previous != aliases {
                        return Err("worker rejected overlapping private owner partitions".into());
                    }
                }
            }
        }
    }
    Ok(())
}

pub(super) fn authorized_private_aliases(
    source_map: &ExecutableSourceMapV2,
) -> std::collections::HashSet<crate::PrivateSourceAlias> {
    source_map
        .entries
        .iter()
        .map(|entry| {
            crate::PrivateSourceAlias::from_source_path(&entry.owner_identity, &entry.relative_path)
        })
        .collect()
}

pub(super) fn validate_function_component_with_aliases(
    source_map: &ExecutableSourceMapV2,
    revision: &runmat_execution::ProgramRevision,
    function: &crate::FunctionBytecode,
    authorized: &std::collections::HashSet<crate::PrivateSourceAlias>,
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
    if let Some(owner) = &function.private_source_owner {
        let aliases = owner.aliases();
        let expected = function
            .source_id
            .and_then(BytecodeSourceRef::program)
            .and_then(|ordinal| source_map.entries.get(usize::try_from(ordinal.0).ok()?))
            .map(|entry| {
                crate::PrivateSourceAlias::from_source_path(
                    &entry.owner_identity,
                    &entry.relative_path,
                )
            });
        if expected
            .as_ref()
            .is_none_or(|expected| !aliases.contains(expected))
            || aliases.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err("worker rejected private owner outside its selected source".into());
        }
        if aliases.iter().any(|alias| !authorized.contains(alias)) {
            return Err("worker rejected private owner alias outside its source map".into());
        }
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
            private_source_owner: Some(crate::PrivateSourceDirectory::captured(
                None,
                vec![crate::PrivateSourceAlias::from_source_path(
                    "fixture", "helper.m",
                )],
            )),
            binding_origin: Some(origin),
            ..Default::default()
        };
        let registry = FunctionRegistry::new(HashMap::from([(FunctionId(2), body)]));
        validate_function_components(&map, &revision, &registry).expect("matched components");

        let mut wrong_private_alias = registry.clone();
        wrong_private_alias
            .functions
            .get_mut(&FunctionId(2))
            .unwrap()
            .private_source_owner = Some(crate::PrivateSourceDirectory::captured(
            None,
            vec![crate::PrivateSourceAlias::from_source_path(
                "other", "helper.m",
            )],
        ));
        assert!(
            validate_function_components(&map, &revision, &wrong_private_alias)
                .unwrap_err()
                .contains("private owner")
        );

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

    #[test]
    fn overlapping_in_map_private_aliases_are_rejected_before_source_leases() {
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
        let text = "function y = f; y = 1; end";
        let record = |id, relative: &str| runmat_execution::ExecutableSourceRecordV2 {
            source_id: ProgramSourceId(id),
            owner_identity: "fixture".into(),
            relative_path: relative.into(),
            content_revision: runmat_execution::Digest::sha256(text.as_bytes()),
            display_name: relative.into(),
            text: text.into(),
        };
        let map = ExecutableSourceMapV2::new(vec![
            record(0, "left/entry.m"),
            record(1, "right/private/helper.m"),
        ])
        .expect("map");
        let left = crate::PrivateSourceAlias::from_source_path("fixture", "left/entry.m");
        let right =
            crate::PrivateSourceAlias::from_source_path("fixture", "right/private/helper.m");
        let body = |id, path: &str, aliases| FunctionBytecode {
            function: FunctionId(id),
            portable_function: ProgramFunctionId(id as u32),
            source_id: Some(BytecodeSourceRef::Program(ProgramSourceId((id - 1) as u32))),
            source_file: Some(path.into()),
            binding_origin: Some(
                runmat_execution::FunctionBindingOrigin::from_revision(
                    &revision,
                    ProgramFunctionId(id as u32),
                )
                .expect("origin"),
            ),
            private_source_owner: Some(crate::PrivateSourceDirectory::captured(None, aliases)),
            ..Default::default()
        };
        let registry = FunctionRegistry::new(HashMap::from([
            (
                FunctionId(1),
                body(1, "left/entry.m", vec![left, right.clone()]),
            ),
            (
                FunctionId(2),
                body(2, "right/private/helper.m", vec![right]),
            ),
        ]));
        assert!(validate_function_components(&map, &revision, &registry)
            .unwrap_err()
            .contains("overlapping private owner partitions"));
    }
}
