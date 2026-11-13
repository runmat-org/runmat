use crate::{Bytecode, BytecodeSourceRef};
use runmat_execution::ExecutableSourceMapV2;
use runmat_execution_artifact::InterpreterRegionSourcesV1;
use runmat_runtime::context::RuntimeContext;

#[cfg(test)]
const PREFIX: &[u8] = b"runmat-region-sources-v1\n";
#[cfg(test)]
const SCHEMA: u16 = 1;

pub(super) fn capture(bytecode: &Bytecode, runtime: &RuntimeContext) -> Result<Vec<u8>, String> {
    let source_map = runtime
        .portable_program_source_map()
        .ok_or_else(|| "isolated region has no compiler-bound portable source map".to_string())?;
    source_map.validate().map_err(|error| error.to_string())?;
    let revision = runtime
        .program_revision()
        .ok_or_else(|| "isolated region has no compiler-bound program revision".to_string())?;
    let project = |source: BytecodeSourceRef| match source {
        BytecodeSourceRef::Program(ordinal)
            if usize::try_from(ordinal.0)
                .ok()
                .is_some_and(|index| index < source_map.entries.len()) =>
        {
            Some(BytecodeSourceRef::Program(ordinal))
        }
        BytecodeSourceRef::Runtime(reference) => runtime
            .portable_ordinal_for_source(reference)
            .map(BytecodeSourceRef::Program),
        _ => None,
    };
    let mut portable = bytecode.clone();
    portable.project_source_ids(project)?;
    for body in portable
        .bound_functions
        .values_mut()
        .chain(portable.function_registry.functions.values_mut())
    {
        body.binding_origin = Some(
            runmat_execution::FunctionBindingOrigin::from_revision(
                revision,
                body.portable_function,
            )
            .map_err(|error| error.to_string())?,
        );
        body.source_file = body
            .source_id
            .and_then(BytecodeSourceRef::program)
            .and_then(|ordinal| source_map.entries.get(usize::try_from(ordinal.0).ok()?))
            .map(|entry| entry.relative_path.clone());
    }
    portable
        .validate_portable_source_origins()
        .map_err(str::to_string)?;
    encode_portable_region_sources_v1(&portable, source_map, revision)
}

/// Encode a compiler-issued portable bytecode/map pair. The worker validates
/// both again before issuing any local source identity.
pub fn encode_portable_region_sources_v1(
    portable: &Bytecode,
    source_map: &ExecutableSourceMapV2,
    revision: &runmat_execution::ProgramRevision,
) -> Result<Vec<u8>, String> {
    source_map.validate().map_err(|error| error.to_string())?;
    portable
        .validate_portable_source_origins()
        .map_err(str::to_string)?;
    let authorized = super::unit_sources::authorized_private_aliases(source_map);
    for (key, body) in portable
        .bound_functions
        .iter()
        .chain(portable.function_registry.functions.iter())
    {
        if *key != body.function {
            return Err("portable region function key/body identity mismatch".into());
        }
        super::unit_sources::validate_function_component_with_aliases(
            source_map,
            revision,
            body,
            &authorized,
        )?;
    }
    super::unit_sources::validate_private_partitions(
        portable
            .bound_functions
            .values()
            .chain(portable.function_registry.functions.values()),
    )?;
    let script =
        crate::encode_interpreter_script_v2(portable).map_err(|error| error.to_string())?;
    InterpreterRegionSourcesV1::new(revision.clone(), source_map.clone(), script)
        .and_then(|payload| payload.encode())
        .map_err(|error| error.to_string())
}

pub(super) fn bind(
    bytes: &[u8],
    runtime: RuntimeContext,
    expected_revision: &runmat_execution::ProgramRevision,
) -> Result<(Bytecode, RuntimeContext), String> {
    let payload = InterpreterRegionSourcesV1::decode(bytes, expected_revision)
        .map_err(|error| error.to_string())?;
    let mut bytecode = crate::decode_interpreter_script_v2(payload.script())
        .map_err(|error| format!("worker rejected region bytecode: {error}"))?;
    bytecode
        .validate_portable_source_origins()
        .map_err(str::to_string)?;
    let in_map = |source: BytecodeSourceRef| {
        source
            .program()
            .and_then(|ordinal| usize::try_from(ordinal.0).ok())
            .is_some_and(|index| index < payload.source_map().entries.len())
    };
    if bytecode.source_id.is_some_and(|source| !in_map(source))
        || bytecode
            .bound_functions
            .values()
            .any(|body| body.source_id.is_some_and(|source| !in_map(source)))
        || bytecode
            .function_registry
            .functions
            .values()
            .any(|body| body.source_id.is_some_and(|source| !in_map(source)))
    {
        return Err("worker rejected region source ordinal outside its map".into());
    }
    let authorized = super::unit_sources::authorized_private_aliases(payload.source_map());
    for (key, body) in bytecode
        .bound_functions
        .iter()
        .chain(bytecode.function_registry.functions.iter())
    {
        if *key != body.function {
            return Err("worker rejected region function key/body identity mismatch".into());
        }
        super::unit_sources::validate_function_component_with_aliases(
            payload.source_map(),
            expected_revision,
            body,
            &authorized,
        )?;
    }
    super::unit_sources::validate_private_partitions(
        bytecode
            .bound_functions
            .values()
            .chain(bytecode.function_registry.functions.values()),
    )?;
    let runtime =
        runmat_runtime::source_context::bind_portable_source_map_v2(runtime, payload.source_map())
            .map_err(|error| error.to_string())?;
    let project = |source: BytecodeSourceRef| {
        source
            .program()
            .and_then(|ordinal| runtime.program_source_ref(ordinal))
            .map(BytecodeSourceRef::Runtime)
    };
    bytecode.project_source_ids(project)?;
    Ok((bytecode, runtime))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instr;
    use runmat_execution::{Digest, ExecutableSourceRecordV2};
    use runmat_types::ProgramSourceId;
    use std::rc::Rc;

    fn context() -> RuntimeContext {
        RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_program_revision(Some(revision()))
    }

    fn revision() -> runmat_execution::ProgramRevision {
        runmat_execution::ProgramRevision::new(
            Digest::sha256(b"graph"),
            Digest::sha256(b"source"),
            runmat_execution::ProgramEnvironment {
                semantic_schema: 1,
                compiler_schema: 1,
                runtime_fingerprint: Digest::sha256(b"runtime"),
                catalog_fingerprint: Digest::sha256(b"catalog"),
                compatibility_mode: runmat_execution::LanguageCompatibilityMode::RunMat,
            },
        )
        .expect("revision")
    }

    fn map() -> ExecutableSourceMapV2 {
        ExecutableSourceMapV2::new(vec![ExecutableSourceRecordV2 {
            source_id: ProgramSourceId(0),
            owner_identity: "fixture".into(),
            relative_path: "main.m".into(),
            content_revision: Digest::sha256(b"answer = 7;"),
            display_name: "main.m".into(),
            text: "answer = 7;".into(),
        }])
        .expect("map")
    }

    #[test]
    fn region_payload_projects_fresh_worker_sources_and_rejects_stale_schema() {
        let captured =
            runmat_runtime::source_context::bind_portable_source_map_v2(context(), &map())
                .expect("capture source");
        let mut bytecode =
            Bytecode::with_instructions(vec![Instr::LoadConst(7.0), Instr::Return], 0);
        bytecode.source_id = Some(BytecodeSourceRef::Runtime(
            captured.program_source_ref(ProgramSourceId(0)).unwrap(),
        ));
        let payload = capture(&bytecode, &captured).expect("source-bearing region artifact");
        let (first, first_runtime) = bind(&payload, context(), &revision()).expect("first worker");
        let (second, second_runtime) =
            bind(&payload, context(), &revision()).expect("second worker");
        assert_ne!(first.source_id, second.source_id);
        assert_eq!(
            first.source_id.unwrap().resolve_in(&first_runtime),
            second.source_id.unwrap().resolve_in(&second_runtime),
            "numeric IDs may collide, but their issuer authority must not"
        );
        assert_eq!(first.source_id.unwrap().resolve_in(&second_runtime), None);
        let source_id = first.source_id.unwrap().resolve_in(&first_runtime).unwrap();
        assert_eq!(
            runmat_runtime::source_context::source_info_in_context(&first_runtime, source_id)
                .unwrap()
                .text
                .as_ref(),
            "answer = 7;"
        );

        let mut stale: serde_json::Value =
            serde_json::from_slice(payload.strip_prefix(PREFIX).unwrap())
                .expect("decoded versioned payload");
        stale["schema_version"] = serde_json::json!(SCHEMA + 1);
        let mut bytes = PREFIX.to_vec();
        bytes.extend(serde_json::to_vec(&stale).unwrap());
        let unbound = context();
        assert!(bind(&bytes, unbound.clone(), &revision())
            .unwrap_err()
            .contains("schema"));
        let next = runmat_runtime::source_context::lease_source_context_in_context(
            &unbound,
            "first.m",
            None,
            "first = 1;",
        )
        .expect("rejection issued no source");
        assert_eq!(next.source_id().0, 0);
    }

    #[test]
    fn forged_text_and_foreign_runtime_source_fail_before_worker_effects() {
        let captured =
            runmat_runtime::source_context::bind_portable_source_map_v2(context(), &map())
                .expect("capture source");
        let mut bytecode = Bytecode::with_instructions(vec![Instr::Return], 0);
        bytecode.source_id = Some(BytecodeSourceRef::Runtime(
            captured.program_source_ref(ProgramSourceId(0)).unwrap(),
        ));
        let payload = capture(&bytecode, &captured).unwrap();
        let mut forged: serde_json::Value =
            serde_json::from_slice(payload.strip_prefix(PREFIX).unwrap()).unwrap();
        forged["source_map"]["entries"][0]["text"] = serde_json::json!("answer = 9;");
        let mut bytes = PREFIX.to_vec();
        bytes.extend(serde_json::to_vec(&forged).unwrap());
        let unbound = context();
        assert!(bind(&bytes, unbound.clone(), &revision()).is_err());
        let next = runmat_runtime::source_context::lease_source_context_in_context(
            &unbound,
            "first.m",
            None,
            "first = 1;",
        )
        .unwrap();
        assert_eq!(next.source_id().0, 0);

        let foreign = context();
        let lease = runmat_runtime::source_context::lease_source_context_in_context(
            &foreign,
            "other.m",
            None,
            "other = 1;",
        )
        .unwrap();
        bytecode.source_id = Some(BytecodeSourceRef::Runtime(lease.source_ref()));
        let error = capture(&bytecode, &captured).unwrap_err();
        assert!(
            error.contains("entrypoint source") && error.contains("is absent"),
            "foreign source must fail projection before worker effects: {error}"
        );
    }
}
