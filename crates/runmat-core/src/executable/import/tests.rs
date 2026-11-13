use super::*;
use crate::{ExecutableSource, InvocationControl, ProcedureInvocation, RunMatSession};
use futures::executor::block_on;
use runmat_value::Value;

fn original() -> ExecutableUnit {
    let directory = tempfile::tempdir().unwrap();
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let source = |name: &str, text: &str| {
        ExecutableSource::frozen(
            "portable-owner",
            name,
            directory.path().join(name).to_string_lossy(),
            text,
        )
    };
    block_on(session.compile_executable_unit(source("unrelated.m", "unrelated = 1;"), None))
        .unwrap();
    block_on(session.compile_executable_unit(
        source(
            "outer.m",
            concat!(
                "function [y,z] = outer(x)\n",
                "g = @(t) helper(t) + x;\ny = g(3);\nz = helper(x);\nend\n",
                "function v = helper(t)\nv = 2 * t;\nend\n",
            ),
        ),
        None,
    ))
    .unwrap()
}

fn change(
    envelope: &mut ExecutableUnitEnvelope,
    kind: Kind,
    modify: impl FnOnce(&mut serde_json::Value),
) {
    let payload = envelope
        .payloads
        .iter_mut()
        .find(|payload| payload.kind == kind)
        .unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&payload.bytes).unwrap();
    modify(&mut value);
    payload.bytes = serde_json::to_vec(&value).unwrap();
    let descriptor = envelope
        .manifest
        .components
        .iter_mut()
        .find(|descriptor| descriptor.kind == kind)
        .unwrap();
    descriptor.digest = runmat_execution::Digest::sha256(&payload.bytes);
    descriptor.encoded_length = payload.bytes.len() as u64;
    // Payload hashes are refreshed, but structural admission may now reject
    // malformed source authority before the deeper product consistency check.
    // Every caller still requires import itself to reject the altered product.
}

#[test]
fn restores_original_product_and_executes_captures_helpers_and_output_ordinals() {
    let original = original();
    let source = original
        .bytecode()
        .source_id
        .unwrap()
        .program()
        .expect("compiler source coordinate, not session authority");
    let primary = original
        .source_map()
        .entries()
        .iter()
        .find(|entry| entry.source_id == source)
        .expect("original source entry");
    assert_eq!(primary.owner_identity, "portable-owner");
    assert_eq!(primary.relative_path, "outer.m");
    let envelope = original.portable_envelope().unwrap();
    assert_ne!(
        envelope.manifest.identity.entrypoint_function.unwrap().0 as usize,
        original.native_function_id("outer").unwrap().0,
        "the fixture must distinguish portable ordinals from compiler coordinates"
    );
    let cold: ExecutableUnitEnvelope =
        serde_json::from_slice(&serde_json::to_vec(&envelope).unwrap()).unwrap();
    let imported = ExecutableUnit::from_portable_envelope(&cold).unwrap();
    assert_eq!(imported.portable_envelope().unwrap(), envelope);
    assert_eq!(imported.revision(), original.revision());
    assert_eq!(
        imported.source().owner_identity,
        original.source().owner_identity
    );
    assert_eq!(imported.source().relative_path, "outer.m");
    assert_eq!(imported.source().text, original.source().text);
    assert!(imported
        .source_map()
        .entries()
        .iter()
        .all(|entry| entry.full_path.is_none()));
    assert_eq!(imported.coverage_plan(), original.coverage_plan());
    assert_eq!(imported.analysis(), original.analysis());
    assert_eq!(imported.mir(), original.mir());
    assert_eq!(imported.vm_layout(), original.vm_layout());
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let mut invocation = ProcedureInvocation::function("outer", vec![Value::Num(10.0)]);
    invocation.requested_outputs = 2;
    let value =
        block_on(session.invoke_executable(&imported, invocation, &InvocationControl::default()))
            .unwrap();
    assert_eq!(value.as_slice(), &[Value::Num(16.0), Value::Num(20.0)]);
}

#[test]
fn cold_interop_round_trip_reconstitutes_only_required_capabilities() {
    use runmat_types::{CapabilityRequirement as Capability, CapabilitySet};

    let unit = original();
    let mut expected = unit.portable_envelope().unwrap().manifest.capabilities;
    assert!(!expected.0.contains(&Capability::Network));
    expected
        .0
        .extend([Capability::NativeCode, Capability::ForeignRuntime]);
    let interop = runmat_types::InteropManifest {
        adapters: vec![runmat_types::ForeignAdapterRequirement {
            adapter: runmat_types::ForeignAdapterId::new("cold-native-fixture").unwrap(),
            minimum_version: 1,
            capabilities: CapabilitySet(std::collections::BTreeSet::from([Capability::NativeCode])),
            execution_stack: runmat_types::ExecutionStackRequirement::Process,
            artifact_identities: vec![runmat_types::ForeignArtifactIdentity::new(
                "cold-native-fixture:v1",
            )
            .unwrap()],
        }],
        ..runmat_types::InteropManifest::empty()
    };
    let envelope = unit
        .portable_envelope_for_with_interop(None, interop)
        .unwrap();
    assert_eq!(envelope.manifest.capabilities, expected);
    let cold: ExecutableUnitEnvelope =
        serde_json::from_slice(&envelope.canonical_bytes().unwrap()).unwrap();
    let restored = ExecutableUnit::from_portable_envelope(&cold).unwrap();
    assert_eq!(restored.portable_envelope().unwrap(), cold);

    // Even a structurally valid extra capability must not acquire authority
    // merely because an incoming product claims it.
    let mut extra = cold;
    extra.manifest.capabilities.0.insert(Capability::Network);
    extra.validate().unwrap();
    assert!(ExecutableUnit::from_portable_envelope(&extra).is_err());
}

#[test]
fn rejects_missing_foreign_and_duplicated_source_authority_after_hash_refresh() {
    let envelope = original().portable_envelope().unwrap();
    for mutation in 0..6 {
        let mut altered = envelope.clone();
        match mutation {
            0 => change(&mut altered, Kind::Bytecode, |value| {
                value["source_id"] = serde_json::Value::Null
            }),
            1 => change(&mut altered, Kind::Bytecode, |value| {
                value["source_id"] = 0.into()
            }),
            2 => change(&mut altered, Kind::SourceMap, |value| {
                let repeated = value["entries"][0].clone();
                value["entries"].as_array_mut().unwrap().push(repeated);
            }),
            3 => altered.manifest.identity.root_package = "foreign-owner".into(),
            4 => change(&mut altered, Kind::SourceMap, |value| {
                for entry in value["entries"].as_array_mut().unwrap() {
                    entry["text"] =
                        format!("{}\n% altered source", entry["text"].as_str().unwrap()).into();
                }
            }),
            _ => {
                altered.manifest.identity.entrypoint_function =
                    Some(runmat_types::ProgramFunctionId(
                        altered.manifest.identity.entrypoint_function.unwrap().0 + 1,
                    ));
                altered.validate().unwrap();
            }
        }
        assert!(ExecutableUnit::from_portable_envelope(&altered).is_err());
    }
}

#[test]
fn rejects_duplicate_components_inconsistent_coverage_and_unsupported_revisions() {
    let envelope = original().portable_envelope().unwrap();
    for mutation in 0..4 {
        let mut altered = envelope.clone();
        match mutation {
            0 => {
                let registry: serde_json::Value = serde_json::from_slice(
                    &envelope.component(Kind::FunctionRegistry).unwrap().bytes,
                )
                .unwrap();
                change(&mut altered, Kind::Bytecode, |value| {
                    value["function_registry"] = registry
                });
            }
            1 => change(&mut altered, Kind::Bytecode, |value| {
                value["coverage_sites"] = serde_json::json!([[1234]])
            }),
            2 => change(&mut altered, Kind::Analysis, |value| {
                value["revision"]["schema_version"] = 65535.into()
            }),
            _ => {
                altered.manifest.revisions.bytecode_schema += 1;
                altered
                    .manifest
                    .components
                    .iter_mut()
                    .find(|descriptor| descriptor.kind == Kind::Bytecode)
                    .unwrap()
                    .schema_version += 1;
                altered.validate().unwrap();
            }
        }
        assert!(ExecutableUnit::from_portable_envelope(&altered).is_err());
    }
}

#[test]
fn preserves_opaque_optional_metadata_and_rejects_required_unknown_sections() {
    let mut envelope = original().portable_envelope().unwrap();
    envelope
        .manifest
        .optional_sections
        .push(runmat_execution::ExecutableOptionalSection::new(
            "compiler.extension",
            1,
            runmat_execution::SectionRequirement::Optional,
            b"original metadata".to_vec(),
        ));
    let imported = ExecutableUnit::from_portable_envelope(&envelope).unwrap();
    assert_eq!(imported.portable_envelope().unwrap(), envelope);
    envelope.manifest.optional_sections[0].requirement =
        runmat_execution::SectionRequirement::Required;
    envelope.validate().unwrap();
    assert!(ExecutableUnit::from_portable_envelope(&envelope).is_err());
}

#[test]
fn rejects_products_advertising_pre_external_deferred_schemas() {
    let envelope = original().portable_envelope().unwrap();
    for kind in [Kind::Mir, Kind::Bytecode, Kind::FunctionRegistry] {
        let mut legacy = envelope.clone();
        let revision = match kind {
            Kind::Mir => &mut legacy.manifest.revisions.mir_schema,
            Kind::Bytecode => &mut legacy.manifest.revisions.bytecode_schema,
            Kind::FunctionRegistry => &mut legacy.manifest.revisions.function_registry_schema,
            _ => unreachable!(),
        };
        *revision -= 1;
        legacy
            .manifest
            .components
            .iter_mut()
            .find(|descriptor| descriptor.kind == kind)
            .unwrap()
            .schema_version -= 1;
        legacy.validate().unwrap();
        assert!(
            ExecutableUnit::from_portable_envelope(&legacy).is_err(),
            "old {kind:?} schema must not silently acquire new deferred semantics"
        );
    }
}
