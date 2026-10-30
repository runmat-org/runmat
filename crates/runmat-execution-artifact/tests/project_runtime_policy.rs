mod support;
#[path = "support/executable_unit.rs"]
mod executable_unit_support;
use runmat_execution_artifact::{
    archive::{read_bundle, write_bundle, ArchiveLimits},
    capture_typed_project_runtime_policy, decode_typed_project_runtime_policy,
    validate_project_runtime_policy_descriptor, validate_project_runtime_policy_object,
    ExecutableForm, ExecutionBundle, ExecutionBundleBuilder, LogicalObject, ObjectNamespace,
    EXECUTION_BUNDLE_SCHEMA_VERSION, MAX_PROJECT_RUNTIME_POLICY_BYTES,
};
use serde::{Deserialize, Serialize};

// Artifact tests own transport semantics only; Config tests own the complete execution policy.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Policy {
    schema_version: u16,
    threads: u32,
}
fn policy(threads: u32) -> LogicalObject {
    capture_typed_project_runtime_policy(&Policy {
        schema_version: 1,
        threads,
    })
    .unwrap()
}
fn bundle() -> ExecutionBundle {
    let (_directory, project, revision) = support::frozen_project();
    ExecutionBundleBuilder::native(&project, revision.clone())
        .unwrap()
        .with_materialized_program(
            support::recipe(revision),
            ExecutableForm::InterpreterBytecodeV2,
            b"canonical-bytecode".to_vec(),
        )
        .build()
        .unwrap()
}

#[test]
fn project_policy_is_canonical_bounded_versioned_and_typed_lossless() {
    let object = policy(2);
    assert_eq!(ObjectNamespace::ProjectRuntimePolicy as u8, 10);
    assert_eq!(
        object.descriptor.namespace,
        ObjectNamespace::ProjectRuntimePolicy
    );
    assert_eq!(object.bytes, br#"{"schema_version":1,"threads":2}"#);
    assert_eq!(
        decode_typed_project_runtime_policy::<Policy>(&object).unwrap(),
        Policy {
            schema_version: 1,
            threads: 2
        }
    );
    let mut descriptor = object.descriptor.clone();
    descriptor.encoded_length = MAX_PROJECT_RUNTIME_POLICY_BYTES + 1;
    assert!(validate_project_runtime_policy_descriptor(&descriptor).is_err());
    descriptor = object.descriptor.clone();
    descriptor.media_type = "application/json".into();
    assert!(validate_project_runtime_policy_descriptor(&descriptor).is_err());
    let wrong_version = capture_typed_project_runtime_policy(&Policy {
        schema_version: 2,
        threads: 2,
    });
    assert!(wrong_version.is_err());
    let noncanonical_bytes = b"{\"threads\":2,\"schema_version\":1}".to_vec();
    let noncanonical = LogicalObject::new(
        ObjectNamespace::ProjectRuntimePolicy,
        format!(
            "project-runtime-policy/{}",
            runmat_execution::Digest::sha256(&noncanonical_bytes)
        ),
        object.descriptor.media_type.clone(),
        noncanonical_bytes,
    )
    .unwrap();
    assert!(validate_project_runtime_policy_object(&noncanonical).is_err());
    assert!(capture_typed_project_runtime_policy(&serde_json::json!({ "schema_version":1, "payload":"x".repeat(MAX_PROJECT_RUNTIME_POLICY_BYTES as usize) })).is_err());
}

#[test]
fn project_policy_never_carries_collector_settings_or_ignored_typed_fields() {
    for key in [
        "logging",
        "telemetry",
        "headers",
        "otel",
        "openTelemetryHeaders",
    ] {
        let mut value = serde_json::json!({ "schema_version": 1 });
        value[key] = serde_json::json!({ "authorization": "not-a-permitted-project-secret" });
        assert!(
            capture_typed_project_runtime_policy(&value).is_err(),
            "{key}"
        );
    }
    let extra = capture_typed_project_runtime_policy(
        &serde_json::json!({ "schema_version":1, "threads":2, "extra":true }),
    )
    .unwrap();
    assert!(decode_typed_project_runtime_policy::<Policy>(&extra).is_err());
    #[derive(Serialize, Deserialize)]
    struct Defaulted {
        schema_version: u16,
        #[serde(default)]
        threads: u32,
    }
    let missing =
        capture_typed_project_runtime_policy(&serde_json::json!({ "schema_version":1 })).unwrap();
    assert!(
        decode_typed_project_runtime_policy::<Defaulted>(&missing).is_err(),
        "typed defaults must not silently change captured authority"
    );
}

#[test]
fn policy_descriptor_changes_identity_and_survives_exact_archive_inventory() {
    let base = bundle();
    assert_eq!(base.manifest.schema_version, 6);
    assert_eq!(EXECUTION_BUNDLE_SCHEMA_VERSION, 6);
    assert!(base.project_runtime_policy_object().unwrap().is_none());
    let first = base
        .clone()
        .with_project_runtime_policy_object(policy(1))
        .unwrap();
    let second = base.with_project_runtime_policy_object(policy(2)).unwrap();
    assert_ne!(first.identity().unwrap(), second.identity().unwrap());
    let mut archive = Vec::new();
    write_bundle(&first, &mut archive, ArchiveLimits::default()).unwrap();
    let decoded = read_bundle(archive.as_slice(), ArchiveLimits::default()).unwrap();
    assert_eq!(decoded, first);
    assert_eq!(
        decoded.project_runtime_policy_object().unwrap(),
        Some(&policy(1))
    );
    assert!(first
        .clone()
        .with_project_runtime_policy_object(policy(2))
        .is_err());
    let mut missing = first.clone();
    missing
        .objects
        .retain(|object| object.descriptor.namespace != ObjectNamespace::ProjectRuntimePolicy);
    assert!(missing.validate().is_err());
    let mut unreferenced = first.clone();
    unreferenced.manifest.project_runtime_policy = None;
    assert!(unreferenced.validate().is_err());
    let mut substituted = first.clone();
    *substituted
        .objects
        .iter_mut()
        .find(|object| object.descriptor.namespace == ObjectNamespace::ProjectRuntimePolicy)
        .unwrap() = policy(2);
    substituted
        .objects
        .sort_by(|left, right| left.descriptor.cmp(&right.descriptor));
    assert!(substituted.validate().is_err());
    let mut old = first.clone();
    old.manifest.schema_version = 5;
    assert!(old.validate().is_err());
    let marker = b"runmat-execution-bundle-manifest-v6\0";
    let index = archive
        .windows(marker.len())
        .position(|bytes| bytes == marker)
        .unwrap();
    archive[index + marker.len() - 2] = b'5';
    assert!(
        read_bundle(archive.as_slice(), ArchiveLimits::default()).is_err(),
        "schema5 archives cannot alias schema6 policy authority"
    );
}

#[test]
fn builder_accepts_exactly_one_project_policy_object() {
    let (_directory, project, revision) = support::frozen_project();
    let builder = ExecutionBundleBuilder::native(&project, revision.clone())
        .unwrap()
        .with_recipe(support::recipe(revision))
        .with_project_runtime_policy_object(policy(4))
        .unwrap();
    let built = builder.build().unwrap();
    assert_eq!(
        built.project_runtime_policy_object().unwrap(),
        Some(&policy(4))
    );
    let builder = ExecutionBundleBuilder::native(&project, support::revision_for(&project))
        .unwrap()
        .with_project_runtime_policy_object(policy(4))
        .unwrap();
    assert!(builder
        .with_project_runtime_policy_object(policy(8))
        .is_err());
}

#[test]
fn selected_compiled_bundle_retains_parent_project_policy_and_private_policy_materialization() {
    use runmat_execution::{
        Digest, ExecutableComponentDescriptor, ExecutableComponentKind, ExecutableComponentPayload,
        ExecutableSourceMapV2, ExecutableSourceRecordV2, ExecutableUnitEnvelope,
    };
    let (_directory, project, revision) = support::frozen_project();
    let mut unit = ExecutableUnitEnvelope::from_canonical_bytes(&executable_unit_support::bytes(
        revision.clone(),
    ))
    .unwrap();
    // Selection decodes source-map schema2: use the actual frozen source, not this
    // shared opaque-component fixture's schema1 placeholder bytes.
    let package = project.sources.packages.values().next().unwrap();
    let source = package
        .sources
        .iter()
        .find(|source| source.qualified_name == "main")
        .unwrap();
    let text = std::fs::read_to_string(project.access_paths.get(&source.id).unwrap()).unwrap();
    let content_revision = Digest::sha256(text.as_bytes());
    assert_eq!(content_revision.bytes(), source.id.content_digest.bytes());
    let map = ExecutableSourceMapV2::new(vec![ExecutableSourceRecordV2 {
        source_id: runmat_types::ProgramSourceId(0),
        owner_identity: package.package_instance.to_string(),
        relative_path: source.id.relative_path.to_string(),
        content_revision,
        display_name: source.id.relative_path.to_string(),
        text,
    }])
    .unwrap();
    let payload = ExecutableComponentPayload::new(
        ExecutableComponentKind::SourceMap,
        serde_json::to_vec(&map).unwrap(),
    )
    .unwrap();
    let descriptor = ExecutableComponentDescriptor::from_payload(
        ExecutableComponentKind::SourceMap,
        2,
        &payload.bytes,
    )
    .unwrap();
    *unit
        .payloads
        .iter_mut()
        .find(|part| part.kind == ExecutableComponentKind::SourceMap)
        .unwrap() = payload;
    *unit
        .manifest
        .components
        .iter_mut()
        .find(|part| part.kind == ExecutableComponentKind::SourceMap)
        .unwrap() = descriptor;
    unit.manifest.revisions.source_map_schema = 2;
    unit.manifest.identity.source_digest = content_revision;
    let bytes = unit.canonical_bytes().unwrap();
    let foreign = runmat_execution_artifact::foreign_policy_object(
        br#"{"fixture":"selected-runtime-policy","schema_version":1}"#.to_vec(),
    )
    .unwrap();
    let mut recipe = executable_unit_support::recipe(support::recipe(revision.clone()));
    recipe.foreign_policy = Some(foreign.descriptor.clone());
    let parent = ExecutionBundleBuilder::native(&project, revision)
        .unwrap()
        .with_compiled_package_closure()
        .with_project_runtime_policy_object(policy(2))
        .unwrap()
        .with_foreign_policy_object(foreign)
        .unwrap()
        .with_materialized_program(recipe, ExecutableForm::ExecutableUnitV4, bytes.clone())
        .build()
        .unwrap();
    assert!(parent.manifest.foreign_artifacts.is_empty());
    assert!(
        parent.requires_materialization(),
        "foreign policy remains meaningful without source/foreign files"
    );
    let selected = parent
        .selected_unit_bundle_for(&parent.manifest.recipes[0], &bytes)
        .unwrap();
    assert_eq!(
        selected.project_runtime_policy_object().unwrap(),
        parent.project_runtime_policy_object().unwrap()
    );
    let mut archive = Vec::new();
    write_bundle(&selected, &mut archive, ArchiveLimits::default()).unwrap();
    assert_eq!(
        read_bundle(archive.as_slice(), ArchiveLimits::default()).unwrap(),
        selected
    );
}
