#[path = "support/executable_unit.rs"]
mod executable_unit_support;
mod support;

use runmat_execution_artifact::{
    archive::{read_bundle, write_bundle, ArchiveLimits},
    foreign_policy_object, ExecutableForm, ExecutionBundleBuilder, ForeignArtifactClosure,
    LogicalObject, ObjectNamespace, PreparedForeignParentView, SelectedPreparedForeignMaterial,
};

#[test]
fn identical_projects_have_checkout_independent_bundle_identity_and_archive() {
    let (_first_temp, first, first_revision) = support::frozen_project();
    let (_second_temp, second, second_revision) = support::frozen_project();
    assert_eq!(first_revision, second_revision);

    let first = ExecutionBundleBuilder::native(&first, first_revision.clone())
        .unwrap()
        .with_materialized_program(
            support::recipe(first_revision),
            ExecutableForm::InterpreterBytecodeV2,
            b"canonical-bytecode".to_vec(),
        )
        .build()
        .unwrap();
    let second = ExecutionBundleBuilder::native(&second, second_revision.clone())
        .unwrap()
        .with_materialized_program(
            support::recipe(second_revision),
            ExecutableForm::InterpreterBytecodeV2,
            b"canonical-bytecode".to_vec(),
        )
        .build()
        .unwrap();

    assert_eq!(first.identity().unwrap(), second.identity().unwrap());
    let mut first_archive = Vec::new();
    write_bundle(&first, &mut first_archive, ArchiveLimits::default()).unwrap();
    let mut second_archive = Vec::new();
    write_bundle(&second, &mut second_archive, ArchiveLimits::default()).unwrap();
    assert_eq!(first_archive, second_archive);
    let archive_text = String::from_utf8_lossy(&first_archive);
    assert!(!archive_text.contains("/private/"));
    assert!(!archive_text.contains("/Users/"));

    let decoded = read_bundle(first_archive.as_slice(), ArchiveLimits::default()).unwrap();
    assert_eq!(decoded, first);
}

#[test]
fn source_change_after_freeze_is_rejected() {
    let (_temp, project, revision) = support::frozen_project();
    let source_path = project.access_paths.values().next().unwrap();
    std::fs::write(source_path, "changed after freeze").unwrap();
    let error = ExecutionBundleBuilder::native(&project, revision)
        .unwrap()
        .with_recipe(support::recipe(support::revision_for(&project)))
        .build()
        .unwrap_err();
    assert!(error.to_string().contains("changed after project freeze"));
}

#[test]
fn complete_executable_unit_survives_package_archive_round_trip() {
    let (_temp, project, revision) = support::frozen_project();
    let bytes = executable_unit_support::bytes(revision.clone());
    let policy =
        foreign_policy_object(br#"{"fixture":"selected-unit","schema_version":1}"#.to_vec())
            .unwrap();
    let mut recipe = executable_unit_support::recipe(support::recipe(revision.clone()));
    recipe.foreign_policy = Some(policy.descriptor.clone());
    let bundle = ExecutionBundleBuilder::native(&project, revision.clone())
        .unwrap()
        .with_foreign_policy_object(policy)
        .unwrap()
        .with_materialized_program(recipe, ExecutableForm::ExecutableUnitV4, bytes.clone())
        .build()
        .unwrap();
    let mut archive = Vec::new();
    write_bundle(&bundle, &mut archive, ArchiveLimits::default()).unwrap();
    let decoded = read_bundle(archive.as_slice(), ArchiveLimits::default()).unwrap();
    assert_eq!(decoded, bundle);
    assert_eq!(decoded.manifest.artifacts[0].executable_bytes, bytes);
    let envelope = decoded.manifest.artifacts[0]
        .executable_unit()
        .unwrap()
        .unwrap();
    assert_eq!(envelope.manifest.regions.len(), 2);
    assert_eq!(envelope.manifest.interop.foreign_types.len(), 1);
    assert_eq!(envelope.manifest.parallel.parfor_regions.len(), 1);
    assert_eq!(envelope.manifest.parallel.spmd_regions.len(), 1);
    assert_eq!(envelope.manifest.parallel.distributed_values.len(), 1);
    assert_eq!(envelope.manifest.parallel.collectives.len(), 1);
}

#[test]
fn accelerator_requirements_survive_package_archive_round_trip() {
    use runmat_execution::resource::{
        AcceleratorClass, AcceleratorFeature, AcceleratorProvider, AcceleratorProviderId,
        AcceleratorProviderVersion, AcceleratorRequest,
    };

    let (_temp, project, revision) = support::frozen_project();
    let mut recipe = support::recipe(revision.clone());
    recipe.accelerators = vec![AcceleratorRequest {
        class: AcceleratorClass::new("gpu").unwrap(),
        count: 1,
        minimum_allocation_bytes: 1 << 20,
        provider: Some(AcceleratorProvider {
            id: AcceleratorProviderId::new("runmat.cuda").unwrap(),
            version: AcceleratorProviderVersion::new("1.0.0").unwrap(),
            abi_fingerprint: runmat_execution::Digest::sha256(b"cuda-native-v1"),
        }),
        required_features: [AcceleratorFeature::Compute].into_iter().collect(),
    }];
    let bundle = ExecutionBundleBuilder::native(&project, revision)
        .unwrap()
        .with_materialized_program(
            recipe.clone(),
            ExecutableForm::InterpreterBytecodeV2,
            b"accelerated-bytecode".to_vec(),
        )
        .build()
        .unwrap();
    let mut archive = Vec::new();
    write_bundle(&bundle, &mut archive, ArchiveLimits::default()).unwrap();
    let decoded = read_bundle(archive.as_slice(), ArchiveLimits::default()).unwrap();

    assert_eq!(
        decoded.manifest.recipes[0].accelerators,
        recipe.accelerators
    );
    assert_eq!(decoded, bundle);
}

#[test]
fn meshing_host_workload_form_survives_package_archive_round_trip() {
    let (_temp, project, revision) = support::frozen_project();
    let mut recipe = support::recipe(revision.clone());
    recipe.entrypoint = "meshing_workload".into();
    recipe.execution_mode = "meshing".into();
    recipe.target = runmat_execution_artifact::ProgramTarget::portable("portable-meshing-host-v2");
    let bundle = ExecutionBundleBuilder::native(&project, revision)
        .unwrap()
        .with_compiled_package_closure()
        .with_materialized_program(
            recipe,
            ExecutableForm::MeshingWorkload,
            b"canonical-meshing-host-contract".to_vec(),
        )
        .build()
        .unwrap();
    let mut archive = Vec::new();
    write_bundle(&bundle, &mut archive, ArchiveLimits::default()).unwrap();
    let decoded = read_bundle(archive.as_slice(), ArchiveLimits::default()).unwrap();
    assert_eq!(decoded, bundle);
    assert_eq!(
        decoded.manifest.artifacts[0].form,
        ExecutableForm::MeshingWorkload
    );
}

#[test]
fn analysis_host_workload_form_survives_package_archive_round_trip() {
    let (_temp, project, revision) = support::frozen_project();
    let mut recipe = support::recipe(revision.clone());
    recipe.entrypoint = "analysis_workload".into();
    recipe.execution_mode = "analysis".into();
    recipe.target = runmat_execution_artifact::ProgramTarget::portable("portable-analysis-host-v1");
    let bundle = ExecutionBundleBuilder::native(&project, revision)
        .unwrap()
        .with_compiled_package_closure()
        .with_materialized_program(
            recipe,
            ExecutableForm::AnalysisWorkload,
            b"canonical-analysis-host-contract".to_vec(),
        )
        .build()
        .unwrap();
    let mut archive = Vec::new();
    write_bundle(&bundle, &mut archive, ArchiveLimits::default()).unwrap();
    let decoded = read_bundle(archive.as_slice(), ArchiveLimits::default()).unwrap();
    assert_eq!(decoded, bundle);
    assert_eq!(
        decoded.manifest.artifacts[0].form,
        ExecutableForm::AnalysisWorkload
    );
}

#[test]
fn geometry_host_workload_form_survives_package_archive_round_trip() {
    let (_temp, project, revision) = support::frozen_project();
    let mut recipe = support::recipe(revision.clone());
    recipe.entrypoint = "geometry_workload".into();
    recipe.execution_mode = "geometry-materialization".into();
    recipe.target = runmat_execution_artifact::ProgramTarget::portable("portable-geometry-host");
    let bundle = ExecutionBundleBuilder::native(&project, revision)
        .unwrap()
        .with_compiled_package_closure()
        .with_materialized_program(
            recipe,
            ExecutableForm::GeometryWorkload,
            b"canonical-geometry-host-contract".to_vec(),
        )
        .build()
        .unwrap();
    let mut archive = Vec::new();
    write_bundle(&bundle, &mut archive, ArchiveLimits::default()).unwrap();
    let decoded = read_bundle(archive.as_slice(), ArchiveLimits::default()).unwrap();
    assert_eq!(decoded, bundle);
    assert_eq!(
        decoded.manifest.artifacts[0].form,
        ExecutableForm::GeometryWorkload
    );
}

#[test]
fn compiled_package_closure_round_trips_without_source_or_project_payloads() {
    let (_temp, project, revision) = support::frozen_project();
    let bytes = executable_unit_support::bytes(revision.clone());
    let policy =
        foreign_policy_object(br#"{"fixture":"compiled-unit","schema_version":1}"#.to_vec())
            .unwrap();
    let mut recipe = executable_unit_support::recipe(support::recipe(revision.clone()));
    recipe.foreign_policy = Some(policy.descriptor.clone());
    let bundle = ExecutionBundleBuilder::native(&project, revision.clone())
        .unwrap()
        .with_compiled_package_closure()
        .with_foreign_policy_object(policy)
        .unwrap()
        .with_materialized_program(recipe, ExecutableForm::ExecutableUnitV4, bytes)
        .build()
        .unwrap();

    assert_eq!(bundle.objects.len(), 1); // only the authenticated policy, no source tree
    assert!(bundle.manifest.sources.is_empty());
    assert!(bundle.manifest.callables.is_empty());
    assert!(!bundle.requires_source_project());
    let runmat_execution_artifact::BundleCodeClosure::Compiled { package } =
        &bundle.manifest.code_closure
    else {
        panic!("compiled bundle retained a source project");
    };
    assert_eq!(package.package_instances.len(), 1);
    assert_eq!(
        package.graph_digest,
        bundle.manifest.project_revision.graph_digest
    );
    assert_eq!(
        package.source_digest,
        bundle.manifest.project_revision.source_digest
    );

    let mut archive = Vec::new();
    write_bundle(&bundle, &mut archive, ArchiveLimits::default()).unwrap();
    let decoded = read_bundle(archive.as_slice(), ArchiveLimits::default()).unwrap();
    assert_eq!(decoded, bundle);
    assert!(decoded
        .project_handoff_at(std::path::Path::new("unused"))
        .is_err());

    let parent_recipe = &decoded.manifest.recipes[0];
    let policy_object = decoded
        .objects
        .iter()
        .find(|object| object.descriptor.namespace == ObjectNamespace::ForeignPolicy)
        .unwrap();
    let foreign_objects: [&LogicalObject; 0] = [];
    let foreign_closures: [&ForeignArtifactClosure; 0] = [];
    assert!(PreparedForeignParentView::new(
        &parent_recipe.interop,
        None,
        None,
        &foreign_objects,
        &foreign_closures
    )
    .is_err());
    let parent = PreparedForeignParentView::new(
        &parent_recipe.interop,
        parent_recipe.foreign_policy.as_ref(),
        Some(policy_object),
        &foreign_objects,
        &foreign_closures,
    )
    .unwrap();
    let mut child = runmat_execution::ExecutableUnitEnvelope::from_canonical_bytes(
        &executable_unit_support::bytes(decoded.manifest.program_revision.clone()),
    )
    .unwrap();
    child.manifest.interop = runmat_types::InteropManifest::empty();
    let selected = SelectedPreparedForeignMaterial::for_compiled_unit(
        &child.canonical_bytes().unwrap(),
        &parent,
    )
    .unwrap();
    let (_, _, _, retained_policy) = selected.into_parts();
    assert!(std::ptr::eq(retained_policy.unwrap(), policy_object));
}

#[test]
fn foreign_artifacts_are_exact_first_class_bundle_objects() {
    let (_temp, project, revision) = support::frozen_project();
    let sidecar = LogicalObject::new(
        ObjectNamespace::ForeignArtifact,
        "foreign/fixture/interface.runmat.json",
        "application/vnd.runmat.native-interface+json",
        b"canonical sidecar".to_vec(),
    )
    .unwrap();
    let library = LogicalObject::new(
        ObjectNamespace::ForeignArtifact,
        "foreign/fixture/library.native",
        "application/vnd.runmat.native-library",
        b"synthetic native library".to_vec(),
    )
    .unwrap();
    let adapter = runmat_types::ForeignAdapterId::new("native-ffi").unwrap();
    let identity = runmat_types::ForeignArtifactIdentity::new("native-ffi:v1:fixture").unwrap();
    let mut recipe = support::recipe(revision.clone());
    recipe
        .interop
        .adapters
        .push(runmat_types::ForeignAdapterRequirement {
            adapter: adapter.clone(),
            minimum_version: 1,
            capabilities: runmat_types::CapabilitySet::default(),
            execution_stack: runmat_types::ExecutionStackRequirement::Process,
            artifact_identities: vec![identity.clone()],
        });
    recipe.interop.validate().unwrap();
    let mut legacy_foreign = recipe.clone();
    legacy_foreign.schema_version =
        runmat_execution_artifact::program::LEGACY_PROGRAM_BUILD_RECIPE_SCHEMA_VERSION;
    assert!(legacy_foreign.validate().is_err());
    let policy =
        foreign_policy_object(br#"{"fixture":"native-ffi","schema_version":1}"#.to_vec()).unwrap();
    recipe.foreign_policy = Some(policy.descriptor.clone());
    let closure = ForeignArtifactClosure::new(
        adapter,
        identity,
        vec![sidecar.descriptor.digest, library.descriptor.digest],
    )
    .unwrap();
    let bundle = ExecutionBundleBuilder::native(&project, revision.clone())
        .unwrap()
        .with_foreign_object(sidecar)
        .unwrap()
        .with_foreign_object(library)
        .unwrap()
        .with_foreign_policy_object(policy)
        .unwrap()
        .with_foreign_artifact_closure(closure)
        .unwrap()
        .with_materialized_program(
            recipe,
            ExecutableForm::InterpreterBytecodeV2,
            b"canonical-bytecode".to_vec(),
        )
        .build()
        .unwrap();

    assert_eq!(bundle.manifest.foreign_artifacts.len(), 2);
    assert_eq!(
        bundle
            .objects
            .iter()
            .filter(|object| object.descriptor.namespace == ObjectNamespace::ForeignArtifact)
            .count(),
        2
    );
    let mut archive = Vec::new();
    write_bundle(&bundle, &mut archive, ArchiveLimits::default()).unwrap();
    assert_eq!(
        read_bundle(archive.as_slice(), ArchiveLimits::default()).unwrap(),
        bundle
    );

    let mut missing_binding = bundle.clone();
    missing_binding.manifest.foreign_artifact_closures.clear();
    assert!(missing_binding.validate().is_err());

    let mut absent_object = bundle.clone();
    absent_object.manifest.foreign_artifact_closures[0].object_digests[0] =
        runmat_execution::Digest::sha256(b"absent foreign object");
    absent_object.manifest.foreign_artifact_closures[0]
        .object_digests
        .sort();
    assert!(absent_object.validate().is_err());

    let mut missing_policy = bundle.clone();
    missing_policy
        .objects
        .retain(|object| object.descriptor.namespace != ObjectNamespace::ForeignPolicy);
    assert!(missing_policy.validate().is_err());

    let mut substituted_policy = bundle.clone();
    let replacement =
        foreign_policy_object(br#"{"fixture":"replacement","schema_version":1}"#.to_vec()).unwrap();
    substituted_policy.manifest.recipes[0].foreign_policy = Some(replacement.descriptor.clone());
    assert!(substituted_policy.validate().is_err());

    let mut tampered_policy = bundle.clone();
    tampered_policy
        .objects
        .iter_mut()
        .find(|object| object.descriptor.namespace == ObjectNamespace::ForeignPolicy)
        .unwrap()
        .bytes
        .push(b' ');
    assert!(tampered_policy.validate().is_err());

    let mut tampered = bundle;
    let foreign = tampered
        .objects
        .iter_mut()
        .find(|object| object.descriptor.namespace == ObjectNamespace::ForeignArtifact)
        .unwrap();
    foreign.bytes.push(0);
    assert!(tampered.validate().is_err());
}

#[test]
fn compiled_package_closure_rejects_non_compiled_artifacts_after_decode() {
    let (_temp, project, revision) = support::frozen_project();
    let bytes = executable_unit_support::bytes(revision.clone());
    let policy =
        foreign_policy_object(br#"{"fixture":"compiled-reject","schema_version":1}"#.to_vec())
            .unwrap();
    let mut selected_recipe = executable_unit_support::recipe(support::recipe(revision.clone()));
    selected_recipe.foreign_policy = Some(policy.descriptor.clone());
    let mut bundle = ExecutionBundleBuilder::native(&project, revision.clone())
        .unwrap()
        .with_compiled_package_closure()
        .with_foreign_policy_object(policy)
        .unwrap()
        .with_materialized_program(selected_recipe, ExecutableForm::ExecutableUnitV4, bytes)
        .build()
        .unwrap();
    let recipe = bundle.manifest.recipes[0].clone();
    bundle.manifest.artifacts[0] = runmat_execution_artifact::ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::InterpreterBytecodeV2,
        b"legacy-bytecode".to_vec(),
    )
    .unwrap();
    assert!(bundle.validate().is_err());
}

#[path = "deterministic_bundle/environment.rs"]
mod environment;
