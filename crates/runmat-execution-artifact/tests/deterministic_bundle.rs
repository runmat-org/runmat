#[path = "support/executable_unit.rs"]
mod executable_unit_support;
mod support;

use runmat_execution_artifact::{
    archive::{read_bundle, write_bundle, ArchiveLimits},
    ExecutableForm, ExecutionBundleBuilder,
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
            ExecutableForm::InterpreterBytecodeV1,
            b"canonical-bytecode".to_vec(),
        )
        .build()
        .unwrap();
    let second = ExecutionBundleBuilder::native(&second, second_revision.clone())
        .unwrap()
        .with_materialized_program(
            support::recipe(second_revision),
            ExecutableForm::InterpreterBytecodeV1,
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
    let bundle = ExecutionBundleBuilder::native(&project, revision.clone())
        .unwrap()
        .with_materialized_program(
            executable_unit_support::recipe(support::recipe(revision)),
            ExecutableForm::ExecutableUnitV3,
            bytes.clone(),
        )
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
fn compiled_package_closure_round_trips_without_source_or_project_payloads() {
    let (_temp, project, revision) = support::frozen_project();
    let bytes = executable_unit_support::bytes(revision.clone());
    let bundle = ExecutionBundleBuilder::native(&project, revision.clone())
        .unwrap()
        .with_compiled_package_closure()
        .with_materialized_program(
            executable_unit_support::recipe(support::recipe(revision)),
            ExecutableForm::ExecutableUnitV3,
            bytes,
        )
        .build()
        .unwrap();

    assert!(bundle.objects.is_empty());
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
}

#[test]
fn compiled_package_closure_rejects_non_compiled_artifacts_after_decode() {
    let (_temp, project, revision) = support::frozen_project();
    let bytes = executable_unit_support::bytes(revision.clone());
    let mut bundle = ExecutionBundleBuilder::native(&project, revision.clone())
        .unwrap()
        .with_compiled_package_closure()
        .with_materialized_program(
            executable_unit_support::recipe(support::recipe(revision)),
            ExecutableForm::ExecutableUnitV3,
            bytes,
        )
        .build()
        .unwrap();
    let recipe = bundle.manifest.recipes[0].clone();
    bundle.manifest.artifacts[0] = runmat_execution_artifact::ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::InterpreterBytecodeV1,
        b"legacy-bytecode".to_vec(),
    )
    .unwrap();
    assert!(bundle.validate().is_err());
}
