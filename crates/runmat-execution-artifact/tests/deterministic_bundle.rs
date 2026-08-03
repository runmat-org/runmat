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
