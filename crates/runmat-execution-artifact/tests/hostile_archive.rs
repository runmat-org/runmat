mod support;

use runmat_execution_artifact::{
    archive::{read_bundle, write_bundle, ArchiveLimits},
    ExecutableForm, ExecutionBundleBuilder,
};

fn archive() -> Vec<u8> {
    let (_temp, project, revision) = support::frozen_project();
    let bundle = ExecutionBundleBuilder::native(&project, revision.clone())
        .unwrap()
        .with_materialized_program(
            support::recipe(revision),
            ExecutableForm::InterpreterBytecodeV1,
            b"bytecode".to_vec(),
        )
        .build()
        .unwrap();
    let mut bytes = Vec::new();
    write_bundle(&bundle, &mut bytes, ArchiveLimits::default()).unwrap();
    bytes
}

#[test]
fn archive_rejects_trailing_tampered_and_oversized_data() {
    let mut trailing = archive();
    trailing.push(0);
    assert!(read_bundle(trailing.as_slice(), ArchiveLimits::default()).is_err());

    let mut tampered = archive();
    *tampered.last_mut().unwrap() ^= 1;
    assert!(read_bundle(tampered.as_slice(), ArchiveLimits::default()).is_err());

    let limits = ArchiveLimits {
        max_manifest_bytes: 1,
        ..ArchiveLimits::default()
    };
    assert!(read_bundle(archive().as_slice(), limits).is_err());
}

#[test]
fn logical_paths_cannot_escape_an_archive_namespace() {
    for name in [
        "../../secret",
        "C:\\secret",
        "source\\nested.m",
        "source//nested.m",
        "source/./nested.m",
    ] {
        assert!(
            runmat_execution_artifact::LogicalObject::new(
                runmat_execution_artifact::ObjectNamespace::ProgramSource,
                name,
                "application/octet-stream",
                Vec::new(),
            )
            .is_err(),
            "{name:?} must not be accepted as a canonical logical path"
        );
    }
}
