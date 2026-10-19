use super::*;

#[test]
fn frozen_environment_is_identity_bound_and_preserved_by_cold_archive() {
    let (_root, project, revision) = support::frozen_project();
    let build = |value: &str| {
        ExecutionBundleBuilder::native(&project, revision.clone())
            .unwrap()
            .with_materialized_program(
                support::recipe(revision.clone()),
                ExecutableForm::InterpreterBytecodeV1,
                b"canonical-bytecode".to_vec(),
            )
            .with_portable_environment(vec![("RUNMAT_MODE".into(), value.into())])
            .unwrap()
            .build()
            .unwrap()
    };
    let original = build("first");
    assert_ne!(
        original.identity().unwrap(),
        build("second").identity().unwrap()
    );
    let mut bytes = Vec::new();
    write_bundle(&original, &mut bytes, ArchiveLimits::default()).unwrap();
    let cold = read_bundle(&bytes[..], ArchiveLimits::default()).unwrap();
    assert_eq!(cold, original);
    assert_eq!(
        cold.manifest.portable_environment,
        [("RUNMAT_MODE".into(), "first".into())]
    );
    let mut modified = original;
    modified
        .manifest
        .portable_environment
        .push(("RUNMAT_MODE".into(), "duplicate".into()));
    assert!(modified.validate().is_err());
}
