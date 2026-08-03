mod support;

use runmat_execution_artifact::{ExecutableForm, ExecutionBundleBuilder, ProgramArtifact};

#[test]
fn recipe_and_materialized_artifact_have_distinct_exact_identities() {
    let (_temp, project, revision) = support::frozen_project();
    let bundle = ExecutionBundleBuilder::native(&project, revision.clone())
        .unwrap()
        .with_materialized_program(
            support::recipe(revision),
            ExecutableForm::InterpreterBytecodeV1,
            b"bytecode-a".to_vec(),
        )
        .build()
        .unwrap();
    let recipe = &bundle.manifest.recipes[0];
    let artifact = &bundle.manifest.artifacts[0];
    assert_ne!(recipe.id().unwrap().0, artifact.id.0);
    artifact.validate_against(recipe).unwrap();

    let changed = ProgramArtifact::materialize(
        recipe,
        ExecutableForm::InterpreterBytecodeV1,
        b"bytecode-b".to_vec(),
    )
    .unwrap();
    assert_ne!(changed.id, artifact.id);
}

#[test]
fn artifact_tampering_and_revision_mismatch_are_rejected() {
    let (_temp, project, revision) = support::frozen_project();
    let mut bundle = ExecutionBundleBuilder::native(&project, revision.clone())
        .unwrap()
        .with_materialized_program(
            support::recipe(revision),
            ExecutableForm::InterpreterBytecodeV1,
            b"bytecode".to_vec(),
        )
        .build()
        .unwrap();
    bundle.manifest.artifacts[0].executable_bytes.push(0);
    assert!(bundle.validate().is_err());

    let mut wrong_revision = support::recipe(support::revision_for(&project));
    wrong_revision.program_revision = runmat_execution::ProgramRevision::new(
        runmat_execution::Digest::sha256(b"wrong"),
        wrong_revision.program_revision.source_digest().to_owned(),
        wrong_revision.program_revision.environment(),
    )
    .unwrap();
    assert!(
        ExecutionBundleBuilder::native(&project, support::revision_for(&project))
            .unwrap()
            .with_recipe(wrong_revision)
            .build()
            .is_err()
    );
}
