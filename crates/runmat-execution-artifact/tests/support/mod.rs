use std::collections::BTreeSet;
use std::fs;

use runmat_execution::{Digest, OutputContract, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::ProgramBuildRecipe;
use runmat_package::{build_frozen_project, FrozenProject};
use tempfile::TempDir;

pub fn frozen_project() -> (TempDir, FrozenProject, ProgramRevision) {
    let temp = TempDir::new().unwrap();
    fs::create_dir_all(temp.path().join("src/nested")).unwrap();
    fs::write(
        temp.path().join("runmat.toml"),
        "[package]\nname = \"artifact-fixture\"\nversion = \"1.0.0\"\n\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    fs::write(
        temp.path().join("src/main.m"),
        "function y = main(x)\ny = helper(x);\nend\n",
    )
    .unwrap();
    fs::write(
        temp.path().join("src/nested/helper.m"),
        "function y = helper(x)\ny = x + 1;\nend\n",
    )
    .unwrap();
    fs::write(temp.path().join("src/empty.m"), "").unwrap();
    let project = build_frozen_project(&temp.path().join("runmat.toml"), BTreeSet::new()).unwrap();
    let revision = revision_for(&project);
    (temp, project, revision)
}

pub fn revision_for(project: &FrozenProject) -> ProgramRevision {
    ProgramRevision::new(
        Digest::from_bytes(*project.graph_digest().bytes()),
        Digest::from_bytes(*project.source_revision().bytes()),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"runtime-v1"),
            Digest::sha256(b"catalog-v1"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
}

pub fn recipe(revision: ProgramRevision) -> ProgramBuildRecipe {
    ProgramBuildRecipe {
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: revision,
        entrypoint: "main".into(),
        outputs: OutputContract {
            requested_outputs: 1,
        },
        execution_mode: "interpreter".into(),
        target: runmat_execution_artifact::ProgramTarget::portable("portable-bytecode-v1"),
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    }
}
